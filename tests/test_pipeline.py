"""Comprehensive test suite for the News-to-Prompt pipeline.

Tests cover:
- Database schema initialization and seeding
- Ticker disambiguation and keyword generation
- Text cleaning and context cropping
- Regex matching engine (two-stage)
- Taxonomy classification with portfolio override
- Prompt building and persistence
- Full end-to-end integration
"""

import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    AMBIGUOUS_SYMBOLS,
    FINANCIAL_CONTEXT_WORDS,
    LEGAL_SUFFIXES_PATTERN,
)
from database.db_manager import DatabaseManager
from database.seeder import (
    _clean_company_name,
    _extract_base_symbol,
    _generate_keywords,
    _is_short_or_ambiguous,
    seed_all,
    seed_news_sources,
    seed_watchlist,
)
from filtering.regex_engine import RegexEngine
from filtering.taxonomy_classifier import TaxonomyClassifier
from ingestion.rss_poller import LRUDeduplicator, RSSPoller, compute_url_hash
from ingestion.text_cleaner import (
    clean_text,
    crop_context,
    normalize_unicode,
    remove_boilerplate,
    split_sentences,
    strip_html,
)
from models.schemas import (
    FilteredNewsItem,
    PreparedPromptPayload,
    RawNewsItem,
    TargetLLMResponseSchema,
    TickerMatch,
)
from prompt_builder.builder import PromptBuilder
from prompt_builder.templates import (
    format_market_context,
    format_portfolio_context,
    get_system_prompt,
    get_target_json_schema,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_db_singleton():
    """Reset the DatabaseManager singleton before each test."""
    DatabaseManager.reset()
    yield
    DatabaseManager.reset()


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database with full schema."""
    db_path = tmp_path / "test_trading.db"
    DatabaseManager.reset()
    db = DatabaseManager(db_path)
    db.initialize_schema()
    return db


@pytest.fixture
def seeded_db(tmp_db):
    """Database seeded with companies and news sources."""
    seed_watchlist(tmp_db)
    seed_news_sources(tmp_db)
    return tmp_db


# ═══════════════════════════════════════════════════════════════
# 1. DATABASE TESTS
# ═══════════════════════════════════════════════════════════════

class TestDatabaseManager:
    """Tests for the database manager and schema initialization."""

    def test_schema_creates_all_tables(self, tmp_db):
        """Verify all 5 tables exist after schema init."""
        tables = tmp_db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {row['name'] for row in tables}
        expected = {'watchlist', 'keywords', 'news_sources', 'news_archive', 'prepared_prompts'}
        assert expected.issubset(table_names), f"Missing tables: {expected - table_names}"

    def test_wal_mode_active(self, tmp_db):
        """Verify WAL journal mode is active."""
        row = tmp_db.fetch_one("PRAGMA journal_mode;")
        assert row[0] == 'wal'

    def test_foreign_keys_enabled(self, tmp_db):
        """Verify foreign keys are enforced."""
        row = tmp_db.fetch_one("PRAGMA foreign_keys;")
        assert row[0] == 1

    def test_indices_created(self, tmp_db):
        """Verify B-Tree indices exist."""
        indices = tmp_db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        idx_names = {row['name'] for row in indices}
        expected_indices = {
            'idx_keywords_ticker', 'idx_keywords_match_type',
            'idx_news_archive_url_hash', 'idx_news_archive_source_id',
            'idx_news_archive_priority', 'idx_news_archive_ingested',
            'idx_prepared_prompts_status', 'idx_prepared_prompts_priority',
            'idx_watchlist_portfolio', 'idx_watchlist_active',
        }
        assert expected_indices.issubset(idx_names), f"Missing indices: {expected_indices - idx_names}"

    def test_execute_and_fetch(self, tmp_db):
        """Test basic CRUD operations."""
        tmp_db.execute(
            "INSERT INTO watchlist (ticker, base_symbol, company_name, clean_name, sector, index_country) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ('TEST.MI', 'TEST', 'Test Company SpA', 'Test Company', 'Tech', 'IT_MID')
        )
        row = tmp_db.fetch_one("SELECT * FROM watchlist WHERE ticker = ?", ('TEST.MI',))
        assert row is not None
        assert row['company_name'] == 'Test Company SpA'
        assert row['is_portfolio_open'] == 0
        assert row['is_active'] == 1


# ═══════════════════════════════════════════════════════════════
# 2. SEEDER TESTS
# ═══════════════════════════════════════════════════════════════

class TestSeeder:
    """Tests for the data seeder and keyword generation."""

    def test_clean_company_name_removes_suffixes(self):
        """Test legal suffix removal from company names."""
        assert _clean_company_name("Interpump Group") == "Interpump"
        assert _clean_company_name("Reply SpA") == "Reply"
        assert _clean_company_name("Deutsche Lufthansa") == "Deutsche Lufthansa"
        assert _clean_company_name("freenet AG") == "freenet"
        assert _clean_company_name("Nordex SE") == "Nordex"
        assert _clean_company_name("Signify NV") == "Signify"

    def test_clean_name_preserves_short_names(self):
        """If cleaned name < 3 chars, keep original."""
        result = _clean_company_name("B.F. SpA")
        assert len(result) >= 3  # Should keep original or enough chars

    def test_extract_base_symbol(self):
        """Test base symbol extraction from full tickers."""
        assert _extract_base_symbol("IP.MI") == "IP"
        assert _extract_base_symbol("LHA.DE") == "LHA"
        assert _extract_base_symbol("EMBRAC-B.ST") == "EMBRAC"
        assert _extract_base_symbol("BGN.MI") == "BGN"

    def test_is_short_or_ambiguous(self):
        """Test ambiguity detection for tickers."""
        assert _is_short_or_ambiguous("IP") is True   # <= 3 chars
        assert _is_short_or_ambiguous("AM") is True    # <= 3 and in blacklist
        assert _is_short_or_ambiguous("SOL") is True   # In blacklist
        assert _is_short_or_ambiguous("AIXA") is False  # 4 chars, not in blacklist
        assert _is_short_or_ambiguous("POST") is True  # In blacklist

    def test_keyword_generation_short_ticker(self):
        """Short tickers should NOT get SAFE_LONG_TICKER patterns."""
        keywords = _generate_keywords("IP.MI", "IP", "Interpump")
        match_types = {kw[2] for kw in keywords}
        assert 'COMPANY_NAME' in match_types
        assert 'FULL_TICKER' in match_types
        assert 'EXPLICIT_SYMBOL' in match_types
        assert 'SAFE_LONG_TICKER' not in match_types

    def test_keyword_generation_long_safe_ticker(self):
        """Long non-ambiguous tickers SHOULD get SAFE_LONG_TICKER."""
        keywords = _generate_keywords("AIXA.DE", "AIXA", "Aixtron")
        match_types = {kw[2] for kw in keywords}
        assert 'SAFE_LONG_TICKER' in match_types

    def test_keyword_generation_ambiguous_long_ticker(self):
        """Long but ambiguous tickers should NOT get SAFE_LONG_TICKER."""
        keywords = _generate_keywords("POST.VI", "POST", "Oesterreichische Post")
        match_types = {kw[2] for kw in keywords}
        assert 'SAFE_LONG_TICKER' not in match_types

    def test_seed_watchlist_populates_data(self, tmp_db):
        """Seed should populate watchlist with 500+ companies."""
        count = seed_watchlist(tmp_db)
        assert count >= 500, f"Expected 500+ companies, got {count}"

        # Verify some known entries
        row = tmp_db.fetch_one("SELECT * FROM watchlist WHERE ticker = 'IP.MI'")
        assert row is not None
        assert row['base_symbol'] == 'IP'
        assert 'Interpump' in row['company_name']

    def test_seed_keywords_generated(self, tmp_db):
        """Seeding should create keyword patterns."""
        seed_watchlist(tmp_db)
        count = tmp_db.fetch_one("SELECT COUNT(*) as cnt FROM keywords")['cnt']
        # At least 3 keywords per company (COMPANY_NAME, FULL_TICKER, EXPLICIT_SYMBOL)
        assert count >= 500 * 3, f"Expected 1500+ keywords, got {count}"

    def test_seed_news_sources(self, tmp_db):
        """Seed should populate news sources from file."""
        count = seed_news_sources(tmp_db)
        assert count >= 30, f"Expected 30+ sources, got {count}"

        row = tmp_db.fetch_one("SELECT * FROM news_sources WHERE source_id = 'GNW_MNA'")
        assert row is not None
        assert row['feed_type'] == 'PR_WIRE'
        assert row['language'] == 'en'

    def test_full_seed(self, tmp_db):
        """Test complete seeding pipeline."""
        result = seed_all(tmp_db)
        assert result['companies'] >= 500
        assert result['keywords'] >= 1500
        assert result['sources'] >= 30


# ═══════════════════════════════════════════════════════════════
# 3. TEXT CLEANER TESTS
# ═══════════════════════════════════════════════════════════════

class TestTextCleaner:
    """Tests for HTML stripping, Unicode normalization, and context cropping."""

    def test_strip_html_tags(self):
        """Remove all HTML tags from text."""
        html_text = '<p>Interpump <b>acquires</b> <a href="#">target</a> for €500M</p>'
        result = strip_html(html_text)
        assert '<' not in result
        assert 'Interpump' in result
        assert '€500M' in result

    def test_html_entity_decoding(self):
        """Decode HTML entities like &amp; &euro;."""
        text = 'Revenue &amp; profit grew by 15% to &euro;100M'
        result = strip_html(text)
        assert '&amp;' not in result
        assert '&' in result or 'Revenue' in result

    def test_normalize_unicode(self):
        """Test Unicode normalization of special characters."""
        text = 'Evonik\u2019s Q3 \u201cresults\u201d beat \u2013 expectations'
        result = normalize_unicode(text)
        assert '\u2019' not in result
        assert "'" in result
        assert '-' in result

    def test_remove_boilerplate(self):
        """Remove news agency suffixes from text."""
        text = 'Interpump acquires competitor - Reuters'
        result = remove_boilerplate(text)
        assert 'Reuters' not in result
        assert 'Interpump' in result

    def test_clean_text_full_pipeline(self):
        """Test the complete cleaning pipeline."""
        raw = '<p>Interpump <b>acquires</b> rival &amp; grows revenue - MilanoFinanza</p>'
        result = clean_text(raw)
        assert '<' not in result
        assert 'MilanoFinanza' not in result
        assert 'Interpump' in result

    def test_split_sentences(self):
        """Test sentence splitting."""
        text = "Interpump reported Q3 results. Revenue grew 15%. EBITDA margin expanded to 22%."
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_crop_context_short_text(self):
        """Text shorter than max_chars should be returned as-is."""
        text = "Short text about Interpump Q3 results."
        result = crop_context(text, ["Interpump"], max_chars=1000)
        assert result == text

    def test_crop_context_long_text(self):
        """Long text should be cropped around match terms."""
        sentences = [
            "The European market opened higher today.",
            "Several sectors showed gains in early trading.",
            "Interpump Group reported record Q3 revenue of €500M.",
            "EBITDA margin expanded to 22.5% from 20.1%.",
            "The company confirmed its full-year guidance.",
            "Other industrial stocks followed the positive trend.",
            "Analysts expect continued growth in the sector.",
            "Central bank policy remains accommodative.",
            "Bond yields stayed flat across major European markets.",
            "Commodity prices showed mixed signals.",
        ]
        text = " ".join(sentences)
        result = crop_context(text, ["Interpump"], max_chars=500)
        assert "Interpump" in result
        assert len(result) <= 500

    def test_crop_context_preserves_window(self):
        """Cropped context should include sentences around the match."""
        sentences = [
            "Sentence one about markets.",
            "Sentence two about indices.",
            "Sentence three mentioning Interpump acquisition.",
            "Sentence four with financial details.",
            "Sentence five about outlook.",
            "Sentence six unrelated.",
            "Sentence seven unrelated.",
        ]
        text = " ".join(sentences)
        result = crop_context(text, ["Interpump"], max_chars=1000, window=2)
        assert "Interpump" in result


# ═══════════════════════════════════════════════════════════════
# 4. RSS POLLER TESTS
# ═══════════════════════════════════════════════════════════════

class TestRSSPoller:
    """Tests for deduplication, hashing, and portfolio feed generation."""

    def test_lru_deduplicator_basic(self):
        """Test LRU dedup basic operations."""
        dedup = LRUDeduplicator(maxsize=3)
        assert not dedup.contains("a")
        dedup.add("a")
        assert dedup.contains("a")
        dedup.add("b")
        dedup.add("c")
        dedup.add("d")  # Should evict "a"
        assert not dedup.contains("a")
        assert dedup.contains("d")

    def test_lru_deduplicator_touch_on_contains(self):
        """Accessing an item should move it to end (prevent eviction)."""
        dedup = LRUDeduplicator(maxsize=3)
        dedup.add("a")
        dedup.add("b")
        dedup.add("c")
        dedup.contains("a")  # Touch "a"
        dedup.add("d")  # Should evict "b" not "a"
        assert dedup.contains("a")
        assert not dedup.contains("b")

    def test_compute_url_hash_regular(self):
        """Regular URLs should hash on the link."""
        h1 = compute_url_hash("Title", "https://example.com/article/123")
        h2 = compute_url_hash("Different Title", "https://example.com/article/123")
        assert h1 == h2  # Same link -> same hash

    def test_compute_url_hash_google_news(self):
        """Google News URLs should hash on normalized title."""
        h1 = compute_url_hash(
            "Interpump Q3 Results",
            "https://news.google.com/rss/articles/token123456"
        )
        h2 = compute_url_hash(
            "Interpump Q3 Results",
            "https://news.google.com/rss/articles/different_token"
        )
        assert h1 == h2  # Same title -> same hash for Google News

    def test_portfolio_feed_generation(self, seeded_db):
        """Test dynamic portfolio feed generation."""
        # Set some companies as portfolio positions
        seeded_db.execute(
            "UPDATE watchlist SET is_portfolio_open = 1, portfolio_avg_price = 45.0, "
            "portfolio_allocation_pct = 5.0 WHERE ticker = 'IP.MI'"
        )
        seeded_db.execute(
            "UPDATE watchlist SET is_portfolio_open = 1, portfolio_avg_price = 130.0, "
            "portfolio_allocation_pct = 3.0 WHERE ticker = 'FCT.MI'"
        )

        poller = RSSPoller(db=seeded_db)
        feeds = poller.generate_portfolio_priority_feeds()
        assert len(feeds) >= 1
        assert feeds[0]['feed_type'] == 'DYNAMIC_PORTFOLIO'
        assert 'news.google.com' in feeds[0]['url']


# ═══════════════════════════════════════════════════════════════
# 5. REGEX ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestRegexEngine:
    """Tests for the two-stage regex matching engine."""

    def test_match_by_company_name(self, seeded_db):
        """Match via company name pattern."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        matches = engine.match("Interpump Group reported record Q3 revenue of €500M")
        tickers = {m.ticker for m in matches}
        assert 'IP.MI' in tickers

    def test_match_by_full_ticker(self, seeded_db):
        """Match via full ticker pattern."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        matches = engine.match("Shares of IP.MI surged 5% after the announcement")
        tickers = {m.ticker for m in matches}
        assert 'IP.MI' in tickers

    def test_match_explicit_symbol(self, seeded_db):
        """Match via exchange:symbol pattern."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        matches = engine.match("BIT:IP reported Q3 results with strong revenue growth")
        tickers = {m.ticker for m in matches}
        assert 'IP.MI' in tickers

    def test_no_false_positive_short_ticker(self, seeded_db):
        """Short tickers like IP should NOT match when used as a regular word."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        # "IP" as a word (Internet Protocol) should not trigger a match
        matches = engine.match("The IP address was configured correctly in the network setup")
        tickers = {m.ticker for m in matches}
        assert 'IP.MI' not in tickers

    def test_financial_context_required(self, seeded_db):
        """Patterns requiring financial context should validate it."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        # With financial context -> should match
        matches_with_ctx = engine.match(
            "Reply reported strong EBITDA growth of 15% in Q3, shares surged"
        )
        tickers_with = {m.ticker for m in matches_with_ctx}

        # Without financial context -> should NOT match on name alone
        matches_no_ctx = engine.match(
            "Please reply to this email as soon as possible"
        )
        tickers_no = {m.ticker for m in matches_no_ctx}
        assert 'REY.MI' not in tickers_no

    def test_multiple_tickers_in_one_text(self, seeded_db):
        """Multiple companies in one article should all be detected."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        text = (
            "Interpump Group and Fincantieri both reported Q3 results today. "
            "Interpump's EBITDA margin expanded while Fincantieri secured a new naval contract."
        )
        matches = engine.match(text)
        tickers = {m.ticker for m in matches}
        assert 'IP.MI' in tickers
        assert 'FCT.MI' in tickers

    def test_safe_long_ticker_requires_context(self, seeded_db):
        """SAFE_LONG_TICKER matches should require financial context."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        # AIXA (Aixtron) - long, non-ambiguous
        # With financial context
        matches = engine.match("AIXA reported record revenue of €600M for fiscal year 2024")
        tickers = {m.ticker for m in matches}
        assert 'AIXA.DE' in tickers

    def test_rules_loaded_count(self, seeded_db):
        """Loading rules should return a positive count."""
        engine = RegexEngine(db=seeded_db)
        count = engine.load_rules()
        assert count >= 1500  # At least 3 rules per 500+ companies


# ═══════════════════════════════════════════════════════════════
# 6. TAXONOMY CLASSIFIER TESTS
# ═══════════════════════════════════════════════════════════════

class TestTaxonomyClassifier:
    """Tests for multilingual taxonomy classification."""

    def test_ma_classification_english(self, seeded_db):
        """M&A keywords in English should classify as M&A_CORPORATE, Priority 1."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Interpump launches takeover bid for competitor at 30% premium", ["IP.MI"]
        )
        assert category == "M&A_CORPORATE"
        assert priority == 1

    def test_ma_classification_italian(self, seeded_db):
        """M&A keywords in Italian."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Interpump annuncia acquisizione di un concorrente per €200M", ["IP.MI"]
        )
        assert category == "M&A_CORPORATE"
        assert priority == 1

    def test_earnings_classification(self, seeded_db):
        """Earnings keywords should classify as EARNINGS_GUIDANCE, Priority 1."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Q3 net profit surged 25% as EBITDA margin expanded to record levels", ["IP.MI"]
        )
        assert category == "EARNINGS_GUIDANCE"
        assert priority == 1

    def test_contract_classification(self, seeded_db):
        """Contract keywords should classify as CONTRACT_ORDER, Priority 2."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Fincantieri awarded major naval contract worth €1.2 billion", ["FCT.MI"]
        )
        assert category == "CONTRACT_ORDER"
        assert priority == 2

    def test_regulatory_classification(self, seeded_db):
        """Regulatory keywords should classify as REGULATORY_LEGAL, Priority 2."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Consob launches investigation into insider trading allegations", ["IP.MI"]
        )
        assert category == "REGULATORY_LEGAL"
        assert priority == 2

    def test_analyst_classification(self, seeded_db):
        """Analyst rating keywords should classify as ANALYST_RATING, Priority 3."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Morgan Stanley upgrades Interpump with target price of €55", ["IP.MI"]
        )
        assert category == "ANALYST_RATING"
        assert priority == 3

    def test_general_classification(self, seeded_db):
        """Unmatched text should classify as GENERAL, Priority 3."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Italian industrial sector continues to show resilience", ["IP.MI"]
        )
        assert category == "GENERAL"
        assert priority == 3

    def test_portfolio_override(self, seeded_db):
        """Portfolio positions should force Priority 1."""
        # Mark IP.MI as portfolio position
        seeded_db.execute(
            "UPDATE watchlist SET is_portfolio_open = 1 WHERE ticker = 'IP.MI'"
        )
        classifier = TaxonomyClassifier(db=seeded_db)
        # This would normally be Priority 3
        category, priority = classifier.classify(
            "Italian industrial sector continues to show resilience", ["IP.MI"]
        )
        assert priority == 1  # Overridden to 1

    def test_german_classification(self, seeded_db):
        """German keywords should work."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Lufthansa Übernahme des Konkurrenten für €5 Mrd.", ["LHA.DE"]
        )
        assert category == "M&A_CORPORATE"
        assert priority == 1

    def test_french_classification(self, seeded_db):
        """French keywords should work."""
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(
            "Résultats du troisième trimestre supérieurs aux attentes", ["AF.PA"]
        )
        assert category == "EARNINGS_GUIDANCE"
        assert priority == 1


# ═══════════════════════════════════════════════════════════════
# 7. PYDANTIC MODELS TESTS
# ═══════════════════════════════════════════════════════════════

class TestPydanticModels:
    """Tests for Pydantic v2 model validation."""

    def test_raw_news_item(self):
        """Test RawNewsItem creation and validation."""
        item = RawNewsItem(
            source_id="GNW_MNA",
            title="Test News",
            link="https://example.com/news/1",
            url_hash="abc123",
        )
        assert item.source_id == "GNW_MNA"
        assert item.published_at is None

    def test_filtered_news_item(self):
        """Test FilteredNewsItem with all fields."""
        item = FilteredNewsItem(
            source_id="GNW_MNA",
            title="Interpump acquires competitor",
            link="https://example.com/news/1",
            url_hash="abc123",
            cropped_context="Interpump Group announced acquisition...",
            matched_tickers=[
                TickerMatch(ticker="IP.MI", matched_text="Interpump", match_type="COMPANY_NAME")
            ],
            matched_ticker_symbols=["IP.MI"],
            event_category="M&A_CORPORATE",
            priority_level=1,
        )
        assert item.priority_level == 1
        assert len(item.matched_tickers) == 1

    def test_target_llm_schema_generation(self):
        """Test JSON Schema generation from TargetLLMResponseSchema."""
        schema_str = PreparedPromptPayload.generate_target_schema()
        schema = json.loads(schema_str)
        assert 'properties' in schema
        assert 'event_type' in schema['properties']
        assert 'impacted_assets' in schema['properties']

    def test_token_estimation(self):
        """Test rough token count estimation."""
        tokens = PreparedPromptPayload.estimate_tokens(
            "System prompt" * 100,
            "User prompt" * 100,
            "Schema" * 100,
        )
        assert tokens > 0
        # len("System prompt")*100 + len("User prompt")*100 + len("Schema")*100 = 1300+1100+600 = 3000
        # 3000 // 4 = 750
        assert tokens == (len("System prompt" * 100) + len("User prompt" * 100) + len("Schema" * 100)) // 4


# ═══════════════════════════════════════════════════════════════
# 8. PROMPT BUILDER TESTS
# ═══════════════════════════════════════════════════════════════

class TestPromptBuilder:
    """Tests for prompt building and database persistence."""

    def test_system_prompt_not_empty(self):
        """System prompt should be substantial."""
        prompt = get_system_prompt()
        assert len(prompt) > 500
        assert "European Small and Mid Cap" in prompt

    def test_json_schema_valid(self):
        """Generated JSON schema should be valid JSON."""
        schema_str = get_target_json_schema()
        schema = json.loads(schema_str)
        assert isinstance(schema, dict)
        assert 'properties' in schema

    def test_market_context(self):
        """Market context should contain relevant information."""
        ctx = format_market_context()
        assert "European" in ctx or "FTSE" in ctx or "DAX" in ctx

    def test_portfolio_context_empty(self):
        """Empty portfolio should return appropriate message."""
        ctx = format_portfolio_context([])
        assert "No open positions" in ctx

    def test_portfolio_context_with_positions(self):
        """Portfolio context should format positions correctly."""
        positions = [
            {'ticker': 'IP.MI', 'company_name': 'Interpump Group', 'avg_price': 45.0, 'allocation_pct': 5.0}
        ]
        ctx = format_portfolio_context(positions)
        assert 'IP.MI' in ctx
        assert 'Interpump' in ctx

    def test_build_and_persist_prompt(self, seeded_db):
        """Full prompt build + DB persist cycle."""
        builder = PromptBuilder(db=seeded_db)

        # Create a filtered news item
        news = FilteredNewsItem(
            source_id="GNW_MNA",
            title="Interpump Group acquires German competitor for €200M",
            link="https://example.com/news/interpump-acquisition",
            published_at="2024-01-15T09:30:00Z",
            url_hash="test_hash_12345",
            cropped_context=(
                "Interpump Group SpA announced today the acquisition of a leading German "
                "hydraulic components manufacturer for approximately €200 million. The deal "
                "is expected to strengthen Interpump's position in the European market. "
                "The board unanimously approved the transaction."
            ),
            matched_tickers=[
                TickerMatch(ticker="IP.MI", matched_text="Interpump Group", match_type="COMPANY_NAME"),
            ],
            matched_ticker_symbols=["IP.MI"],
            event_category="M&A_CORPORATE",
            priority_level=1,
        )

        # First, insert into news_archive
        cursor = seeded_db.execute(
            """INSERT INTO news_archive 
               (url_hash, source_id, title, link, published_at, raw_summary, 
                cropped_context, matched_tickers_json, event_category, priority_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                news.url_hash, news.source_id, news.title, news.link,
                news.published_at, news.raw_summary or '',
                news.cropped_context,
                json.dumps(news.matched_ticker_symbols),
                news.event_category, news.priority_level,
            )
        )
        news_id = cursor.lastrowid

        # Build and persist the prompt
        prompt_id = builder.process_news_item(news, news_id)
        assert prompt_id > 0

        # Verify it's in the database
        row = seeded_db.fetch_one(
            "SELECT * FROM prepared_prompts WHERE id = ?", (prompt_id,)
        )
        assert row is not None
        assert row['status'] == 'READY_FOR_LLM'
        assert row['priority_level'] == 1
        assert row['news_id'] == news_id
        assert len(row['system_prompt']) > 500
        assert 'Interpump' in row['user_prompt']
        assert row['estimated_tokens'] > 0

        # Verify JSON schema is valid
        schema = json.loads(row['expected_json_schema'])
        assert 'properties' in schema


# ═══════════════════════════════════════════════════════════════
# 9. FULL INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════

class TestFullPipelineIntegration:
    """End-to-end integration test of the entire pipeline."""

    def test_full_pipeline_ma_news(self, seeded_db):
        """Test complete pipeline: seed -> clean -> match -> classify -> build prompt -> persist."""
        # 1. Text Cleaning
        raw_html = (
            '<p><b>Interpump Group SpA</b> (BIT:IP) announced a definitive agreement '
            'to acquire a leading German hydraulic manufacturer for approximately '
            '<span>€200 million</span> in cash. The transaction, unanimously approved '
            'by the board, is expected to be accretive to EBITDA from day one.</p>'
            '<p>The target company generated €85M in revenue and €18M EBITDA '
            'in the last fiscal year. Closing is expected in Q2 2024 subject to '
            'regulatory approvals.</p> - Reuters'
        )
        cleaned = clean_text(raw_html)
        assert '<' not in cleaned
        assert 'Reuters' not in cleaned
        assert 'Interpump' in cleaned

        # 2. Regex Matching
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        combined_text = f"Interpump Group SpA acquires German rival for €200M. {cleaned}"
        matches = engine.match(combined_text)
        tickers = {m.ticker for m in matches}
        assert 'IP.MI' in tickers, f"IP.MI not found in {tickers}"

        # 3. Context Cropping
        match_terms = [m.matched_text for m in matches]
        cropped = crop_context(cleaned, match_terms, max_chars=1000)
        assert len(cropped) <= 1000
        assert 'Interpump' in cropped

        # 4. Taxonomy Classification
        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(combined_text, list(tickers))
        assert category == "M&A_CORPORATE"
        assert priority == 1

        # 5. Build FilteredNewsItem
        filtered = FilteredNewsItem(
            source_id="GNW_MNA",
            title="Interpump Group acquires German competitor",
            link="https://example.com/interpump-ma",
            published_at="2024-01-15T09:30:00Z",
            url_hash=compute_url_hash("Interpump Group acquires German competitor",
                                       "https://example.com/interpump-ma"),
            cropped_context=cropped,
            matched_tickers=matches,
            matched_ticker_symbols=list(tickers),
            event_category=category,
            priority_level=priority,
        )

        # 6. Persist to news_archive
        cursor = seeded_db.execute(
            """INSERT INTO news_archive 
               (url_hash, source_id, title, link, published_at, raw_summary,
                cropped_context, matched_tickers_json, event_category, priority_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filtered.url_hash, filtered.source_id, filtered.title,
                filtered.link, filtered.published_at, '',
                filtered.cropped_context,
                json.dumps(filtered.matched_ticker_symbols),
                filtered.event_category, filtered.priority_level,
            )
        )
        news_id = cursor.lastrowid
        assert news_id > 0

        # 7. Build and persist prompt
        builder = PromptBuilder(db=seeded_db)
        prompt_id = builder.process_news_item(filtered, news_id)
        assert prompt_id > 0

        # 8. Verify final state in prepared_prompts
        row = seeded_db.fetch_one(
            "SELECT * FROM prepared_prompts WHERE id = ?", (prompt_id,)
        )
        assert row is not None
        assert row['status'] == 'READY_FOR_LLM'
        assert row['priority_level'] == 1
        assert row['estimated_tokens'] > 100
        assert 'Interpump' in row['user_prompt']
        assert 'M&A_CORPORATE' in row['user_prompt']

        # Verify the JSON schema is embedded
        schema = json.loads(row['expected_json_schema'])
        assert 'impacted_assets' in schema['properties']

        print(f"\n[PASS] FULL PIPELINE INTEGRATION TEST PASSED")
        print(f"   News Archive ID: {news_id}")
        print(f"   Prepared Prompt ID: {prompt_id}")
        print(f"   Status: {row['status']}")
        print(f"   Priority: {row['priority_level']}")
        print(f"   Estimated Tokens: {row['estimated_tokens']}")
        print(f"   Event Category: M&A_CORPORATE")

    def test_full_pipeline_earnings_news(self, seeded_db):
        """Integration test with earnings news in Italian."""
        raw = (
            "Fincantieri ha pubblicato i risultati del terzo trimestre 2024. "
            "I ricavi sono cresciuti del 18% a €2.1 miliardi, mentre l'EBITDA "
            "ha raggiunto €180 milioni con un margine del 8.6%. "
            "Il CdA ha confermato la guidance per l'intero esercizio. "
            "Il portafoglio ordini rimane robusto a €36 miliardi. "
            "Gli analisti di Mediobanca hanno alzato il prezzo obiettivo."
        )

        cleaned = clean_text(raw)
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        matches = engine.match(cleaned)
        tickers = {m.ticker for m in matches}
        assert 'FCT.MI' in tickers

        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(cleaned, list(tickers))
        assert category == "EARNINGS_GUIDANCE"
        assert priority == 1

        match_terms = [m.matched_text for m in matches]
        cropped = crop_context(cleaned, match_terms, max_chars=1000)

        filtered = FilteredNewsItem(
            source_id="ANSA_ECONOMIA",
            title="Fincantieri, ricavi Q3 +18%, confermata guidance",
            link="https://example.com/fincantieri-q3",
            url_hash=compute_url_hash("Fincantieri ricavi Q3",
                                       "https://example.com/fincantieri-q3"),
            cropped_context=cropped,
            matched_tickers=matches,
            matched_ticker_symbols=list(tickers),
            event_category=category,
            priority_level=priority,
        )

        cursor = seeded_db.execute(
            """INSERT INTO news_archive 
               (url_hash, source_id, title, link, published_at, raw_summary,
                cropped_context, matched_tickers_json, event_category, priority_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filtered.url_hash, filtered.source_id, filtered.title,
                filtered.link, filtered.published_at, '',
                filtered.cropped_context,
                json.dumps(filtered.matched_ticker_symbols),
                filtered.event_category, filtered.priority_level,
            )
        )
        news_id = cursor.lastrowid

        builder = PromptBuilder(db=seeded_db)
        prompt_id = builder.process_news_item(filtered, news_id)

        row = seeded_db.fetch_one(
            "SELECT * FROM prepared_prompts WHERE id = ?", (prompt_id,)
        )
        assert row['status'] == 'READY_FOR_LLM'
        assert 'Fincantieri' in row['user_prompt']
        print(f"\n[PASS] EARNINGS PIPELINE TEST PASSED (Prompt #{prompt_id})")

    def test_full_pipeline_with_portfolio_override(self, seeded_db):
        """Integration test with portfolio priority override."""
        # Set IP.MI as open position
        seeded_db.execute(
            "UPDATE watchlist SET is_portfolio_open = 1, portfolio_avg_price = 42.50, "
            "portfolio_allocation_pct = 8.0 WHERE ticker = 'IP.MI'"
        )

        text = "Interpump Group was mentioned in a general industry report about European industrials."
        cleaned = clean_text(text)

        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        matches = engine.match(cleaned)
        tickers = {m.ticker for m in matches}

        classifier = TaxonomyClassifier(db=seeded_db)
        category, priority = classifier.classify(cleaned, list(tickers))
        # Category would be GENERAL, but priority should be 1 due to portfolio
        assert priority == 1, f"Expected priority 1 (portfolio override), got {priority}"

        print(f"\n[PASS] PORTFOLIO OVERRIDE TEST PASSED (Priority: {priority})")

    def test_pipeline_prompt_count_and_query(self, seeded_db):
        """Verify we can query prepared_prompts by status."""
        builder = PromptBuilder(db=seeded_db)

        # Create and persist two prompts
        for i, (title, ticker, category) in enumerate([
            ("Interpump M&A deal", "IP.MI", "M&A_CORPORATE"),
            ("Lufthansa Q3 results", "LHA.DE", "EARNINGS_GUIDANCE"),
        ]):
            news = FilteredNewsItem(
                source_id="TEST",
                title=title,
                link=f"https://example.com/{i}",
                url_hash=f"hash_{i}_{title.replace(' ', '_')}",
                cropped_context=f"Context about {title} with financial details and revenue growth.",
                matched_tickers=[
                    TickerMatch(ticker=ticker, matched_text=title.split()[0], match_type="COMPANY_NAME")
                ],
                matched_ticker_symbols=[ticker],
                event_category=category,
                priority_level=1,
            )
            cursor = seeded_db.execute(
                """INSERT INTO news_archive 
                   (url_hash, source_id, title, link, cropped_context, 
                    matched_tickers_json, event_category, priority_level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (news.url_hash, news.source_id, news.title, news.link,
                 news.cropped_context, json.dumps(news.matched_ticker_symbols),
                 news.event_category, news.priority_level)
            )
            builder.process_news_item(news, cursor.lastrowid)

        # Query all READY_FOR_LLM prompts
        ready = seeded_db.fetch_all(
            "SELECT * FROM prepared_prompts WHERE status = 'READY_FOR_LLM' ORDER BY priority_level"
        )
        assert len(ready) >= 2
        print(f"\n[PASS] QUERY TEST PASSED ({len(ready)} prompts READY_FOR_LLM)")


# ═══════════════════════════════════════════════════════════════
# 10. PHASE 2 ARCHITECTURAL EVOLUTIONS TESTS (v1.3)
# ═══════════════════════════════════════════════════════════════

class TestPhase2ArchitecturalEvolutions:
    """Tests for Form 8.3/SEO noise filtering, broker disambiguation, German ratings, and title dedup."""

    def test_passive_noise_blocks_form_83_and_seo_calculators(self, seeded_db):
        """Form 8.3 filings and retrospective 'So viel Gewinn...' SEO articles must be discarded."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        # UK Form 8.3 passive disclosure -> must be blocked
        form83_matches = engine.match("Man Group PLC : Form 8.3 - Rotork plc. #FORM 8.3")
        assert form83_matches == []

        # German SEO retrospective calculator -> must be blocked
        seo_matches = engine.match(
            "MDAX-Papier Salzgitter-Aktie: So viel Gewinn hätte ein Investment in Salzgitter von vor 10 Jahren eingebracht"
        )
        assert seo_matches == []

    def test_uk_rule_27_takeover_passes_and_classifies_as_ma(self, seeded_db):
        """Real UK Rule 2.7 takeover announcements must pass and classify as M&A_CORPORATE (Priority 1)."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()
        classifier = TaxonomyClassifier(db=seeded_db)

        text = "Rule 2.7 Announcement: Recommended cash offer and firm intention to make an offer for Rotork plc"
        matches = engine.match(text)
        tickers = {m.ticker for m in matches}
        assert "ROR.L" in tickers

        cat, prio = classifier.classify(text, list(tickers))
        assert cat == "M&A_CORPORATE"
        assert prio == 1

    def test_financial_intermediary_broker_disambiguation(self, seeded_db):
        """Equita (EQUI.MI) issuing a rating on a third-party stock must be suppressed,
        while corporate governance news about Equita itself must match."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        # Case 1: Equita acting as broker on DiaSorin (not in watchlist) -> 0 matches
        broker_only = engine.match(
            "Diasorin, Equita alza il target price a 90 euro e conferma il giudizio Buy"
        )
        assert broker_only == []

        # Case 2: Equita acting as broker on Salvatore Ferragamo (in watchlist) -> only SFER.MI matches
        broker_and_target = engine.match(
            "Borsa di Milano: DiaSorin sale dopo la promozione di Equita, Ferragamo in calo a Piazza Affari"
        )
        matched_tickers = {m.ticker for m in broker_and_target}
        assert "EQUI.MI" not in matched_tickers
        assert "SFER.MI" in matched_tickers

        # Case 3: Equita's own financial results -> EQUI.MI must match
        own_results = engine.match(
            "Equita Group: il CdA di Equita approva il bilancio semestrale con utile netto e dividendo in crescita"
        )
        own_tickers = {m.ticker for m in own_results}
        assert "EQUI.MI" in own_tickers

    def test_german_analyst_rating_verbs_classification(self, seeded_db):
        """German analyst rating phrases ('hob Ziel', 'startet mit Neutral') must classify as ANALYST_RATING."""
        classifier = TaxonomyClassifier(db=seeded_db)

        cat1, prio1 = classifier.classify(
            "Deutsche Bank hob Ziel für die Jenoptik-Aktie auf 52 Euro an", ["JEN.DE"]
        )
        assert cat1 == "ANALYST_RATING"
        assert prio1 == 3

        cat2, prio2 = classifier.classify(
            "Citigroup startet mit Neutral: CTS Eventim-Aktie legt 4,05 Prozent zu", ["EVD.DE"]
        )
        assert cat2 == "ANALYST_RATING"
        assert prio2 == 3

    def test_combine_title_and_summary_prevents_duplication(self):
        """Google News summaries that repeat the title must not be duplicated in combined text."""
        from ingestion.text_cleaner import combine_title_and_summary

        title = clean_text("Diasorin, Equita alza il target price a 90 euro e conferma il giudizio Buy")
        summary = clean_text("Diasorin, Equita alza il target price a 90 euro e conferma il giudizio Buy Teleborsa")
        combined = combine_title_and_summary(title, summary)
        assert combined.count("Diasorin") == 1


class TestPhase3Live60MinHardening:
    @pytest.fixture
    def classifier(self):
        return TaxonomyClassifier()

    def test_management_board_contract_not_classified_as_contract_order(self, classifier):
        # Esempio reale dal test live: rinnovo contratto supervisory/executive board
        text_de = "Siltronic AG: Aufsichtsrat verlängert Vorstandsvertrag von Dr. Michael Heckmeier vorzeitig bis 2029"
        cat, prio = classifier.classify(text_de, ["WAF.DE"])
        assert cat != "CONTRACT_ORDER"
        assert prio == 3

        text_en = "Executive Board contract extension: Supervisory Board renews CEO contract for three years"
        cat_en, prio_en = classifier.classify(text_en, ["XYZ.DE"])
        assert cat_en != "CONTRACT_ORDER"
        assert prio_en == 3

    def test_calendar_invitation_not_classified_as_earnings(self, classifier):
        # Esempio reale dal test live: solo invito alla telefonkonferenz / webcast
        text = "STRATEC SE invites investors to Q3 interim report webcast and conference call"
        cat, prio = classifier.classify(text, ["SBS.DE"])
        assert cat != "EARNINGS_GUIDANCE"
        assert prio == 3

    @pytest.mark.asyncio
    async def test_syndication_dedup_cross_lingual_and_upgrade(self, tmp_path):
        from database.manager import DatabaseManager
        from ingestion.models import FilteredNewsItem, RegexMatchResult
        from main import NewsPipelineOrchestrator

        db = DatabaseManager(str(tmp_path / "test_dedup.db"))
        db.initialize_schema()
        orchestrator = NewsPipelineOrchestrator(db=db)

        # 1. Simula l'inserimento di una notizia inglese breve (stub) già accodata per LLM
        db.execute(
            """INSERT INTO news_archive
               (source_id, title, link, published_at, raw_summary, url_hash,
                cropped_context, matched_tickers_json, event_category, priority_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "INVESTING",
                "Nexans reports Q3 2024 revenue of 1.7 billion euros up 4.7 percent",
                "https://investing.com/news/nexans-stub",
                "2024-10-24 08:00:00",
                "Nexans revenue up 4.7%",
                "hash1",
                "Nexans reports Q3 2024 revenue of 1.7 billion euros up 4.7 percent.",
                json.dumps(["NEX.PA"]),
                "EARNINGS_GUIDANCE",
                1,
            )
        )
        db.execute(
            """INSERT INTO prepared_prompts (news_id, priority_level, system_prompt, user_prompt, expected_json_schema, estimated_tokens, status)
               VALUES (1, 1, 'sys', 'Short Prompt', '{}', 50, 'READY_FOR_LLM')"""
        )

        # 2. Arriva la versione francese dettagliata (con gli stessi numeri '1.7' e '4.7')
        rich_context = (
            "Nexans a publié un chiffre d'affaires de 1,7 milliard d'euros au troisième trimestre, "
            "en hausse de 4,7%. La rentabilité opérationnelle s'est nettement redressée sur l'ensemble "
            "des divisions industrielles grâce à una solide exécution stratégique et à un carnet de commandes record. "
            * 8
        )
        french_item = FilteredNewsItem(
            source_id="BOURSIER_FR",
            title="Nexans: chiffre d'affaires de 1,7 MdE au 3ème trimestre, en croissance de 4,7%",
            link="https://globenewswire.com/news/nexans-fr-full",
            published_at="2024-10-24 08:05:00",
            raw_summary=rich_context,
            url_hash="hash2",
            cropped_context=rich_context,
            matched_tickers=[RegexMatchResult("Nexans", "NEX.PA", "Nexans", 0, 6, "HIGH")],
            matched_ticker_symbols=["NEX.PA"],
            event_category="EARNINGS_GUIDANCE",
            priority_level=1,
        )

        is_dup = orchestrator._is_cross_source_duplicate(french_item, current_news_id=2)
        assert is_dup is True

        # Verifica che sia scattato il Context Upgrade sul prompt #1
        updated_prompt = db.fetch_one("SELECT user_prompt FROM prepared_prompts WHERE id = 1")
        assert "chiffre d'affaires" in updated_prompt["user_prompt"]
        assert len(updated_prompt["user_prompt"]) > 500


# ═══════════════════════════════════════════════════════════════
# 12. PHASE 4 GLOBAL HARDENING TESTS (v1.5)
# ═══════════════════════════════════════════════════════════════

class TestPhase4GlobalHardening:
    """Tests for v1.5: German compound words, US symbol collision prevention,
    non-corporate entity exclusion, and B2B idiom masking."""

    def test_german_compound_words_classified_correctly(self, seeded_db):
        """Umsatzprognose and Kartellamtsentscheid must match via morphological roots."""
        classifier = TaxonomyClassifier(db=seeded_db)

        # Krones Profit Warning with compound word Umsatzprognose -> EARNINGS_GUIDANCE (Priority 1)
        cat1, prio1 = classifier.classify(
            "Krones senkt Umsatzprognose: Krones-Aktie verliert 6,00 Prozent", ["KRN.DE"]
        )
        assert cat1 == "EARNINGS_GUIDANCE"
        assert prio1 == 1

        # Hornbach Antitrust approval with Kartellamtsentscheid -> REGULATORY_LEGAL / M&A (Priority 1 or 2)
        cat2, prio2 = classifier.classify(
            "Kartellamtsentscheid: Hornbach darf Hellweg-Märkte übernehmen", ["HBH.DE"]
        )
        assert cat2 in ("M&A_CORPORATE", "REGULATORY_LEGAL")
        assert prio2 in (1, 2)

    def test_us_ticker_in_parentheses_does_not_match(self, seeded_db):
        """Bare ticker in parentheses (PHM) for PulteGroup NYSE must not match Pharma Mar (PHM.MC)."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        text = "Flash résultats (PHM) : PulteGroup, Inc. annonce un chiffre d'affaires de 3,97 Md$"
        matches = engine.match(text)
        assert matches == []

    def test_non_corporate_entity_elis_excluded(self, seeded_db):
        """Consorzio ELIS must not match Elis SA (ELIS.PA) without hard financial context."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        text = "Elis, alla guida del consorzio arriva l'ad di Accenture, Teodoro Lio"
        matches = engine.match(text)
        assert matches == []

        # But Elis SA with real financial results must match
        text_real = "Elis SA: risultati semestrali a Parigi, fatturato ed EBITDA in forte crescita sul mercato Euronext"
        real_matches = engine.match(text_real)
        assert any(m.ticker == "ELIS.PA" for m in real_matches)

    def test_sports_betting_prognose_blocked_on_juventus(self, seeded_db):
        """Match prediction on Juventus with KI-Prognose must be blocked by passive noise."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        text = "Juventus - Rennes Tipp, KI-Prognose & Quoten | 22.10.2026 - Wettfreunde"
        assert engine.match(text) == []

    def test_b2b_revenue_cycle_management_does_not_trigger_earnings(self, seeded_db):
        """Product announcement mentioning 'revenue cycle management' must not become EARNINGS_GUIDANCE."""
        classifier = TaxonomyClassifier(db=seeded_db)

        text = "ACL Digital Launches AgentixCare AI Platform for Healthcare Providers with revenue cycle management"
        cat, prio = classifier.classify(text, ["ATE.PA"])
        assert cat == "GENERAL"
        assert prio == 3

    def test_equita_broker_on_moncler_suppressed(self, seeded_db):
        """Equita cutting target price on Moncler must be suppressed as broker action."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        text = "Moncler, Equita abbassa target price con fatturato del terzo trimestre debole"
        assert engine.match(text) == []


# ═══════════════════════════════════════════════════════════════
# 13. PHASE 5 TWO-HOUR LIVE TEST HARDENING & SHORT-CIRCUIT (v1.5)
# ═══════════════════════════════════════════════════════════════

class TestPhase5TwoHourLiveTestHardening:
    """Tests for v1.5 improvements from the 2-hour live test audit:
    - Common word case-sensitivity & strict context ('frontline teams' vs Frontline FRO.OL)
    - Redefined third-party acronyms ('My OnChain Net Yield Fund ("MONY")' vs Mony Group MONY.L)
    - Protection of 'International Workplace Group' (IWG.L) from matching bare 'Workplace'
    - Expanded broker disambiguation (TP ICAP Midcap TCAP.L on Somec)
    - Routine Consob PNC short-selling updates downgraded to GENERAL (Priority 3)
    - Football match previews without betting words ('Juventus gegen Rennes Prognose: Spitzenspiel')
    - 'sales' keyword in EARNINGS_GUIDANCE ('Valeo posts lower sales')
    - Step 0 Entity Coherence Short-Circuit in System/User Prompts
    """

    def test_lowercase_frontline_and_workplace_do_not_false_positive(self, seeded_db):
        """Lowercase 'frontline teams' (#01) and generic 'Workplace Disruption' (#32) must not match FRO.OL or IWG.L."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        # #01 from 2h test: Healthcare merger mentioning 'frontline ideas' and 'frontline teams'
        text_frontline = (
            "Redesign Health and ARC Global Merge to Help Health Systems Shape the Future of Healthcare. "
            "The merger brings capital to turn frontline ideas into products alongside frontline teams."
        )
        assert engine.match(text_frontline) == []

        # #32 from 2h test: HR guide mentioning 'Workplace Disruption'
        text_workplace = (
            "McLean & Company Publishes New Guide to Help HR Leaders Stabilize Culture Amid Workplace Disruption"
        )
        assert engine.match(text_workplace) == []

    def test_redefined_acronym_mony_fund_blocked(self, seeded_db):
        """J.P. Morgan's 'My OnChain Net Yield Fund ("MONY")' (#42) must not match Mony Group (MONY.L)."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        text_mony = (
            'J.P. Morgan Asset Management Expands Access to Its Tokenized Money Market Funds on Solana: '
            'My OnChain Net Yield Fund ("MONY") is a 506(c) private placement fund for qualified U.S. investors.'
        )
        assert engine.match(text_mony) == []

    def test_tp_icap_broker_disambiguated_on_somec(self, seeded_db):
        """TP ICAP Midcap (#08/#36) issuing target price on Somec must match only SOM.MI, not TCAP.L."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        text = 'Somec, TP ICAP Midcap alza target price a 30 euro e conferma Buy a Piazza Affari'
        matches = engine.match(text)
        tickers = {m.ticker for m in matches}
        assert "SOM.MI" in tickers
        assert "TCAP.L" not in tickers

    def test_consob_short_selling_pnc_downgraded_to_general_p3(self, seeded_db):
        """Daily Consob PNC short position updates (#04/#05) must be GENERAL (P3), not REGULATORY_LEGAL (P2)."""
        classifier = TaxonomyClassifier(db=seeded_db)

        text = (
            "Lottomatica, Capital Fund Management riduce lo short. Secondo l'aggiornamento Consob "
            "sulle posizioni nette corte (PNC), la posizione short è scesa allo 0,48%."
        )
        cat, prio = classifier.classify(text, ["LTMC.MI"])
        assert cat == "GENERAL"
        assert prio == 3

    def test_juventus_spitzenspiel_match_preview_blocked(self, seeded_db):
        """Football match preview 'Juventus gegen Rennes Prognose: Ein heiß umkämpftes Spitzenspiel' (#28) must be blocked."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        text = "Juventus gegen Rennes Prognose: Ein heiß umkämpftes Spitzenspiel im Allianz Stadium"
        assert engine.match(text) == []

    def test_sales_keyword_classifies_as_earnings_guidance(self, seeded_db):
        """'Valeo posts lower sales' (#46) must classify as EARNINGS_GUIDANCE (Priority 1)."""
        classifier = TaxonomyClassifier(db=seeded_db)
        cat, prio = classifier.classify(
            "Valeo posts lower sales as auto industry weakness persists", ["FR.PA"]
        )
        assert cat == "EARNINGS_GUIDANCE"
        assert prio == 1

    def test_step0_coherence_short_circuit_in_prompts(self, seeded_db):
        """System and User prompts must contain the STEP 0 Entity & Sector Coherence Short-Circuit rule."""
        sys_prompt = get_system_prompt()
        assert "STEP 0" in sys_prompt
        assert "[ERRORE_COERENZA_ENTITA]" in sys_prompt

    def test_static_quote_pages_and_cvs_us_collision_blocked(self, seeded_db):
        """Static LSE/Reuters quote pages (#05 AVON, #06 CVS) must be discarded at Stage 0/2."""
        engine = RegexEngine(db=seeded_db)
        engine.load_rules()

        assert engine.match("AVON TECHNOLOGIES PLC (AVON) Equity Price and Quote - London Stock Exchange") == []
        assert engine.match("(CVS) | Stock Price & Latest News") == []


