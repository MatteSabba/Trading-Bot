"""Seeder module: parses input files and populates watchlist, keywords, and news_sources tables.

Implements rigorous disambiguation rules for short/ambiguous ticker symbols
and generates brand alias patterns for compound European company names.
"""

import re
from pathlib import Path
from typing import Optional

from config.settings import (
    AMBIGUOUS_CLEAN_NAMES,
    AMBIGUOUS_SYMBOLS,
    COMMON_NAME_WORDS,
    COMPANY_BRAND_ALIASES,
    EXCHANGE_PREFIXES,
    LEGAL_SUFFIXES_PATTERN,
    NEWS_SOURCES_FILE,
    WATCHLIST_FILE,
)
from database.db_manager import DatabaseManager


def _clean_company_name(raw_name: str) -> str:
    """Remove legal and descriptive suffixes from company name.

    If the cleaned result is < 3 characters or in AMBIGUOUS_CLEAN_NAMES, keep the original name.
    """
    cleaned = re.sub(LEGAL_SUFFIXES_PATTERN, '', raw_name, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = cleaned.strip('.,;:- ')
    if len(cleaned) < 3 or cleaned.lower() in AMBIGUOUS_CLEAN_NAMES:
        return raw_name.strip()
    return cleaned

def _extract_base_symbol(ticker: str) -> str:
    """Extract the base symbol from a full ticker (e.g., 'IP.MI' -> 'IP', 'EMBRAC-B.ST' -> 'EMBRAC')."""
    base = ticker.split('.')[0]
    return base.split('-')[0] if '-' in base else base


def _is_short_or_ambiguous(base_symbol: str) -> bool:
    """Check if a base symbol is short (<= 3 chars) or in the ambiguous blacklist."""
    return len(base_symbol) <= 3 or base_symbol.upper() in AMBIGUOUS_SYMBOLS


def _build_exchange_prefix_pattern() -> str:
    """Build the alternation pattern for exchange prefixes."""
    return '|'.join(re.escape(p) for p in EXCHANGE_PREFIXES)


def _generate_keywords(ticker: str, base_symbol: str, clean_name: str) -> list[tuple]:
    """Generate keyword patterns for a given company following the disambiguation rules.

    Returns list of tuples: (ticker, pattern, match_type, is_case_sensitive, requires_financial_context)
    """
    keywords = []
    escaped_clean_name = re.escape(clean_name)
    escaped_ticker = re.escape(ticker)
    exchange_prefixes = _build_exchange_prefix_pattern()
    escaped_base = re.escape(base_symbol)

    # Rule 2a: Primary COMPANY_NAME pattern
    # Dynamic case-sensitivity: short names (<= 4 chars) AND common dictionary words
    # in COMMON_NAME_WORDS must match case (capitalized) to avoid lowercase common words in prose
    is_short_name = len(clean_name) <= 4
    is_common_word = clean_name.lower() in COMMON_NAME_WORDS
    requires_fin_ctx = 1 if (is_common_word or is_short_name) else 0
    keywords.append((
        ticker,
        rf'\b{escaped_clean_name}\b',
        'COMPANY_NAME',
        1 if (is_short_name or is_common_word) else 0,
        requires_fin_ctx,
    ))

    # Rule 2a-bis: Brand aliases for multi-word European companies (e.g. "Deutsche Lufthansa" -> "Lufthansa")
    for alias in COMPANY_BRAND_ALIASES.get(ticker, []):
        if alias.lower() != clean_name.lower():
            escaped_alias = re.escape(alias)
            keywords.append((
                ticker,
                rf'\b{escaped_alias}\b',
                'COMPANY_NAME',
                0,
                1,  # aliases always require financial context for safety
            ))

    # Rule 2b: FULL_TICKER pattern (case sensitive)
    keywords.append((
        ticker,
        rf'\b{escaped_ticker}\b',
        'FULL_TICKER',
        1,  # case sensitive
        0,
    ))

    # Rule 2c: EXPLICIT_SYMBOL pattern (case sensitive)
    # Strictly requires an explicit European exchange prefix code (e.g. BIT:IP, XETRA:LHA, BME:PHM)
    # Never match bare (SYMBOL) in parentheses without exchange to prevent collision with US NYSE/NASDAQ stocks.
    explicit_pattern = (
        rf'\b(?:{exchange_prefixes})\s*:\s*{escaped_base}\b'
    )
    keywords.append((
        ticker,
        explicit_pattern,
        'EXPLICIT_SYMBOL',
        1,  # case sensitive
        0,
    ))

    # Rule 3: SAFE_LONG_TICKER only if base_symbol >= 4 chars AND not in ambiguous set
    if len(base_symbol) >= 4 and base_symbol.upper() not in AMBIGUOUS_SYMBOLS:
        keywords.append((
            ticker,
            rf'\b{escaped_base}\b',
            'SAFE_LONG_TICKER',
            1,  # case sensitive
            1,  # requires financial context
        ))

    return keywords


def _find_watchlist_file() -> Path:
    """Find the watchlist file, handling the .txt.txt double extension case."""
    if WATCHLIST_FILE.exists():
        return WATCHLIST_FILE
    alt_path = WATCHLIST_FILE.with_suffix('.txt.txt')
    if alt_path.exists():
        return alt_path
    double_ext = WATCHLIST_FILE.parent / (WATCHLIST_FILE.name + '.txt')
    if double_ext.exists():
        return double_ext
    raise FileNotFoundError(
        f"Cannot find watchlist file at {WATCHLIST_FILE} or {alt_path} or {double_ext}"
    )


def seed_watchlist(db: DatabaseManager) -> int:
    """Parse eu_small_mid_caps.txt and populate watchlist + keywords tables.

    Returns the number of companies seeded.
    """
    filepath = _find_watchlist_file()
    companies = []
    all_keywords = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) < 4:
                continue

            ticker = parts[0].strip()
            company_name = parts[1].strip()
            sector = parts[2].strip()
            index_country = parts[3].strip()

            base_symbol = _extract_base_symbol(ticker)
            clean_name = _clean_company_name(company_name)

            companies.append((
                ticker, base_symbol, company_name, clean_name,
                sector, index_country, 0, None, 0.0, 1
            ))

            kw_list = _generate_keywords(ticker, base_symbol, clean_name)
            all_keywords.extend(kw_list)

    # Insert or update watchlist so updated clean_names are reflected
    db.execute_many(
        """INSERT INTO watchlist 
           (ticker, base_symbol, company_name, clean_name, sector, index_country,
            is_portfolio_open, portfolio_avg_price, portfolio_allocation_pct, is_active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(ticker) DO UPDATE SET
               base_symbol = excluded.base_symbol,
               company_name = excluded.company_name,
               clean_name = excluded.clean_name,
               sector = excluded.sector,
               index_country = excluded.index_country""",
        companies
    )

    db.execute_many(
        """INSERT INTO keywords (ticker, pattern, match_type, is_case_sensitive, requires_financial_context)
           VALUES (?, ?, ?, ?, ?)""",
        all_keywords
    )

    return len(companies)


def seed_news_sources(db: DatabaseManager) -> int:
    """Parse eu_news_sources.txt and populate news_sources table.

    Returns the number of sources seeded.
    """
    sources = []

    with open(NEWS_SOURCES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) < 5:
                continue

            source_id = parts[0].strip()
            feed_type = parts[1].strip()
            language = parts[2].strip()
            poll_interval_sec = int(parts[3].strip())
            url = parts[4].strip()

            sources.append((
                source_id, feed_type, language, poll_interval_sec, url,
                None, None, None, 0, 1
            ))

    if sources:
        active_ids = [s[0] for s in sources]
        placeholders = ','.join('?' for _ in active_ids)
        db.execute(
            f"DELETE FROM news_sources WHERE source_id NOT IN ({placeholders})",
            tuple(active_ids),
        )

    db.execute_many(
        """INSERT INTO news_sources
           (source_id, feed_type, language, poll_interval_sec, url,
            last_polled_at, last_etag, last_modified, consecutive_failures, is_enabled)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_id) DO UPDATE SET
               feed_type = excluded.feed_type,
               language = excluded.language,
               poll_interval_sec = excluded.poll_interval_sec,
               url = excluded.url""",
        sources
    )

    return len(sources)


def seed_all(db: Optional[DatabaseManager] = None) -> dict:
    """Run full seeding pipeline: schema init + watchlist + news sources.

    Returns dict with counts.
    """
    if db is None:
        db = DatabaseManager()
    db.initialize_schema()
    db.execute("DELETE FROM keywords")
    n_companies = seed_watchlist(db)
    n_sources = seed_news_sources(db)
    return {
        'companies': n_companies,
        'keywords': db.fetch_one('SELECT COUNT(*) as cnt FROM keywords')['cnt'],
        'sources': n_sources,
    }