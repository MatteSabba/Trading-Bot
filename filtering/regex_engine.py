"""Two-stage multi-pattern matching engine with O(1) ticker rule lookup and B2C noise filtering.

Stage 1 (O(N) single pass): flashtext KeywordProcessor for candidate pre-screening.
Stage 2 (O(1) per-candidate regex lookup): Direct dictionary lookup by candidate ticker
(_rules_by_ticker) instead of scanning all 2,200+ rules linearly, plus Domain Noise
Blacklist protection (blocks Formula 1, MotoGP, football match reports, and fashion shows
unless strict equity/corporate terms are present).

Designed to scale to 1000+ companies in < 0.2ms per article.
"""

import logging
import re
from typing import Dict, List, Optional, Set

from config.settings import (
    COMPANY_BRAND_ALIASES,
    DOMAIN_NOISE_BLACKLIST,
    FINANCIAL_CONTEXT_WORDS,
    FINANCIAL_INTERMEDIARY_TICKERS,
    ANALYST_ISSUER_ACTION_PATTERNS,
    SELF_CORPORATE_GOVERNANCE_MARKERS,
    HARD_FINANCIAL_CONTEXT_WORDS,
    PASSIVE_NOISE_PATTERNS,
    PASSIVE_NOISE_OVERRIDE_PATTERNS,
    STRICT_DICTIONARY_COMPANY_NAMES,
)
from database.db_manager import DatabaseManager
from models.schemas import TickerMatch

logger = logging.getLogger(__name__)

# Try flashtext for O(N) keyword pre-screening, fallback to simple set lookup
try:
    from flashtext import KeywordProcessor
    _HAS_FLASHTEXT = True
except ImportError:
    _HAS_FLASHTEXT = False
    logger.info("flashtext not available, using fallback set-based pre-screening")


class RegexRule:
    """A single compiled regex rule for ticker matching."""
    __slots__ = (
        'ticker', 'pattern', 'compiled', 'match_type',
        'is_case_sensitive', 'requires_financial_context',
    )

    def __init__(
        self,
        ticker: str,
        pattern: str,
        match_type: str,
        is_case_sensitive: bool,
        requires_financial_context: bool,
    ):
        self.ticker = ticker
        self.pattern = pattern
        self.match_type = match_type
        self.is_case_sensitive = is_case_sensitive
        self.requires_financial_context = requires_financial_context
        flags = 0 if is_case_sensitive else re.IGNORECASE
        self.compiled = re.compile(pattern, flags)


class RegexEngine:
    """Two-stage matching engine with O(1) per-ticker indexing and noise guardrails."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self._case_sensitive_rules: List[RegexRule] = []
        self._case_insensitive_rules: List[RegexRule] = []
        self._rules_by_ticker: Dict[str, List[RegexRule]] = {}
        self._clean_name_by_ticker: Dict[str, str] = {}
        self._keyword_processor: Optional['KeywordProcessor'] = None
        self._keyword_set: Set[str] = set()
        self._keyword_to_tickers: Dict[str, Set[str]] = {}
        self._financial_context_pattern = self._build_word_set_pattern(
            FINANCIAL_CONTEXT_WORDS, word_boundaries=False
        )
        self._hard_financial_context_pattern = self._build_word_set_pattern(
            HARD_FINANCIAL_CONTEXT_WORDS, word_boundaries=True
        )
        self._domain_noise_pattern = self._build_word_set_pattern(
            DOMAIN_NOISE_BLACKLIST, word_boundaries=True
        )
        self._passive_noise_pattern = re.compile(
            r'(?:' + '|'.join(PASSIVE_NOISE_PATTERNS) + r')',
            re.IGNORECASE,
        )
        self._passive_noise_override_pattern = re.compile(
            r'(?:' + '|'.join(PASSIVE_NOISE_OVERRIDE_PATTERNS) + r')',
            re.IGNORECASE,
        )
        self._analyst_issuer_action_pattern = re.compile(
            r'(?:' + '|'.join(ANALYST_ISSUER_ACTION_PATTERNS) + r')',
            re.IGNORECASE,
        )
        self._self_corporate_governance_pattern = re.compile(
            r'(?:' + '|'.join(SELF_CORPORATE_GOVERNANCE_MARKERS) + r')',
            re.IGNORECASE,
        )
        # Non-corporate entity qualifiers (Consorzio, Fondazione, Istituto, Associazione, etc.)
        self._non_corporate_qualifiers_pattern = re.compile(
            r'\b(?:consorzio|fondazione|associazione|istituto|onlus|ong|comune\s+di|universit[àa])\s+',
            re.IGNORECASE,
        )
        # US exchange markers that signal a foreign American ticker
        self._us_exchange_marker_pattern = re.compile(
            r'\b(?:NYSE|NASDAQ|AMEX)\s*:\s*|\bInc\.\b|\bCorporation\b',
            re.IGNORECASE,
        )
        # US Private Placement / Crypto Token / US Fund markers (e.g., 506(c), Solana, qualified U.S. investors)
        self._us_fund_or_crypto_pattern = re.compile(
            r'\b(?:506\(c\)|qualified\s+u\.?s\.?\s+investors|onchain|solana|stablecoin|nyse\s*:|nasdaq\s*:)\b',
            re.IGNORECASE,
        )
        self._loaded = False

    @staticmethod
    def _build_word_set_pattern(words_set: Set[str], word_boundaries: bool = False) -> re.Pattern:
        """Build a pre-compiled regex alternation pattern from a set of words."""
        words = sorted(words_set, key=len, reverse=True)
        escaped = [re.escape(w) for w in words]
        inner = '|'.join(escaped)
        pattern = rf'\b(?:{inner})\b' if word_boundaries else rf'(?:{inner})'
        return re.compile(pattern, re.IGNORECASE)

    def load_rules(self) -> int:
        """Load all keyword rules from the database and build the O(1) indexed matching engine.

        Returns the total number of rules loaded.
        """
        rows = self.db.fetch_all(
            """SELECT k.ticker, k.pattern, k.match_type, k.is_case_sensitive, 
                      k.requires_financial_context, w.clean_name, w.base_symbol
               FROM keywords k
               JOIN watchlist w ON k.ticker = w.ticker
               WHERE w.is_active = 1"""
        )

        self._case_sensitive_rules.clear()
        self._case_insensitive_rules.clear()
        self._rules_by_ticker.clear()
        self._clean_name_by_ticker.clear()
        self._keyword_to_tickers.clear()
        self._keyword_set.clear()

        if _HAS_FLASHTEXT:
            self._keyword_processor = KeywordProcessor(case_sensitive=False)
        else:
            self._keyword_processor = None

        for row in rows:
            rule = RegexRule(
                ticker=row['ticker'],
                pattern=row['pattern'],
                match_type=row['match_type'],
                is_case_sensitive=bool(row['is_case_sensitive']),
                requires_financial_context=bool(row['requires_financial_context']),
            )

            if rule.is_case_sensitive:
                self._case_sensitive_rules.append(rule)
            else:
                self._case_insensitive_rules.append(rule)

            clean_name = row['clean_name']
            base_symbol = row['base_symbol']
            ticker = row['ticker']
            self._clean_name_by_ticker[ticker] = clean_name

            stage1_keys = [clean_name.lower(), base_symbol.lower(), ticker.lower()]
            for alias in COMPANY_BRAND_ALIASES.get(ticker, []):
                stage1_keys.append(alias.lower())

            for keyword in stage1_keys:
                if len(keyword) >= 2:
                    self._keyword_set.add(keyword)
                    if keyword not in self._keyword_to_tickers:
                        self._keyword_to_tickers[keyword] = set()
                    self._keyword_to_tickers[keyword].add(ticker)

                    if self._keyword_processor is not None:
                        try:
                            self._keyword_processor.add_keyword(keyword, keyword)
                        except Exception:
                            pass

        # Populate per-ticker index: case-insensitive rules (COMPANY_NAME) first,
        # followed by case-sensitive rules (FULL_TICKER, EXPLICIT_SYMBOL, SAFE_LONG_TICKER)
        for rule in self._case_insensitive_rules:
            self._rules_by_ticker.setdefault(rule.ticker, []).append(rule)
        for rule in self._case_sensitive_rules:
            self._rules_by_ticker.setdefault(rule.ticker, []).append(rule)

        self._loaded = True
        total = len(self._case_sensitive_rules) + len(self._case_insensitive_rules)
        logger.info(
            f"Loaded {total} regex rules across {len(self._rules_by_ticker)} tickers "
            f"({len(self._case_sensitive_rules)} case-sensitive, "
            f"{len(self._case_insensitive_rules)} case-insensitive)"
        )
        return total

    def _stage1_prescreening(self, text: str) -> Set[str]:
        """Stage 1: Fast O(N) pre-screening to find candidate tickers."""
        candidate_tickers = set()
        text_lower = text.lower()

        if self._keyword_processor is not None:
            found_keywords = self._keyword_processor.extract_keywords(text_lower)
            for kw in found_keywords:
                if kw in self._keyword_to_tickers:
                    candidate_tickers.update(self._keyword_to_tickers[kw])
        else:
            for keyword, tickers in self._keyword_to_tickers.items():
                if keyword in text_lower:
                    candidate_tickers.update(tickers)

        return candidate_tickers

    def _has_financial_context(self, text: str) -> bool:
        """Check if text contains at least one standard financial context word."""
        return bool(self._financial_context_pattern.search(text))

    def _has_hard_financial_context(self, text: str) -> bool:
        """Check if text contains strict stock-market / corporate accounting terms."""
        return bool(self._hard_financial_context_pattern.search(text))

    def _has_domain_noise(self, text: str) -> bool:
        """Check if text contains sports, motorsport, or fashion event noise terms."""
        return bool(self._domain_noise_pattern.search(text))

    def _is_passive_noise(self, text: str) -> bool:
        """Check if text matches passive regulatory disclosures (Form 8.3/8.5) or retrospective SEO spam.

        Returns False if a real takeover override pattern (e.g. Rule 2.7 / Rule 2.4) is present.
        """
        if not self._passive_noise_pattern.search(text):
            return False
        if self._passive_noise_override_pattern.search(text):
            return False
        return True

    def _filter_intermediary_false_positives(
        self, text: str, matches: Dict[str, TickerMatch]
    ) -> Dict[str, TickerMatch]:
        """Remove financial intermediary tickers (e.g., EQUI.MI, MB.MI, DWS.DE, EMG.L)
        when they appear solely as the broker/analyst issuing a rating or filing on another company.
        """
        if not matches:
            return matches

        intermediary_present = [t for t in matches if t in FINANCIAL_INTERMEDIARY_TICKERS]
        if not intermediary_present:
            return matches

        # If the text mentions analyst rating issuance patterns or contains broker report indicators
        # without self-corporate governance markers
        is_analyst_action = bool(self._analyst_issuer_action_pattern.search(text))
        has_self_gov = bool(self._self_corporate_governance_pattern.search(text))

        # Check syntactic structure: title starts with another company followed by comma and broker name
        # e.g., "Moncler, Equita abbassa..."
        has_comma_broker_structure = any(
            re.search(rf'^[A-Z][a-zA-Z0-9\s]+,\s*{re.escape(m.matched_text)}\b', text.strip())
            for m in matches.values() if m.ticker in FINANCIAL_INTERMEDIARY_TICKERS
        )

        if (is_analyst_action or has_comma_broker_structure) and not has_self_gov:
            filtered_matches = {
                t: m for t, m in matches.items() if t not in FINANCIAL_INTERMEDIARY_TICKERS
            }
            return filtered_matches

        return matches

    def _is_redefined_third_party_acronym(
        self, text: str, ticker: str, matched_text: str
    ) -> bool:
        """Detect if a short matched symbol/name (<= 5 chars, e.g. 'MONY', 'FII') is explicitly
        defined in the text as the acronym of an unrelated third-party fund, product, or entity:
        e.g., 'My OnChain Net Yield Fund ("MONY")' or 'Fondo Italiano d'Investimento (FII)'.
        """
        clean_token = matched_text.strip()
        if len(clean_token) > 5:
            return False

        expected_clean_name = self._clean_name_by_ticker.get(ticker, "").lower()
        acronym_def_re = re.compile(
            rf'([A-Z][A-Za-z0-9&\'\-]+(?:\s+[A-Za-z0-9&\'\-]+){{1,5}})\s*'
            rf'\(\s*["\'“”«»]?{re.escape(clean_token)}["\'“”«»]?\s*\)',
        )
        for m in acronym_def_re.finditer(text):
            preceding_entity = m.group(1).strip().lower()
            if expected_clean_name and expected_clean_name not in preceding_entity:
                return True

        # Also block if a short symbol/name (<= 4 chars) without full European ticker suffix
        # appears in a US SEC 506(c) / Solana / US-only fund press release
        if len(clean_token) <= 4 and '.' not in clean_token:
            if self._us_fund_or_crypto_pattern.search(text):
                return True

        return False

    def match(self, text: str) -> List[TickerMatch]:
        """Run the full two-stage matching pipeline on text with O(1) candidate lookup.

        Args:
            text: Combined title + summary text to scan.

        Returns:
            List of TickerMatch objects with unique ticker matches.
        """
        if not self._loaded:
            self.load_rules()

        if not text:
            return []

        # Stage 0: Fast O(1) discard for passive regulatory noise (Form 8.3) & SEO retrospective calculators
        if self._is_passive_noise(text):
            return []

        # Stage 1: Pre-screen to find candidate tickers
        candidate_tickers = self._stage1_prescreening(text)
        if not candidate_tickers:
            return []

        # Check for sports/fashion/motorsport noise (e.g. Formula 1 Brembo, Serie A Juventus).
        if self._has_domain_noise(text) and not self._has_hard_financial_context(text):
            return []

        # Stage 2: Direct O(1) lookup of compiled rules only for candidate tickers
        matches: Dict[str, TickerMatch] = {}
        has_fin_context: Optional[bool] = None
        has_hard_fin_context: Optional[bool] = None

        for ticker in candidate_tickers:
            for rule in self._rules_by_ticker.get(ticker, ()):
                m = rule.compiled.search(text)
                if m:
                    matched_str = m.group()
                    matched_lower = matched_str.lower()

                    # Guard A: Short names (<= 4 chars) OR high-collision dictionary words
                    # (e.g., Frontline, Workplace, Rational, Believe, Mony, Catena)
                    # require HARD equity/corporate financial context
                    is_short_or_strict_dict = (
                        rule.match_type == 'COMPANY_NAME'
                        and (
                            len(rule.pattern.strip(r'\b')) <= 4
                            or matched_lower in STRICT_DICTIONARY_COMPANY_NAMES
                        )
                    )
                    if is_short_or_strict_dict:
                        if has_hard_fin_context is None:
                            has_hard_fin_context = self._has_hard_financial_context(text)
                        if not has_hard_fin_context:
                            continue

                        # Guard B: Reject non-corporate entities (e.g. "Consorzio Elis", "Fondazione...")
                        match_start = m.start()
                        prefix_window = text[max(0, match_start - 35):match_start]
                        if self._non_corporate_qualifiers_pattern.search(prefix_window):
                            continue

                    # Guard C: Reject acronyms explicitly redefined for third-party funds/entities
                    # e.g., 'My OnChain Net Yield Fund ("MONY")' or 'Fondo Italiano d'Investimento (FII)'
                    if rule.match_type in ('COMPANY_NAME', 'SAFE_LONG_TICKER'):
                        if self._is_redefined_third_party_acronym(text, ticker, matched_str):
                            continue

                    if rule.requires_financial_context:
                        if has_fin_context is None:
                            has_fin_context = self._has_financial_context(text)
                        if not has_fin_context:
                            continue

                    matches[ticker] = TickerMatch(
                        ticker=ticker,
                        matched_text=matched_str,
                        match_type=rule.match_type,
                    )
                    break  # Keep the first (most specific) match for this ticker

        # Stage 3: Disambiguate financial intermediaries acting as brokers/analysts on third parties
        matches = self._filter_intermediary_false_positives(text, matches)

        return list(matches.values())