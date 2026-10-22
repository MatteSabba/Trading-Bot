"""Dynamic Prompt Builder with Latency Injection, Small Cap Liquidity, and Microstructure Guardrails.

Constructs the complete prompt payload by combining:
- System prompt with financial analysis, slippage, and freshness rules
- User prompt with news context, latency calculation (minutes_ago), dynamic CET market phase,
  retrospective price-action / routine buyback / ABB microstructure flags, matched assets
  with liquidity tier flags, portfolio state, and market context
- Target JSON schema for structured LLM output
- Token estimation for VRAM monitoring

Persists the result in the prepared_prompts table with status READY_FOR_LLM.
"""

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

from config.settings import STALE_NEWS_MINUTES_THRESHOLD
from database.db_manager import DatabaseManager
from ingestion.rss_poller import parse_rss_datetime
from models.schemas import FilteredNewsItem, PreparedPromptPayload
from prompt_builder.templates import (
    format_market_context,
    format_portfolio_context,
    get_system_prompt,
    get_target_json_schema,
)

logger = logging.getLogger(__name__)

_ILLIQUID_INDEX_MARKERS = ("SMALL", "STAR", "SDAX")

# Retrospective price-action & volatility halt detector (prevents buying after a spike)
_RETROSPECTIVE_SPIKE_REGEX = re.compile(
    r'(?:'
    r'\b(?:vola|balza|crolla|affonda|strappa|accelera|brilla|scivola)\s+(?:in\s+borsa|a\s+piazza\s+affari|del\s+[+-]?\d+)'
    r'|\b(?:sospes[oa]\s+per\s+eccesso\s+di\s+(?:rialzo|ribasso)|asta\s+di\s+volatilit[àa]|trading\s+halt)'
    r'|\b(?:shares|stock)\s+(?:surge[ds]?|plunge[ds]?|soar(?:ed|s)?|jump(?:ed|s)?|tumble[ds]?|drop(?:ped|s)?)\s+\d+'
    r')',
    re.IGNORECASE,
)

# Routine weekly treasury share buyback disclosure detector (administrative noise on IT_STAR / EU)
_ROUTINE_BUYBACK_REGEX = re.compile(
    r'(?:'
    r'comunicazione\s+in\s+merito\s+all[\'’]acquisto\s+di\s+azioni\s+proprie'
    r'|informativa\s+sull[\'’]acquisto\s+di\s+azioni\s+proprie'
    r'|riepilogo\s+settimanale\s+azioni\s+proprie'
    r'|transaction[s]?\s+in\s+own\s+shares'
    r')',
    re.IGNORECASE,
)

# Accelerated Bookbuilding / Dilutive placement detector
_ABB_DILUTION_REGEX = re.compile(
    r'\b(?:accelerated\s+bookbuilding|collocamento\s+accelerato|aumento\s+di\s+capitale|rights\s+issue)\b',
    re.IGNORECASE,
)


class PromptBuilder:
    """Builds and persists complete prompt payloads from filtered news items."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self._system_prompt = get_system_prompt()
        self._json_schema = get_target_json_schema()

    def _get_portfolio_positions(self, tickers: List[str]) -> List[dict]:
        """Fetch open portfolio positions for the given tickers."""
        if not tickers:
            return []

        placeholders = ','.join('?' for _ in tickers)
        rows = self.db.fetch_all(
            f"""SELECT ticker, company_name, portfolio_avg_price, portfolio_allocation_pct
                FROM watchlist 
                WHERE ticker IN ({placeholders}) AND is_portfolio_open = 1""",
            tuple(tickers)
        )

        return [
            {
                'ticker': row['ticker'],
                'company_name': row['company_name'],
                'avg_price': row['portfolio_avg_price'],
                'allocation_pct': row['portfolio_allocation_pct'],
            }
            for row in rows
        ]

    def _get_watchlist_info(self, tickers: List[str]) -> List[dict]:
        """Fetch watchlist details for matched tickers."""
        if not tickers:
            return []

        placeholders = ','.join('?' for _ in tickers)
        rows = self.db.fetch_all(
            f"""SELECT ticker, company_name, sector, index_country
                FROM watchlist WHERE ticker IN ({placeholders})""",
            tuple(tickers)
        )
        return [dict(row) for row in rows]

    @staticmethod
    def _compute_latency_descriptor(published_at: Optional[str], now_utc: datetime) -> str:
        """Compute the age of the news item in minutes and flag stale news."""
        pub_dt = parse_rss_datetime(published_at)
        if pub_dt is None:
            return "N/A (Unverified publication timestamp)"

        minutes_ago = max(0, int((now_utc - pub_dt).total_seconds() // 60))
        if minutes_ago > STALE_NEWS_MINUTES_THRESHOLD:
            return (
                f"{minutes_ago} minutes ago [STALE_NEWS_WARNING: > {STALE_NEWS_MINUTES_THRESHOLD}m "
                f"- likely ALREADY_PRICED_IN]"
            )
        return f"{minutes_ago} minutes ago [FRESH_CATALYST]"

    @staticmethod
    def _detect_microstructure_warnings(title: str, context: str) -> List[str]:
        """Detect retrospective price-action spikes, routine buybacks, or dilutive ABBs."""
        combined = f"{title}. {context}"
        warnings: List[str] = []

        if _RETROSPECTIVE_SPIKE_REGEX.search(combined):
            warnings.append(
                "- [MARKET_REACTION_IN_PROGRESS]: Article reports an ongoing stock price jump/drop "
                "or volatility auction halt. The catalyst is likely ALREADY_PRICED_IN; avoid chasing."
            )
        if _ROUTINE_BUYBACK_REGEX.search(combined):
            warnings.append(
                "- [ROUTINE_BUYBACK_DISCLOSURE]: Mandatory weekly disclosure of past treasury share "
                "purchases. Classify as event_type='NOISE' and recommended_action='IGNORE_NOISE'."
            )
        if _ABB_DILUTION_REGEX.search(combined):
            warnings.append(
                "- [DILUTION_OR_ABB_WARNING]: Mentions Accelerated Bookbuilding or capital increase. "
                "Evaluate short-term discount overhang (typically bearish for existing equity holders)."
            )
        return warnings

    def build_user_prompt(self, news: FilteredNewsItem) -> str:
        """Build the user prompt section with latency, market session, and microstructure guardrails."""
        now_dt = datetime.now(timezone.utc)
        utc_now = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        latency_str = self._compute_latency_descriptor(news.published_at, now_dt)

        asset_details = self._get_watchlist_info(news.matched_ticker_symbols)
        portfolio_positions = self._get_portfolio_positions(news.matched_ticker_symbols)

        # Format matched assets block (preserving exact prefix pattern for parser compatibility)
        assets_block = "## Matched Assets:\n"
        guardrail_notes: List[str] = []
        for asset in asset_details:
            idx_country = asset['index_country']
            assets_block += (
                f"- {asset['ticker']}: {asset['company_name']} "
                f"(Sector: {asset['sector']}, Index: {idx_country})\n"
            )
            if any(marker in idx_country.upper() for marker in _ILLIQUID_INDEX_MARKERS):
                guardrail_notes.append(
                    f"- {asset['ticker']} ({idx_country}): SMALL_CAP_LIQUIDITY_GUARDRAIL "
                    f"(Use BUY_LIMIT_INTRADAY or BUY_LIMIT_PREMARKET only; avoid BUY_MARKET)"
                )

        guardrail_notes.extend(
            self._detect_microstructure_warnings(news.title, news.cropped_context)
        )

        if guardrail_notes:
            assets_block += (
                "\n## Liquidity & Microstructure Guardrails:\n"
                + "\n".join(guardrail_notes)
                + "\n"
            )

        # Format matched text evidence
        match_evidence = "## Match Evidence:\n"
        for tm in news.matched_tickers:
            match_evidence += f"- [{tm.match_type}] \"{tm.matched_text}\" -> {tm.ticker}\n"

        user_prompt = f"""## Analysis Request
Timestamp (UTC): {utc_now}
News Age / Latency: {latency_str}
Event Category: {news.event_category}
Priority Level: {news.priority_level}
Source: {news.source_id}

## News Content
Title: {news.title}
Published: {news.published_at or 'N/A'}

Context:
{news.cropped_context}

{assets_block}
{match_evidence}
{format_market_context(now_dt)}

{format_portfolio_context(portfolio_positions)}

## Instructions
Analyze this news item and respond with a JSON object matching the provided schema.
Focus on:
0. STEP 0 (Coherence Gate): Verify the news actually concerns the matched European company and its Sector; if it is a homonym/unrelated entity, short-circuit immediately with event_type="NOISE", impacted_assets=[], and executive_summary_it="[ERRORE_COERENZA_ENTITA] ...".
1. Immediate price impact assessment for each matched asset (checking fine print for dilution/ABB/debt)
2. Sentiment quantification with confidence level
3. Actionable trading recommendation respecting Small Cap liquidity, market session, and latency guardrails
4. Executive summary in Italian (max 2 sentences)
"""
        return user_prompt

    def build_prompt(self, news: FilteredNewsItem, news_id: int) -> PreparedPromptPayload:
        """Build the complete prompt payload."""
        user_prompt = self.build_user_prompt(news)

        estimated_tokens = PreparedPromptPayload.estimate_tokens(
            self._system_prompt, user_prompt, self._json_schema
        )

        return PreparedPromptPayload(
            news_id=news_id,
            priority_level=news.priority_level,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            expected_json_schema=self._json_schema,
            estimated_tokens=estimated_tokens,
            status='READY_FOR_LLM',
        )

    def persist_prompt(self, payload: PreparedPromptPayload) -> int:
        """Save the prepared prompt to the database."""
        cursor = self.db.execute(
            """INSERT INTO prepared_prompts 
               (news_id, priority_level, system_prompt, user_prompt, 
                expected_json_schema, estimated_tokens, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                payload.news_id,
                payload.priority_level,
                payload.system_prompt,
                payload.user_prompt,
                payload.expected_json_schema,
                payload.estimated_tokens,
                payload.status,
            )
        )
        return cursor.lastrowid

    def process_news_item(self, news: FilteredNewsItem, news_id: int) -> int:
        """Full pipeline: build prompt + persist to DB."""
        payload = self.build_prompt(news, news_id)
        prompt_id = self.persist_prompt(payload)
        logger.info(
            f"Prepared prompt #{prompt_id} for news #{news_id} "
            f"(priority={payload.priority_level}, tokens≈{payload.estimated_tokens})"
        )
        return prompt_id