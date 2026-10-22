"""Pydantic v2 data models for the entire news-to-prompt-to-LLM pipeline."""

from __future__ import annotations

import json
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ─── Raw Ingestion Models ───

class RawNewsItem(BaseModel):
    """Raw news item as parsed from an RSS/Atom feed."""
    source_id: str
    title: str
    link: str
    published_at: Optional[str] = None
    raw_summary: Optional[str] = None
    url_hash: str = Field(..., description="SHA-256 hash for deduplication")


# ─── Filtered / Classified Models ───

class TickerMatch(BaseModel):
    """A single ticker match with the exact text that triggered it."""
    ticker: str
    matched_text: str
    match_type: str  # COMPANY_NAME, FULL_TICKER, EXPLICIT_SYMBOL, SAFE_LONG_TICKER


class RegexMatchResult(TickerMatch):
    """Extended ticker match result with keyword and position information."""
    matched_keyword: str = ""
    start: int = 0
    end: int = 0
    confidence: str = "HIGH"

    def __init__(
        self,
        matched_text: str = "",
        ticker: str = "",
        matched_keyword: str = "",
        start: int = 0,
        end: int = 0,
        confidence: str = "HIGH",
        match_type: str = "COMPANY_NAME",
        **data,
    ):
        super().__init__(
            ticker=ticker,
            matched_text=matched_text,
            match_type=match_type,
            matched_keyword=matched_keyword,
            start=start,
            end=end,
            confidence=confidence,
            **data,
        )


class FilteredNewsItem(BaseModel):
    """News item after regex matching, taxonomy classification, and context cropping."""
    source_id: str
    title: str
    link: str
    published_at: Optional[str] = None
    raw_summary: Optional[str] = None
    url_hash: str
    cropped_context: str
    matched_tickers: List[TickerMatch]
    matched_ticker_symbols: List[str] = Field(
        default_factory=list,
        description="Unique ticker symbols extracted from matched_tickers"
    )
    event_category: Literal[
        "M&A_CORPORATE", "EARNINGS_GUIDANCE", "CONTRACT_ORDER",
        "REGULATORY_LEGAL", "ANALYST_RATING", "GENERAL"
    ]
    priority_level: Literal[1, 2, 3]


# ─── Target LLM Response Schema ───

class AssetImpactSchema(BaseModel):
    """Impact assessment for a single matched asset."""
    ticker: str = Field(..., description="Ticker symbol from the matched_assets list")
    company_name: str
    role: Literal["PRIMARY_SUBJECT", "ACQUIRER", "TARGET", "COMPETITOR", "PARTNER"]
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="-1.0 (extremely bearish) to +1.0 (extremely bullish)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_horizon: Literal[
        "IMMEDIATE_INTRADAY", "SHORT_TERM_DAYS",
        "MEDIUM_TERM_WEEKS", "ALREADY_PRICED_IN"
    ]
    recommended_action: Literal[
        "BUY_LIMIT_INTRADAY", "BUY_LIMIT_PREMARKET", "BUY_MARKET",
        "HOLD", "SCALE_DOWN", "LIQUIDATE", "IGNORE_NOISE"
    ]
    rationale: str = Field(
        ..., max_length=200,
        description="1 concise sentence explaining the financial impact"
    )


class TargetLLMResponseSchema(BaseModel):
    """The JSON schema that will be sent to the LLM to force structured output."""
    event_type: Literal[
        "M&A", "EARNINGS_BEAT", "EARNINGS_MISS", "GUIDANCE_RAISE",
        "GUIDANCE_CUT", "MAJOR_CONTRACT", "REGULATORY", "ANALYST_UPDATE", "NOISE"
    ]
    is_price_sensitive: bool
    overall_sentiment: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    executive_summary_it: str = Field(
        ..., max_length=280,
        description="Sintesi operativa in italiano (max 2 frasi)"
    )
    impacted_assets: List[AssetImpactSchema]


# ─── Prepared Prompt Payload ───

class PreparedPromptPayload(BaseModel):
    """Complete prompt payload ready for LLM dispatch."""
    news_id: int
    priority_level: int
    system_prompt: str
    user_prompt: str
    expected_json_schema: str = Field(
        ...,
        description="JSON Schema string generated from TargetLLMResponseSchema"
    )
    estimated_tokens: int = Field(
        ...,
        description="Rough token estimate: len(total_text) // 4"
    )
    status: str = "READY_FOR_LLM"

    @staticmethod
    def generate_target_schema() -> str:
        """Generate the JSON Schema string from TargetLLMResponseSchema."""
        return json.dumps(
            TargetLLMResponseSchema.model_json_schema(),
            indent=2,
            ensure_ascii=False
        )

    @staticmethod
    def estimate_tokens(system_prompt: str, user_prompt: str, schema: str) -> int:
        """Rough token count estimate: total characters // 4."""
        total_chars = len(system_prompt) + len(user_prompt) + len(schema)
        return total_chars // 4