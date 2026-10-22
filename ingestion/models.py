"""Data models for ingestion, matching and filtering."""
from typing import Optional
from models.schemas import FilteredNewsItem, RawNewsItem, TickerMatch


class RegexMatchResult(TickerMatch):
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


__all__ = ["FilteredNewsItem", "RawNewsItem", "TickerMatch", "RegexMatchResult"]
