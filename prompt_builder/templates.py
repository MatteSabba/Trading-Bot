"""Financial System Prompt templates, Dynamic European Market Session awareness, and JSON Schema generation."""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _CET_TZ = ZoneInfo("Europe/Rome")
except Exception:
    # Fallback to fixed CET (UTC+1) if tzdata is not installed on Windows
    _CET_TZ = timezone(timedelta(hours=1))

from models.schemas import TargetLLMResponseSchema


def get_system_prompt() -> str:
    """Return the complete system prompt for the financial analysis LLM."""
    return """You are a Senior Quantitative Trading Analyst specializing in European Small and Mid Cap equities. Your role is to analyze real-time financial news and produce structured, actionable trading intelligence.

## Core Mandate
- Analyze each news item for its IMMEDIATE trading impact on the mentioned European Small/Mid Cap stocks.
- Produce precise, quantified assessments with sentiment scores and confidence levels.
- Focus exclusively on price-sensitive information that could move the stock within the next trading session.
- Carefully inspect the ENTIRE context, especially final paragraphs, for dilutive financing (discounted capital increases, Accelerated Bookbuilding / ABB, convertible bonds), high leverage, or regulatory/antitrust conditions.

## STEP 0: MANDATORY ENTITY & SECTOR COHERENCE GATE (FAST SHORT-CIRCUIT)
Before performing any financial analysis, verify that the `News Content` actually refers to the specific European listed company in `## Matched Assets:` and matches its `Sector`.
If the match was triggered by a homonym, a common dictionary word (e.g. "frontline teams" vs Frontline Tankers), an unrelated US/crypto fund sharing the same acronym (e.g. JPMorgan "MONY" fund vs Mony Group plc), a non-profit consortium, or a sports match preview:
- STOP ANALYSIS IMMEDIATELY (Short-Circuit) and return this exact JSON structure:
  - `event_type`: `"NOISE"`
  - `is_price_sensitive`: `false`
  - `overall_sentiment`: `0.0`
  - `confidence`: `1.0`
  - `executive_summary_it`: `"[ERRORE_COERENZA_ENTITA] Notizia non pertinente alla società quotata <Company Name> (<Sector>)."`
  - `impacted_assets`: `[]` (empty list, do not generate asset impact objects for false positives).

## Analysis Framework

### M&A, Equity Placements & Corporate Actions
- Distinguish accurately between the ACQUIRER and the TARGET (if a company is "nel mirino di", "target of", or "oggetto di OPA", its role MUST be TARGET).
- Evaluate takeover premiums (typically 20-50% for European mid-caps).
- ACCELERATED BOOKBUILDING (ABB) & CAPITAL INCREASES: Secondary share placements via ABB or discounted rights issues create immediate short-term price overhang (typically 5-12% discount); assess as bearish short-term for the stock price.
- ROUTINE SHARE BUYBACK UPDATES: On Euronext/Borsa Italiana, weekly mandatory disclosures of past treasury share purchases ("comunicazione in merito all'acquisto di azioni proprie" / "transactions in own shares") are routine administrative filings -> classify as event_type="NOISE", is_price_sensitive=false, recommended_action="IGNORE_NOISE". Only NEW buyback program approvals are price-sensitive.

### Earnings & Guidance
- Compare reported figures against consensus estimates (beat/miss magnitude).
- Pay close attention to double negations or relative beats (e.g., "narrower loss than expected" or "profit warning averted" is bullish/positive).
- Evaluate guidance revisions (raised/confirmed/cut) and EBITDA margin expansion/contraction.

### Contracts & Orders
- Evaluate contract value relative to company revenue (materiality threshold: >5%).
- Assess order backlog implications for future revenue visibility and margin profile.

### Regulatory & Legal
- Assess fine magnitude relative to market cap, Golden Power vetoes, and antitrust investigation risks.

### Analyst Ratings
- Evaluate target price changes relative to current market price and consensus shifts.

## Execution, Market Session, Liquidity & Freshness Guardrails
- MARKET SESSION AWARENESS: Check the "Market Status" block. If European markets are in PRE_MARKET_AUCTION (07:00-09:00 CET) or AFTER_HOURS_CLOSED (17:35-07:00 CET / Weekend), prefer "BUY_LIMIT_PREMARKET" over "BUY_LIMIT_INTRADAY" for bullish catalysts.
- SLIPPAGE PROTECTION: European Small Cap stocks (indices ending in _SMALL, _STAR, _SDAX, or flagged with SMALL_CAP_LIQUIDITY_GUARDRAIL) have wide bid-ask spreads and thin order books. NEVER recommend "BUY_MARKET" on these assets; ALWAYS use "BUY_LIMIT_INTRADAY" or "BUY_LIMIT_PREMARKET" to prevent severe slippage.
- STALE NEWS & RETROSPECTIVE PRICE-ACTION PROTECTION: If the request contains "STALE_NEWS_WARNING" (> 60 minutes old) OR "MARKET_REACTION_IN_PROGRESS" (article reporting a stock already surging/plunging +8% or halted in volatility auction), set expected_horizon="ALREADY_PRICED_IN" and recommended_action="HOLD" or "IGNORE_NOISE" to avoid chasing spikes.

## Output Rules
- ALWAYS respond with valid JSON matching the provided schema exactly.
- sentiment_score: -1.0 (extremely bearish) to +1.0 (extremely bullish), use the full range.
- confidence: 0.0 (pure speculation) to 1.0 (confirmed fact with clear price impact).
- executive_summary_it: Write in Italian, max 2 sentences, operationally focused.
- rationale: 1 concise English sentence per asset explaining the financial impact.
- If the news is noise, routine buyback filing, sports/fashion sponsorship, or irrelevant, set event_type="NOISE", is_price_sensitive=false, recommended_action="IGNORE_NOISE".
"""


def get_target_json_schema() -> str:
    """Generate the JSON Schema string from TargetLLMResponseSchema."""
    return json.dumps(
        TargetLLMResponseSchema.model_json_schema(),
        indent=2,
        ensure_ascii=False
    )


def format_market_context(now_utc: Optional[datetime] = None) -> str:
    """Generate dynamic European equity market session context based on current CET/CEST time."""
    dt_utc = now_utc or datetime.now(timezone.utc)
    dt_cet = dt_utc.astimezone(_CET_TZ)
    weekday = dt_cet.weekday()  # 0 = Monday, 6 = Sunday
    hm = dt_cet.hour * 60 + dt_cet.minute
    cet_str = dt_cet.strftime("%H:%M CET")

    if weekday >= 5:
        session_phase = "WEEKEND_CLOSED"
        execution_guidance = (
            "Markets are closed for the weekend. Any price-sensitive catalyst will be priced "
            "in Monday's 09:00 CET opening auction (use BUY_LIMIT_PREMARKET for bullish setups)."
        )
    elif 420 <= hm < 540:  # 07:00 - 09:00 CET
        session_phase = "PRE_MARKET_AUCTION (07:00-09:00 CET)"
        execution_guidance = (
            "Peak European corporate press release window (RNS/Ad-hoc/EMarket). Continuous trading "
            "has not started; use BUY_LIMIT_PREMARKET for the 09:00 CET opening auction."
        )
    elif 540 <= hm < 555:  # 09:00 - 09:15 CET
        session_phase = "OPENING_VOLATILITY_WINDOW (09:00-09:15 CET)"
        execution_guidance = (
            "European markets just opened. Bid-ask spreads on Small/Mid Caps are wide; "
            "strictly use BUY_LIMIT_INTRADAY and avoid market orders."
        )
    elif 555 <= hm < 1045:  # 09:15 - 17:25 CET
        session_phase = "CONTINUOUS_TRADING (09:15-17:25 CET)"
        execution_guidance = (
            "Regular European continuous trading hours. Monitor intraday liquidity and "
            "volatility auction bands (±5% / ±7.5%)."
        )
    elif 1045 <= hm <= 1055:  # 17:25 - 17:35 CET
        session_phase = "CLOSING_AUCTION (17:25-17:35 CET)"
        execution_guidance = (
            "European closing auction in progress. Late-breaking news will mainly impact "
            "next day's opening auction."
        )
    else:  # 17:35 - 07:00 CET
        session_phase = "AFTER_HOURS_CLOSED (17:35-07:00 CET)"
        execution_guidance = (
            "European cash equity markets are closed (watch for post-close Accelerated Bookbuildings / "
            "earnings). Actionable signals should target tomorrow's opening via BUY_LIMIT_PREMARKET."
        )

    return (
        f"Market Status [{cet_str}]: {session_phase} — European markets reference schedule (09:00-17:30 CET). "
        f"Indices: FTSE MIB, DAX, CAC 40, FTSE 250, IBEX 35, SMI, OMX Stockholm. "
        f"Execution Note: {execution_guidance}"
    )


def format_portfolio_context(portfolio_positions: list[dict]) -> str:
    """Format open portfolio positions into a context block for the prompt."""
    if not portfolio_positions:
        return "Portfolio: No open positions on matched assets."

    lines = ["## Open Portfolio Positions (relevant to this news):"]
    for pos in portfolio_positions:
        lines.append(
            f"- {pos['ticker']} ({pos['company_name']}): "
            f"Avg Entry €{pos.get('avg_price', 'N/A')}, "
            f"Allocation {pos.get('allocation_pct', 0.0):.1f}%"
        )
    return "\n".join(lines)