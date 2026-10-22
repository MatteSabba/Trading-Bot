"""Client asincrono per l'inferenza su LLM locale (Ollama / vLLM) con sessione HTTP persistente e fallback.

Caratteristiche architetturali:
- Sessione aiohttp.ClientSession persistente (TCP Keep-Alive verso localhost:11434)
- Pinning del modello in VRAM tramite parametro 'keep_alive': -1 (zero latenza di ricaricamento)
- Controllo della concorrenza GPU tramite asyncio.Semaphore (default: 2 canali)
- Enforcement nativo del JSON Schema (TargetLLMResponseSchema) nel parametro 'format'
- Fallback euristico avanzato:
  1. Gestione negazioni di prossimità e smentite ufficiali
  2. Riconoscimento passivo M&A (role="TARGET" vs "ACQUIRER")
  3. Gestione di Accelerated Bookbuilding (ABB), comunicati di routine su azioni proprie
     e notizie di cronaca a movimento già avvenuto (MARKET_REACTION_IN_PROGRESS).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import aiohttp

from models.schemas import AssetImpactSchema, TargetLLMResponseSchema

logger = logging.getLogger("trading_bot.llm_client")

_BULLISH_KEYWORDS = {
    "opa", "offerta pubblica", "acquisizione", "acquire", "acquires", "acquisition", "takeover", "übernahme",
    "rachat", "accordo", "agreement", "unanimità", "unanimously", "crescita",
    "growth", "record", "surged", "beat", "supera", "übertreffen", "supérieurs",
    "upgrade", "upgrades", "raised", "alza", "erhöht", "commessa", "contract",
    "awarded", "aggiudica", "auftrag", "approval", "approvato", "dividendo",
    "dividend", "ebitda", "incremento", "accrescitivo", "accretive", "buyout",
}

_BEARISH_KEYWORDS = {
    "profit warning", "warning", "calo", "perdita", "loss", "miss", "inferiore",
    "downgrade", "downgrades", "cut", "taglia", "riduce", "senkt", "multa",
    "fine", "indagine", "investigation", "antitrust", "delisting", "debito",
    "debt", "default", "insolvenza", "contrazione", "rincari", "sanzione",
    "aumento di capitale", "collocamento accelerato", "accelerated bookbuilding",
    "diluizione", "dilution",
}

_BULLISH_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(kw) for kw in sorted(_BULLISH_KEYWORDS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)
_BEARISH_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(kw) for kw in sorted(_BEARISH_KEYWORDS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)

_NEGATED_BEARISH_PATTERNS = re.compile(
    r'(?:'
    r'(?:scongiura\w*|evita\w*|esclude\w*|nessun\w*\s+impatto\s+da|no|without|avoids?|averted)\s+(?:\w+\s+){0,3}'
    r'(?:profit\s+warning|perdita|loss|multa|sanzione|rincari|calo|default|debito)'
    r'|'
    r'(?:perdita|loss|calo|debito)\s+(?:netta\s+)?(?:inferiore|minore|narrower|smaller|better)\s+(?:alle\s+attese|than\s+expected|del\s+previsto)'
    r')',
    re.IGNORECASE,
)

_DENIED_BULLISH_PATTERNS = re.compile(
    r'(?:smentisc\w*|nega\w*|esclude\w*|denies|denied|rules\s+out|no\s+ongoing)\s+(?:\w+\s+){0,4}'
    r'(?:opa|acquisizione|fusione|takeover|merger|trattative|talks|dividendo|accordo)',
    re.IGNORECASE,
)

_MA_TARGET_PATTERNS = re.compile(
    r'(?:nel\s+mirino\s+di|oggetto\s+di\s+opa|opa\s+su|offerta\s+pubblica\s+su|'
    r'acquisita\s+da|rilevata\s+da|target\s+of|to\s+be\s+acquired\s+by|buyout\s+of|'
    r'takeover\s+bid\s+for)',
    re.IGNORECASE,
)

_ABB_PATTERNS = re.compile(
    r'\b(?:accelerated\s+bookbuilding|collocamento\s+accelerato)\b',
    re.IGNORECASE,
)

_CATEGORY_TO_EVENT_MAP: Dict[str, Tuple[str, str]] = {
    "M&A_CORPORATE": ("M&A", "M&A"),
    "EARNINGS_GUIDANCE": ("EARNINGS_BEAT", "EARNINGS_MISS"),
    "CONTRACT_ORDER": ("MAJOR_CONTRACT", "MAJOR_CONTRACT"),
    "REGULATORY_LEGAL": ("REGULATORY", "REGULATORY"),
    "ANALYST_RATING": ("ANALYST_UPDATE", "ANALYST_UPDATE"),
    "GENERAL": ("ANALYST_UPDATE", "NOISE"),
}

_ASSET_LINE_REGEX = re.compile(
    r"^-\s+([A-Z0-9.\-]+):\s+(.+?)\s+\(Sector:\s*(.+?),\s*Index:\s*(.+?)\)\s*$",
    re.MULTILINE,
)
_CATEGORY_LINE_REGEX = re.compile(r"^Event Category:\s*(\S+)", re.MULTILINE)
_TITLE_LINE_REGEX = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)
_CONTEXT_BLOCK_REGEX = re.compile(
    r"## News Content\s*(.+?)\s*## Matched Assets:", re.DOTALL
)


class LocalLLMClient:
    """Client asincrono per Ollama (/api/chat) con sessione HTTP persistente e fallback."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:11434/api/chat",
        model_name: str = "qwen2.5:7b",
        max_concurrent_channels: int = 2,
        request_timeout_sec: int = 90,
        temperature: float = 0.1,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.max_concurrent_channels = max_concurrent_channels
        self.request_timeout_sec = request_timeout_sec
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(max_concurrent_channels)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Restituisce o inizializza la sessione aiohttp persistente (Keep-Alive)."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_sec)
            connector = aiohttp.TCPConnector(limit=self.max_concurrent_channels + 2)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        """Chiude la sessione HTTP persistente."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def evaluate_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        expected_json_schema: str,
    ) -> Tuple[str, int, bool]:
        """Invia il prompt al server LLM locale o attiva il fallback euristico."""
        async with self._semaphore:
            start_ns = time.perf_counter_ns()

            try:
                schema_dict = json.loads(expected_json_schema)
            except json.JSONDecodeError:
                schema_dict = TargetLLMResponseSchema.model_json_schema()

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": schema_dict,
                "keep_alive": -1,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": 4096,
                },
            }

            try:
                session = await self._get_session()
                async with session.post(self.endpoint_url, json=payload) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        logger.warning(
                            f"Server LLM ha risposto con HTTP {response.status} "
                            f"({error_body[:120]}). Attivazione fallback euristico."
                        )
                        fallback_json = self._generate_heuristic_fallback(user_prompt)
                        elapsed_ms = max(1, (time.perf_counter_ns() - start_ns) // 1_000_000)
                        return fallback_json, elapsed_ms, True

                    data = await response.json()
                    raw_content = data.get("message", {}).get("content", "").strip()

                    if not raw_content:
                        raise ValueError("Risposta vuota ricevuta dal nodo LLM.")

                    elapsed_ms = max(1, (time.perf_counter_ns() - start_ns) // 1_000_000)
                    return raw_content, elapsed_ms, False

            except (
                aiohttp.ClientConnectorError,
                ConnectionRefusedError,
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                ValueError,
            ) as exc:
                logger.info(
                    f"Server LLM locale non raggiungibile su {self.endpoint_url} "
                    f"({type(exc).__name__}). Generazione mock JSON euristico."
                )
                fallback_json = self._generate_heuristic_fallback(user_prompt)
                elapsed_ms = max(1, (time.perf_counter_ns() - start_ns) // 1_000_000)
                return fallback_json, elapsed_ms, True

    def _generate_heuristic_fallback(self, user_prompt: str) -> str:
        """Costruisce un JSON valido conforme a TargetLLMResponseSchema tramite analisi lessicale."""
        content_match = _CONTEXT_BLOCK_REGEX.search(user_prompt)
        news_section = content_match.group(1) if content_match else user_prompt
        news_lower = news_section.lower()

        cat_match = _CATEGORY_LINE_REGEX.search(user_prompt)
        raw_category = cat_match.group(1).strip() if cat_match else "GENERAL"

        title_match = _TITLE_LINE_REGEX.search(user_prompt)
        news_title = title_match.group(1).strip() if title_match else "Notizia finanziaria rilevata"

        is_routine_buyback = "ROUTINE_BUYBACK_DISCLOSURE" in user_prompt
        is_reaction_in_progress = "MARKET_REACTION_IN_PROGRESS" in user_prompt
        is_stale = "STALE_NEWS_WARNING" in user_prompt
        is_closed_session = any(
            tag in user_prompt
            for tag in ("PRE_MARKET_AUCTION", "AFTER_HOURS_CLOSED", "WEEKEND_CLOSED")
        )

        # 1. Rilevamento negazioni di prossimità, smentite e ABB
        negated_bearish_count = len(_NEGATED_BEARISH_PATTERNS.findall(news_section))
        denied_bullish_count = len(_DENIED_BULLISH_PATTERNS.findall(news_section))
        has_abb = bool(_ABB_PATTERNS.search(news_section))

        bullish_hits = len(_BULLISH_PATTERN.findall(news_section))
        bearish_hits = len(_BEARISH_PATTERN.findall(news_section))

        if negated_bearish_count > 0:
            bearish_hits = max(0, bearish_hits - (negated_bearish_count * 2))
            bullish_hits += negated_bearish_count

        if denied_bullish_count > 0:
            bullish_hits = max(0, bullish_hits - (denied_bullish_count * 2))
            bearish_hits += denied_bullish_count

        if has_abb:
            bearish_hits += 3

        raw_delta = bullish_hits - bearish_hits
        if is_routine_buyback:
            sentiment = 0.05
        elif raw_delta > 0:
            sentiment = min(0.90, round(0.25 + (raw_delta * 0.15), 2))
        elif raw_delta < 0:
            sentiment = max(-0.90, round(-0.25 + (raw_delta * 0.15), 2))
        else:
            sentiment = 0.35 if raw_category in ("M&A_CORPORATE", "CONTRACT_ORDER") else 0.0

        # 2. Determinazione event_type
        pos_event, neg_event = _CATEGORY_TO_EVENT_MAP.get(
            raw_category, ("ANALYST_UPDATE", "NOISE")
        )
        event_type_str = pos_event if sentiment >= 0.0 else neg_event

        if (
            is_routine_buyback
            or (raw_category == "GENERAL" and abs(sentiment) < 0.2)
            or denied_bullish_count > 0
        ):
            event_type_str = "NOISE"

        event_type: Literal[
            "M&A", "EARNINGS_BEAT", "EARNINGS_MISS", "GUIDANCE_RAISE",
            "GUIDANCE_CUT", "MAJOR_CONTRACT", "REGULATORY", "ANALYST_UPDATE", "NOISE"
        ] = event_type_str  # type: ignore[assignment]

        is_price_sensitive = (
            event_type != "NOISE"
            and abs(sentiment) >= 0.25
            and not is_stale
            and not is_reaction_in_progress
        )
        confidence = round(min(0.88, 0.55 + (bullish_hits + bearish_hits) * 0.05), 2)

        # 3. Estrazione degli asset dal blocco '## Matched Assets:'
        extracted_assets = _ASSET_LINE_REGEX.findall(user_prompt)
        if not extracted_assets:
            extracted_assets = [("UNKNOWN.EU", "European Listed Entity", "General", "EU")]

        is_ma_target = bool(_MA_TARGET_PATTERNS.search(news_section))

        impacted_assets: List[AssetImpactSchema] = []
        for idx, (ticker, company_name, _sector, index_country) in enumerate(extracted_assets):
            if event_type == "M&A" and idx == 0:
                role: Literal["PRIMARY_SUBJECT", "ACQUIRER", "TARGET", "COMPETITOR", "PARTNER"] = (
                    "TARGET" if is_ma_target else "ACQUIRER"
                )
            else:
                role = "PRIMARY_SUBJECT" if idx == 0 else "PARTNER"

            if is_routine_buyback:
                action: Literal[
                    "BUY_LIMIT_INTRADAY", "BUY_LIMIT_PREMARKET", "BUY_MARKET",
                    "HOLD", "SCALE_DOWN", "LIQUIDATE", "IGNORE_NOISE"
                ] = "IGNORE_NOISE"
                horizon: Literal[
                    "IMMEDIATE_INTRADAY", "SHORT_TERM_DAYS",
                    "MEDIUM_TERM_WEEKS", "ALREADY_PRICED_IN"
                ] = "ALREADY_PRICED_IN"
            elif is_stale or is_reaction_in_progress:
                action = "HOLD"
                horizon = "ALREADY_PRICED_IN"
            elif event_type == "NOISE":
                action = "IGNORE_NOISE"
                horizon = "ALREADY_PRICED_IN"
            elif sentiment >= 0.55:
                action = "BUY_LIMIT_PREMARKET" if is_closed_session else "BUY_LIMIT_INTRADAY"
                horizon = "IMMEDIATE_INTRADAY"
            elif sentiment >= 0.25:
                action = "BUY_LIMIT_PREMARKET"
                horizon = "SHORT_TERM_DAYS"
            elif sentiment <= -0.60:
                action = "LIQUIDATE"
                horizon = "IMMEDIATE_INTRADAY"
            elif sentiment <= -0.25:
                action = "SCALE_DOWN"
                horizon = "SHORT_TERM_DAYS"
            else:
                action = "HOLD"
                horizon = "MEDIUM_TERM_WEEKS"

            direction_label = (
                "positive" if sentiment > 0.1 else ("negative" if sentiment < -0.1 else "neutral")
            )
            rationale_text = (
                f"Heuristic assessment ({role}): {event_type} catalyst indicates "
                f"{direction_label} impact for {company_name} ({ticker}, {index_country})."
            )[:200]

            impacted_assets.append(
                AssetImpactSchema(
                    ticker=ticker.strip(),
                    company_name=company_name.strip(),
                    role=role,
                    sentiment_score=sentiment,
                    confidence=confidence,
                    expected_horizon=horizon,
                    recommended_action=action,
                    rationale=rationale_text,
                )
            )

        tickers_str = ", ".join(a.ticker for a in impacted_assets)
        sentiment_label_it = (
            "rialzista" if sentiment > 0.15 else ("ribassista" if sentiment < -0.15 else "neutrale")
        )
        short_title = news_title[:105] + ("..." if len(news_title) > 105 else "")
        exec_summary = (
            f"[FALLBACK EURISTICO] Rilevato evento {event_type} su {tickers_str} "
            f"con sentiment {sentiment_label_it} ({sentiment:+.2f}). "
            f"Sintesi notizia: {short_title}."
        )[:280]

        response_model = TargetLLMResponseSchema(
            event_type=event_type,
            is_price_sensitive=is_price_sensitive,
            overall_sentiment=sentiment,
            confidence=confidence,
            executive_summary_it=exec_summary,
            impacted_assets=impacted_assets,
        )

        return response_model.model_dump_json(indent=2)