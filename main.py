"""Orchestratore Asincrono Event-Driven (Producer-Consumer-LLMWorker) e CLI di Gestione.

Pipeline:
1. Ingestion: RSSPoller con sessioni persistenti, jitter anti-ban e pre-warming LRU
2. Cleaning & Cropping: text_cleaner (selectolax C-level + Head/Match/Tail sentence-safe)
3. Matching & Filtering: RegexEngine a 2 stadi (FlashText O(N) + Lookup O(1) per ticker)
4. Taxonomy Classification: TaxonomyClassifier multilingua ponderato con Priority Level 1-3
5. Queue & Syndication Dedup: PriorityQueue asincrona + filtro anti-duplicati cross-sorgente
6. Prompt Building: PromptBuilder con contesto mercati CET dinamico e guardrail microstruttura
7. Event-Driven LLM Dispatch: notifica immediata via asyncio.Event al worker GPU (0ms wait)
8. Inferenza LLM (Fase 2): LocalLLMClient valida con Pydantic v2 e salva in llm_evaluations.
"""

import argparse
import asyncio
import itertools
import json
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

from pydantic import ValidationError

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DB_PATH,
    DEFAULT_MAX_LLM_PRIORITY,
    SEMANTIC_DEDUP_JACCARD_THRESHOLD,
    SEMANTIC_DEDUP_WINDOW_MINUTES,
)
from database.db_manager import DatabaseManager
from database.seeder import seed_all
from filtering.regex_engine import RegexEngine
from filtering.taxonomy_classifier import TaxonomyClassifier
from ingestion.article_extractor import ArticleExtractor
from ingestion.rss_poller import RSSPoller, compute_url_hash
from ingestion.text_cleaner import clean_text, combine_title_and_summary, crop_context
from llm.local_client import LocalLLMClient
from models.schemas import (
    FilteredNewsItem,
    RawNewsItem,
    TargetLLMResponseSchema,
    TickerMatch,
)
from prompt_builder.builder import PromptBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trading_bot.orchestrator")

_WORD_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9À-ÿ]{3,}\b")
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_IGNORE_YEARS = {"2024", "2025", "2026", "2027", "2028", "1", "2", "3", "4", "9"}


def _tokenize_title(title: str) -> Set[str]:
    """Estrae i token significativi (>= 3 caratteri) da un titolo per il calcolo di Jaccard."""
    return {w.lower() for w in _WORD_TOKEN_RE.findall(title)}


def _extract_numeric_tokens(text: str) -> Set[str]:
    """Estrae le cifre numeriche specifiche (es. '21.4', '47', '52', '195') normalizzando la virgola.
    Usato per intercettare duplicati cross-lingua (EN vs FR/DE) dello stesso comunicato."""
    raw_nums = {n.replace(",", ".") for n in _NUMERIC_TOKEN_RE.findall(text)}
    return {n for n in raw_nums if n not in _IGNORE_YEARS}


class NewsPipelineOrchestrator:
    """Event-Driven Producer-Consumer-LLM Pipeline Orchestrator with PriorityQueue."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        llm_client: Optional[LocalLLMClient] = None,
        max_llm_priority: int = DEFAULT_MAX_LLM_PRIORITY,
    ):
        self.db = db or DatabaseManager()
        self.poller = RSSPoller(db=self.db)
        self.article_extractor = ArticleExtractor()
        self.regex_engine = RegexEngine(db=self.db)
        self.classifier = TaxonomyClassifier(db=self.db)
        self.prompt_builder = PromptBuilder(db=self.db)
        self.llm_client = llm_client or LocalLLMClient()
        self.max_llm_priority = max_llm_priority
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._queue_counter = itertools.count()
        self._prompt_ready_event: asyncio.Event = asyncio.Event()
        self._running = False

    def initialize(self):
        """Pre-carica regole RegEx, stato portafoglio e cache LRU url_hash in memoria RAM."""
        logger.info("Inizializzazione componenti della pipeline...")
        self.db.initialize_schema()
        rule_count = self.regex_engine.load_rules()
        self.classifier.load_portfolio_state()
        warmed_hashes = self.poller.warm_dedup_cache()
        logger.info(
            f"Regole caricate: {rule_count} | Hash pre-caricati in LRU: {warmed_hashes}. "
            f"Orchestratore pronto."
        )

    def _is_cross_source_duplicate(
        self, filtered: FilteredNewsItem, current_news_id: int
    ) -> bool:
        """Verifica se la notizia è un duplicato cross-feed o cross-lingua di un evento già inviato a LLM.

        Controlla le notizie archiviate negli ultimi SEMANTIC_DEDUP_WINDOW_MINUTES (45 min)
        con intersezione di ticker ed uguale event_category.
        Rileva il duplicato se:
        1. La similarità Jaccard dei titoli >= SEMANTIC_DEDUP_JACCARD_THRESHOLD (0.55), OPPURE
        2. Condivide numeri chiave specifici (_extract_numeric_tokens) sullo stesso ticker/categoria, OPPURE
        3. Appartiene a una categoria primaria (EARNINGS_GUIDANCE / M&A_CORPORATE) esattamente sullo stesso
           singolo ticker entro la finestra di 45 minuti.
        Se il nuovo articolo ha un contesto molto più ricco (>= 2x e >= 600 char) e il vecchio prompt è
        ancora in stato 'READY_FOR_LLM', aggiorna il prompt esistente (Context Upgrade) prima di restituire True.
        """
        recent_rows = self.db.fetch_all(
            f"""SELECT n.id AS news_id, n.title, n.cropped_context, n.matched_tickers_json,
                       p.id AS prompt_id, p.status AS prompt_status
                FROM news_archive n
                JOIN prepared_prompts p ON p.news_id = n.id
                WHERE n.id != ?
                  AND n.event_category = ?
                  AND n.ingested_at >= datetime('now', '-{int(SEMANTIC_DEDUP_WINDOW_MINUTES)} minutes')
                ORDER BY n.id DESC LIMIT 35""",
            (current_news_id, filtered.event_category),
        )
        if not recent_rows:
            return False

        target_tickers = set(filtered.matched_ticker_symbols)
        new_tokens = _tokenize_title(filtered.title)
        new_numbers = _extract_numeric_tokens(f"{filtered.title} {filtered.cropped_context[:400]}")

        for r in recent_rows:
            try:
                existing_tickers = set(json.loads(r["matched_tickers_json"]))
            except Exception:
                continue

            if not (existing_tickers & target_tickers):
                continue

            existing_tokens = _tokenize_title(r["title"])
            intersection = len(new_tokens & existing_tokens) if (new_tokens and existing_tokens) else 0
            union = len(new_tokens | existing_tokens) if (new_tokens and existing_tokens) else 0
            jaccard = intersection / union if union > 0 else 0.0

            existing_numbers = _extract_numeric_tokens(
                f"{r['title']} {(r['cropped_context'] or '')[:400]}"
            )
            shared_numbers = new_numbers & existing_numbers

            is_same_single_ticker_p1 = (
                filtered.event_category in ("EARNINGS_GUIDANCE", "M&A_CORPORATE")
                and len(target_tickers) == 1
                and target_tickers == existing_tickers
            )

            is_duplicate = (
                jaccard >= SEMANTIC_DEDUP_JACCARD_THRESHOLD
                or bool(shared_numbers)
                or is_same_single_ticker_p1
            )

            if is_duplicate:
                old_ctx = r["cropped_context"] or ""
                new_ctx = filtered.cropped_context or ""
                if (
                    r["prompt_status"] == "READY_FOR_LLM"
                    and len(new_ctx) >= max(600, len(old_ctx) * 2)
                ):
                    new_payload = self.prompt_builder.build_prompt(filtered, r["news_id"])
                    self.db.execute(
                        """UPDATE prepared_prompts
                           SET user_prompt = ?, estimated_tokens = ?
                           WHERE id = ?""",
                        (new_payload.user_prompt, new_payload.estimated_tokens, r["prompt_id"]),
                    )
                    logger.info(
                        f"[CONTEXT UPGRADE] Prompt #{r['prompt_id']} aggiornato con testo più ricco "
                        f"da News #{current_news_id} ({len(old_ctx)} -> {len(new_ctx)} chars)."
                    )
                else:
                    logger.info(
                        f"[SYNDICATION DEDUP] Notizia #{current_news_id} duplicata di News #{r['news_id']} "
                        f"(Jaccard={jaccard:.0%}, SharedNums={shared_numbers or 'N/A'}) "
                        f"su ticker {filtered.matched_ticker_symbols}. Prompt LLM omesso."
                    )
                return True

        return False

    def process_raw_item(self, raw: RawNewsItem) -> Optional[FilteredNewsItem]:
        """Elabora un RawNewsItem: cleaning, anti-duplicazione titolo/summary, regex matching, cropping e tassonomia."""
        clean_title = clean_text(raw.title)
        clean_summary = clean_text(raw.raw_summary or "")
        combined_text = combine_title_and_summary(clean_title, clean_summary)

        # Match a due stadi con lookup O(1) sui ticker candidati, filtro Form 8.3/SEO e disambiguazione broker
        matches = self.regex_engine.match(combined_text)
        if not matches:
            return None

        matched_symbols = sorted({m.ticker for m in matches})
        match_terms = [m.matched_text for m in matches]

        # Context Window Cropping (Head + Match + Tail)
        cropped = crop_context(combined_text, match_terms)

        # Tassonomia ponderata e assegnazione priorità
        event_cat, priority = self.classifier.classify(combined_text, matched_symbols)

        return FilteredNewsItem(
            source_id=raw.source_id,
            title=clean_title,
            link=raw.link,
            published_at=raw.published_at,
            raw_summary=clean_summary,
            url_hash=raw.url_hash,
            cropped_context=cropped,
            matched_tickers=matches,
            matched_ticker_symbols=matched_symbols,
            event_category=event_cat,
            priority_level=priority,
        )

    async def enrich_filtered_item(self, filtered: FilteredNewsItem) -> FilteredNewsItem:
        """Estrae il full-text dalla pagina web se il summary RSS è corto (< 1200 char) o ripete il titolo.

        Ottimizzazione anti-rate-limit: per i link di Google News (che richiedono risoluzione RPC),
        esegue l'estrazione solo se la notizia ha priorità <= max_llm_priority (evitando chiamate
        inutili sulle notizie GENERAL di Priorità 3 che verranno solo archiviate).
        """
        if (
            "news.google.com" in filtered.link
            and filtered.priority_level > self.max_llm_priority
        ):
            return filtered

        extracted_raw = await self.article_extractor.extract_if_needed(
            title=filtered.title,
            clean_summary=filtered.raw_summary or "",
            url=filtered.link,
        )
        cleaned_extracted = clean_text(extracted_raw)
        if cleaned_extracted and len(cleaned_extracted) > len(filtered.raw_summary or ""):
            combined_text = combine_title_and_summary(filtered.title, cleaned_extracted)
            match_terms = [m.matched_text for m in filtered.matched_tickers]
            cropped = crop_context(combined_text, match_terms)
            event_cat, priority = self.classifier.classify(
                combined_text, filtered.matched_ticker_symbols
            )
            filtered.raw_summary = cleaned_extracted
            filtered.cropped_context = cropped
            filtered.event_category = event_cat
            filtered.priority_level = priority
        return filtered

    async def producer(self, poll_interval: int = 60):
        """Producer task: interroga le sorgenti RSS e inserisce gli item filtrati nella PriorityQueue."""
        logger.info(f"Producer avviato (intervallo ciclo: {poll_interval}s)")
        while self._running:
            try:
                logger.info("Polling sorgenti RSS attive e scadute in corso...")
                raw_items = await self.poller.poll_all_active()
                logger.info(f"Ricevuti {len(raw_items)} articoli freschi dai feed.")

                processed_count = 0
                for raw in raw_items:
                    filtered = self.process_raw_item(raw)
                    if filtered:
                        filtered = await self.enrich_filtered_item(filtered)
                        await self.queue.put(
                            (filtered.priority_level, next(self._queue_counter), filtered)
                        )
                        processed_count += 1

                logger.info(
                    f"Filtrati e accodati {processed_count} articoli rilevanti per Small/Mid Cap."
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Errore durante il ciclo di produzione: {e}", exc_info=True)

            await asyncio.sleep(poll_interval)

    async def consumer(self):
        """Consumer task: estrae dalla PriorityQueue, archivia, applica il gate di priorità, genera il Prompt e sveglia il worker LLM."""
        logger.info("Consumer avviato in attesa di item nella PriorityQueue...")
        while self._running:
            try:
                priority, _count, filtered = await asyncio.wait_for(
                    self.queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                logger.info(
                    f"Consumando notizia priorità {priority} [{filtered.event_category}]: "
                    f"'{filtered.title[:60]}...' (Ticker: {filtered.matched_ticker_symbols})"
                )

                cursor = self.db.execute(
                    """INSERT OR IGNORE INTO news_archive
                       (url_hash, source_id, title, link, published_at, raw_summary,
                        cropped_context, matched_tickers_json, event_category, priority_level)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        filtered.url_hash,
                        filtered.source_id,
                        filtered.title,
                        filtered.link,
                        filtered.published_at,
                        filtered.raw_summary or "",
                        filtered.cropped_context,
                        json.dumps(filtered.matched_ticker_symbols),
                        filtered.event_category,
                        filtered.priority_level,
                    ),
                )
                news_id = cursor.lastrowid
                if not news_id:
                    row = self.db.fetch_one(
                        "SELECT id FROM news_archive WHERE url_hash = ?", (filtered.url_hash,)
                    )
                    news_id = row["id"] if row else None

                if news_id:
                    if filtered.priority_level > self.max_llm_priority:
                        logger.info(
                            f"[ARCHIVED ONLY - PRIO {filtered.priority_level}] "
                            f"News #{news_id} archiviata senza prompt LLM "
                            f"(soglia max_llm_priority={self.max_llm_priority})."
                        )
                        continue

                    existing_prompt = self.db.fetch_one(
                        "SELECT id FROM prepared_prompts WHERE news_id = ?", (news_id,)
                    )
                    if not existing_prompt and not self._is_cross_source_duplicate(filtered, news_id):
                        prompt_id = self.prompt_builder.process_news_item(filtered, news_id)
                        self._prompt_ready_event.set()
                        logger.info(
                            f"[PROMPT CREATO] ID: {prompt_id} per News #{news_id} | "
                            f"Stato: READY_FOR_LLM | Priorità: {filtered.priority_level}"
                        )
                    else:
                        logger.debug(f"Prompt già esistente o duplicato semantico per News #{news_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Errore durante l'elaborazione del consumer: {e}", exc_info=True)
            finally:
                self.queue.task_done()

    async def _evaluate_single_prompt_row(self, row: sqlite3.Row) -> bool:
        """Elabora un singolo record di prepared_prompts tramite LocalLLMClient e salva il risultato."""
        prompt_id: int = row["id"]
        news_id: int = row["news_id"]
        priority: int = row["priority_level"]

        try:
            raw_json, inference_time_ms, is_fallback = await self.llm_client.evaluate_prompt(
                system_prompt=row["system_prompt"],
                user_prompt=row["user_prompt"],
                expected_json_schema=row["expected_json_schema"],
            )

            validated: TargetLLMResponseSchema = TargetLLMResponseSchema.model_validate_json(
                raw_json
            )

            impacted_assets_serialized = json.dumps(
                [asset.model_dump() for asset in validated.impacted_assets],
                ensure_ascii=False,
            )

            self.db.execute(
                """INSERT OR REPLACE INTO llm_evaluations
                   (news_id, prompt_id, event_type, is_price_sensitive,
                    overall_sentiment, confidence, executive_summary_it,
                    impacted_assets_json, inference_time_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    news_id,
                    prompt_id,
                    validated.event_type,
                    1 if validated.is_price_sensitive else 0,
                    validated.overall_sentiment,
                    validated.confidence,
                    validated.executive_summary_it,
                    impacted_assets_serialized,
                    inference_time_ms,
                ),
            )

            self.db.execute(
                "UPDATE prepared_prompts SET status = 'PROCESSED' WHERE id = ?",
                (prompt_id,),
            )

            mode_tag = "FALLBACK-MOCK" if is_fallback else "OLLAMA-GPU"
            logger.info(
                f"[LLM {mode_tag}] Prompt #{prompt_id} (Prio {priority}) -> "
                f"Evento: {validated.event_type} | Sentiment: {validated.overall_sentiment:+.2f} | "
                f"PriceSensitive: {validated.is_price_sensitive} | Tempo: {inference_time_ms}ms"
            )
            return True

        except (ValidationError, json.JSONDecodeError) as val_err:
            logger.error(
                f"[VALIDATION ERROR] Validazione JSON fallita per Prompt #{prompt_id} (News #{news_id}): {val_err}"
            )
            self.db.execute(
                "UPDATE prepared_prompts SET status = 'FAILED' WHERE id = ?",
                (prompt_id,),
            )
            return False
        except Exception as exc:
            logger.error(
                f"[ERROR] Errore imprevisto su Prompt #{prompt_id} (News #{news_id}): {exc}",
                exc_info=True,
            )
            self.db.execute(
                "UPDATE prepared_prompts SET status = 'FAILED' WHERE id = ?",
                (prompt_id,),
            )
            return False

    async def llm_worker(self, poll_interval: int = 5):
        """Worker asincrono di Fase 2 con wake-up event-driven istantaneo e fallback di polling."""
        logger.info(
            f"LLM Worker avviato (wake-up: Event-Driven + fallback {poll_interval}s, "
            f"max canali GPU concorrenti: {self.llm_client.max_concurrent_channels})"
        )
        while self._running:
            try:
                pending_rows = self.db.fetch_all(
                    """SELECT id, news_id, priority_level, system_prompt,
                              user_prompt, expected_json_schema
                       FROM prepared_prompts
                       WHERE status = 'READY_FOR_LLM'
                       ORDER BY priority_level ASC, id ASC
                       LIMIT 10"""
                )

                if pending_rows:
                    logger.info(
                        f"Trovati {len(pending_rows)} prompt in stato 'READY_FOR_LLM'. "
                        f"Avvio valutazione..."
                    )
                    tasks = [self._evaluate_single_prompt_row(row) for row in pending_rows]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    if len(pending_rows) == 10:
                        continue

                # Attende l'evento dal consumer (latenza <0.05ms) o il timeout di sicurezza
                self._prompt_ready_event.clear()
                try:
                    await asyncio.wait_for(
                        self._prompt_ready_event.wait(), timeout=float(poll_interval)
                    )
                except asyncio.TimeoutError:
                    pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Errore nel ciclo llm_worker: {e}", exc_info=True)
                await asyncio.sleep(poll_interval)

    async def run(self, poll_interval: int = 60):
        """Avvia l'orchestratore asincrono Producer-Consumer (Fase 1)."""
        self._running = True
        self.initialize()
        producer_task = asyncio.create_task(self.producer(poll_interval))
        consumer_task = asyncio.create_task(self.consumer())

        try:
            await asyncio.gather(producer_task, consumer_task)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            producer_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(producer_task, consumer_task, return_exceptions=True)
            await self.poller.close()
            await self.article_extractor.close()

    async def run_llm_only(self, llm_poll_interval: int = 5):
        """Avvia esclusivamente il worker LLM (Fase 2) sui prompt in coda nel DB."""
        self._running = True
        self.db.initialize_schema()
        worker_task = asyncio.create_task(self.llm_worker(poll_interval=llm_poll_interval))

        try:
            await asyncio.gather(worker_task)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
            await self.llm_client.close()

    async def run_all(self, poll_interval: int = 60, llm_poll_interval: int = 5):
        """Avvia l'intera pipeline end-to-end: Producer + Consumer + LLM Worker in parallelo."""
        self._running = True
        self.initialize()
        producer_task = asyncio.create_task(self.producer(poll_interval))
        consumer_task = asyncio.create_task(self.consumer())
        llm_task = asyncio.create_task(self.llm_worker(poll_interval=llm_poll_interval))

        try:
            await asyncio.gather(producer_task, consumer_task, llm_task)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            producer_task.cancel()
            consumer_task.cancel()
            llm_task.cancel()
            await asyncio.gather(producer_task, consumer_task, llm_task, return_exceptions=True)
            await self.poller.close()
            await self.article_extractor.close()
            await self.llm_client.close()

    def inject_demo_news(self, sample_items: list[dict]):
        """Inietta manualmente notizie demo nella pipeline per validare il flusso completo."""
        self.initialize()
        created_prompts = []

        for idx, item in enumerate(sample_items):
            demo_link = item.get("link") or f"https://demo.feed/article-{idx}-{abs(hash(item['title']))}"
            raw = RawNewsItem(
                source_id=item.get("source_id", "DEMO_FEED"),
                title=item["title"],
                link=demo_link,
                published_at=datetime.now(timezone.utc).isoformat(),
                raw_summary=item.get("summary", ""),
                url_hash=compute_url_hash(item["title"], demo_link),
            )

            filtered = self.process_raw_item(raw)
            if not filtered:
                logger.warning(f"Demo news '{item['title'][:40]}' scartata: nessun ticker trovato.")
                continue

            cursor = self.db.execute(
                """INSERT OR REPLACE INTO news_archive
                   (url_hash, source_id, title, link, published_at, raw_summary,
                    cropped_context, matched_tickers_json, event_category, priority_level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    filtered.url_hash,
                    filtered.source_id,
                    filtered.title,
                    filtered.link,
                    filtered.published_at,
                    filtered.raw_summary or "",
                    filtered.cropped_context,
                    json.dumps(filtered.matched_ticker_symbols),
                    filtered.event_category,
                    filtered.priority_level,
                ),
            )
            news_id = cursor.lastrowid

            prompt_id = self.prompt_builder.process_news_item(filtered, news_id)
            self._prompt_ready_event.set()
            created_prompts.append((prompt_id, filtered))

        return created_prompts


# ═══════════════════════════════════════════════════════════════
# CLI E COMANDI DI GESTIONE
# ═══════════════════════════════════════════════════════════════

def cli_stats(db: DatabaseManager):
    """Stampa statistiche complessive del database inclusa la tabella llm_evaluations."""
    db.initialize_schema()
    print("\n" + "=" * 60)
    print(" STATISTICHE PIPELINE TRADING BOT")
    print("=" * 60)

    w_count = db.fetch_one("SELECT COUNT(*) as c FROM watchlist")["c"]
    kw_count = db.fetch_one("SELECT COUNT(*) as c FROM keywords")["c"]
    ns_count = db.fetch_one("SELECT COUNT(*) as c FROM news_sources")["c"]
    na_count = db.fetch_one("SELECT COUNT(*) as c FROM news_archive")["c"]
    pp_count = db.fetch_one("SELECT COUNT(*) as c FROM prepared_prompts")["c"]
    pp_ready = db.fetch_one(
        "SELECT COUNT(*) as c FROM prepared_prompts WHERE status = 'READY_FOR_LLM'"
    )["c"]
    pp_processed = db.fetch_one(
        "SELECT COUNT(*) as c FROM prepared_prompts WHERE status = 'PROCESSED'"
    )["c"]
    pp_failed = db.fetch_one(
        "SELECT COUNT(*) as c FROM prepared_prompts WHERE status = 'FAILED'"
    )["c"]
    pf_open = db.fetch_one("SELECT COUNT(*) as c FROM watchlist WHERE is_portfolio_open = 1")["c"]
    llm_count = db.fetch_one("SELECT COUNT(*) as c FROM llm_evaluations")["c"]
    llm_ps = db.fetch_one(
        "SELECT COUNT(*) as c FROM llm_evaluations WHERE is_price_sensitive = 1"
    )["c"]

    print(f" - Aziende in Watchlist:          {w_count}")
    print(f" - Posizioni Portafoglio Aperte:  {pf_open}")
    print(f" - Pattern Keywords Compilati:    {kw_count}")
    print(f" - Sorgenti Notizie (RSS/Atom):   {ns_count}")
    print(f" - Notizie Archiviate:            {na_count}")
    print(f" - Prompt Preparati Totali:       {pp_count}")
    print(f"   |- Stato 'READY_FOR_LLM':      {pp_ready}")
    print(f"   |- Stato 'PROCESSED':          {pp_processed}")
    print(f"   +- Stato 'FAILED':             {pp_failed}")
    print(f" - Valutazioni LLM Completate:    {llm_count} (Price-Sensitive: {llm_ps})")
    print("=" * 60 + "\n")


def cli_inspect_prompts(db: DatabaseManager, limit: int = 5):
    """Ispeziona gli ultimi prompt generati pronti per il modulo LLM."""
    db.initialize_schema()
    rows = db.fetch_all(
        f"""SELECT p.id, p.news_id, p.priority_level, p.status, p.estimated_tokens,
                  p.created_at, n.title, n.event_category, n.matched_tickers_json,
                  p.user_prompt
           FROM prepared_prompts p
           JOIN news_archive n ON p.news_id = n.id
           ORDER BY p.id DESC LIMIT {limit}"""
    )

    if not rows:
        print("\nNessun prompt trovato in prepared_prompts.")
        return

    print(f"\n{'='*70}")
    print(f" ISPEZIONE ULTIMI {len(rows)} PROMPT")
    print(f"{'='*70}")

    for r in rows:
        print(
            f"\n--- [PROMPT #{r['id']}] News #{r['news_id']} | "
            f"Priorità {r['priority_level']} | Stato: {r['status']} ---"
        )
        print(f"Titolo:          {r['title']}")
        print(f"Categoria Evento:{r['event_category']}")
        print(f"Ticker Rilevati: {r['matched_tickers_json']}")
        print(f"Stima Token:     ~{r['estimated_tokens']} tokens")
        print(f"Creato il:       {r['created_at']}")
        print("\n--- ANTEPRIMA USER PROMPT (Primi 500 caratteri) ---")
        print(r["user_prompt"][:500] + ("..." if len(r["user_prompt"]) > 500 else ""))
        print("-" * 70)


def cli_inspect_evaluations(db: DatabaseManager, limit: int = 5):
    """Ispeziona le ultime valutazioni strutturate generate dal modello LLM."""
    db.initialize_schema()
    rows = db.fetch_all(
        f"""SELECT e.id, e.news_id, e.prompt_id, e.event_type, e.is_price_sensitive,
                  e.overall_sentiment, e.confidence, e.executive_summary_it,
                  e.impacted_assets_json, e.inference_time_ms, e.created_at,
                  n.title
           FROM llm_evaluations e
           JOIN news_archive n ON e.news_id = n.id
           ORDER BY e.id DESC LIMIT {limit}"""
    )

    if not rows:
        print("\nNessuna valutazione trovata in llm_evaluations.")
        return

    print(f"\n{'='*70}")
    print(f" ISPEZIONE ULTIME {len(rows)} VALUTAZIONI LLM")
    print(f"{'='*70}")

    for r in rows:
        ps_badge = "SI" if r["is_price_sensitive"] else "NO"
        print(
            f"\n--- [EVAL #{r['id']}] Prompt #{r['prompt_id']} (News #{r['news_id']}) | "
            f"Tempo GPU/Mock: {r['inference_time_ms']}ms ---"
        )
        print(f"Titolo Notizia:   {r['title']}")
        print(f"Tipo Evento LLM:  {r['event_type']} | Price-Sensitive: {ps_badge}")
        print(f"Sentiment Globale:{r['overall_sentiment']:+.2f} | Confidenza: {r['confidence']:.2f}")
        print(f"Sintesi (IT):     {r['executive_summary_it']}")
        assets = json.loads(r["impacted_assets_json"])
        print("Asset Impattati:")
        for a in assets:
            print(
                f"  - {a['ticker']} ({a['company_name']}) [{a['role']}] -> "
                f"Action: {a['recommended_action']} | Horizon: {a['expected_horizon']} | "
                f"Sentiment: {a['sentiment_score']:+.2f}\n"
                f"    Rationale: {a['rationale']}"
            )
        print("-" * 70)


def cli_demo_injection(db: DatabaseManager):
    """Inietta notizie demo multi-settore e verifica la generazione immediata dei prompt."""
    orchestrator = NewsPipelineOrchestrator(db=db)

    db.execute(
        "UPDATE watchlist SET is_portfolio_open = 1, portfolio_avg_price = 44.50, "
        "portfolio_allocation_pct = 6.5 WHERE ticker = 'IP.MI'"
    )

    demo_items = [
        {
            "title": "Interpump Group sigla accordo definitivo per acquisizione di azienda tedesca per 210 milioni di euro",
            "summary": "Il Consiglio di Amministrazione di Interpump Group ha approvato all'unanimità l'operazione di acquisizione strategica per rafforzare la divisione idraulica. L'accordo comporterà un incremento immediato dell'EBITDA consolidato. Ricavi attesi a 90 mln.",
            "source_id": "DEMO_MNA",
        },
        {
            "title": "Fincantieri: ricavi trimestrali a 2.2 miliardi (+15%), confermata guidance per l'intero esercizio",
            "summary": "Fincantieri (BIT:FCT) ha chiuso il terzo trimestre con un EBITDA in crescita a 195 milioni di euro e un portafoglio ordini record pari a 37 miliardi. Nessun impatto da rincari materie prime.",
            "source_id": "DEMO_EARNINGS",
        },
        {
            "title": "Deutsche Lufthansa schließt strategische Übernahme ab und erhöht Prognose für das Gesamtjahr",
            "summary": "Lufthansa (LHA.DE) hat die Übernahme erfolgreich abgeschlossen. Der operative Gewinn und Umsatz übertreffen die Markterwartungen deutlich.",
            "source_id": "DEMO_DE",
        },
        {
            "title": "Morgan Stanley upgrades Technoprobe with target price raised to 10.50 euro",
            "summary": "Analysts upgraded Technoprobe (TPRO.MI) citing strong semiconductor probe card recovery and surging demand for AI chips packaging.",
            "source_id": "DEMO_RATING",
        },
    ]

    print(f"\nIniezione di {len(demo_items)} notizie demo...")
    created = orchestrator.inject_demo_news(demo_items)
    print(f"Generati con successo {len(created)} prompt nella tabella prepared_prompts!\n")
    cli_inspect_prompts(db, limit=len(created))


def cli_reset_news(db: DatabaseManager):
    """Pulisce le tabelle storiche delle notizie, prompt e valutazioni LLM e reimposta i metadata dei feed."""
    print("Pulizia tabelle storiche delle notizie (news_archive, prepared_prompts, llm_evaluations)...")
    db.execute("DELETE FROM llm_evaluations")
    db.execute("DELETE FROM prepared_prompts")
    db.execute("DELETE FROM news_archive")
    try:
        db.execute(
            "DELETE FROM sqlite_sequence WHERE name IN ('llm_evaluations', 'prepared_prompts', 'news_archive')"
        )
    except Exception:
        pass
    db.execute(
        "UPDATE news_sources SET last_polled_at = NULL, last_etag = NULL, last_modified = NULL, consecutive_failures = 0"
    )
    print("Reset completato: news_archive, prepared_prompts e llm_evaluations svuotate. Sorgenti RSS reimpostate.\n")
    cli_stats(db)


def main():
    parser = argparse.ArgumentParser(
        description="Small & Mid Cap European News Ingestion, Prompt Generation & Local LLM Pipeline"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Inizializza DB e popola watchlist, keywords e sorgenti",
    )
    parser.add_argument(
        "--reset-news",
        action="store_true",
        help="Svuota news_archive, prepared_prompts, llm_evaluations e resetta lo stato delle sorgenti",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Avvia l'orchestratore asincrono Producer-Consumer (Fase 1)",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Avvia il worker asincrono Local LLM per elaborare i prompt READY_FOR_LLM (Fase 2)",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Avvia in parallelo Producer, Consumer e LLM Worker (Fase 1 + Fase 2)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Intervallo di polling RSS in secondi (default: 60)",
    )
    parser.add_argument(
        "--llm-poll-interval",
        type=int,
        default=5,
        help="Intervallo di polling DB per il worker LLM in secondi (default: 5)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen2.5:7b",
        help="Nome del modello Ollama locale da utilizzare (default: qwen2.5:7b)",
    )
    parser.add_argument(
        "--inspect-prompts",
        type=int,
        nargs="?",
        const=5,
        default=None,
        help="Ispeziona gli ultimi N prompt preparati",
    )
    parser.add_argument(
        "--inspect-evaluations",
        type=int,
        nargs="?",
        const=5,
        default=None,
        help="Ispeziona le ultime N valutazioni LLM salvate in llm_evaluations",
    )
    parser.add_argument(
        "--max-llm-priority",
        type=int,
        choices=[1, 2, 3],
        default=DEFAULT_MAX_LLM_PRIORITY,
        help="Livello massimo di priorità (1, 2 o 3) per generare prompt LLM (default: 2)",
    )
    parser.add_argument(
        "--inject-demo",
        action="store_true",
        help="Inietta notizie demo e genera i prompt",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostra le statistiche correnti del database",
    )

    args = parser.parse_args()
    db = DatabaseManager()

    if args.seed:
        print("Esecuzione seeding database...")
        res = seed_all(db)
        print(
            f"[OK] Seeding completato: {res['companies']} aziende, "
            f"{res['keywords']} keywords, {res['sources']} sorgenti."
        )
        cli_stats(db)
        return

    if args.reset_news:
        cli_reset_news(db)
        return

    if args.stats:
        cli_stats(db)
        return

    if args.inspect_prompts is not None:
        cli_inspect_prompts(db, limit=args.inspect_prompts)
        return

    if args.inspect_evaluations is not None:
        cli_inspect_evaluations(db, limit=args.inspect_evaluations)
        return

    if args.inject_demo:
        cli_demo_injection(db)
        return

    llm_client = LocalLLMClient(model_name=args.llm_model)

    if args.run_all:
        orchestrator = NewsPipelineOrchestrator(
            db=db, llm_client=llm_client, max_llm_priority=args.max_llm_priority
        )
        print(
            f"Avvio Pipeline Completa Producer + Consumer + LLM Worker "
            f"[Modello: {args.llm_model}] (Ctrl+C per terminare)..."
        )
        try:
            asyncio.run(
                orchestrator.run_all(
                    poll_interval=args.poll_interval,
                    llm_poll_interval=args.llm_poll_interval,
                )
            )
        except KeyboardInterrupt:
            print("\nArresto della pipeline completa eseguito.")
        return

    if args.run_llm:
        orchestrator = NewsPipelineOrchestrator(db=db, llm_client=llm_client)
        print(
            f"Avvio LLM Worker [Modello: {args.llm_model}] "
            f"(Polling DB: {args.llm_poll_interval}s | Ctrl+C per terminare)..."
        )
        try:
            asyncio.run(orchestrator.run_llm_only(llm_poll_interval=args.llm_poll_interval))
        except KeyboardInterrupt:
            print("\nArresto del worker LLM completato.")
        return

    if args.run:
        orchestrator = NewsPipelineOrchestrator(
            db=db, llm_client=llm_client, max_llm_priority=args.max_llm_priority
        )
        print("Avvio Orchestratore Pipeline Fase 1 (Ctrl+C per terminare)...")
        try:
            asyncio.run(orchestrator.run(poll_interval=args.poll_interval))
        except KeyboardInterrupt:
            print("\nArresto della pipeline completato.")
        return

    parser.print_help()
    cli_stats(db)


if __name__ == "__main__":
    main()