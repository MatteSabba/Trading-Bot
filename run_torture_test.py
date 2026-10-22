"""Runner per la Fase 1 (Torture Test) della pipeline Trading Bot v1.2.

Esegue il dataset torture_test_news.json attraverso:
1. Text Cleaning & Context Cropping (Selectolax + Head/Match/Tail)
2. RegexEngine a 2 stadi (FlashText O(N) + Lookup O(1) + Anti-Noise B2C/Sport)
3. TaxonomyClassifier (Weighted Multi-Score + Portfolio Override)
4. Cross-Source Jaccard Deduplication
5. PromptBuilder (CET Market Phase + Microstructure Guardrails)
6. LocalLLMClient (Ollama GPU oppure Fallback Euristico se --heuristic-only)
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DatabaseManager
from database.seeder import seed_all
from ingestion.rss_poller import compute_url_hash
from llm.local_client import LocalLLMClient
from main import NewsPipelineOrchestrator
from models.schemas import RawNewsItem


async def run_benchmark(json_path: Path, model_name: str, heuristic_only: bool, reset_history: bool):
    db = DatabaseManager()
    db.initialize_schema()

    # Assicura che watchlist, keywords e news_sources siano popolati
    w_count = db.fetch_one("SELECT COUNT(*) as c FROM watchlist")["c"]
    if w_count == 0:
        print("Database vuoto: esecuzione seeding iniziale...")
        seed_all(db)

    if reset_history:
        print("Pulizia tabelle storiche (news_archive, prepared_prompts, llm_evaluations)...")
        db.execute("DELETE FROM llm_evaluations")
        db.execute("DELETE FROM prepared_prompts")
        db.execute("DELETE FROM news_archive")

    # Imposta IP.MI come posizione aperta in portafoglio per testare T15_PORTFOLIO_OVERRIDE
    db.execute("UPDATE watchlist SET is_portfolio_open = 0")
    db.execute(
        "UPDATE watchlist SET is_portfolio_open = 1, portfolio_avg_price = 44.50, "
        "portfolio_allocation_pct = 6.5 WHERE ticker = 'IP.MI'"
    )

    # Se heuristic_only è attivo, punta a una porta chiusa per forzare il fallback deterministico
    endpoint = "http://127.0.0.1:9/api/chat" if heuristic_only else "http://localhost:11434/api/chat"
    llm_client = LocalLLMClient(endpoint_url=endpoint, model_name=model_name)
    orchestrator = NewsPipelineOrchestrator(db=db, llm_client=llm_client)
    orchestrator.initialize()

    with open(json_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print("\n" + "=" * 105)
    print(f" [BENCHMARK] AVVIO TORTURE TEST FASE 1 ({len(test_cases)} Casi di Test) | Modalità LLM: {'FALLBACK EURISTICO' if heuristic_only else model_name}")
    print("=" * 105)

    passed = 0
    failed = 0
    now_utc = datetime.now(timezone.utc)

    for tc in test_cases:
        test_id = tc["test_id"]
        age_min = tc.get("age_minutes", 5)
        pub_dt = (now_utc - timedelta(minutes=age_min)).isoformat()
        link = f"https://torture.test/articles/{test_id}"

        raw = RawNewsItem(
            source_id=tc["source_id"],
            title=tc["title"],
            link=link,
            published_at=pub_dt,
            raw_summary=tc["summary"],
            url_hash=compute_url_hash(tc["title"], link),
        )

        errors = []
        filtered = orchestrator.process_raw_item(raw)
        matched_Flag = filtered is not None

        # 1. Verifica Match vs Scarto
        if matched_Flag != tc["should_match"]:
            errors.append(f"Match atteso={tc['should_match']}, ottenuto={matched_Flag}")
        elif matched_Flag and filtered is not None:
            # 2. Verifica Ticker individuati
            expected_tickers = sorted(tc.get("expected_tickers", []))
            if filtered.matched_ticker_symbols != expected_tickers:
                errors.append(
                    f"Ticker attesi={expected_tickers}, ottenuti={filtered.matched_ticker_symbols}"
                )

            # 3. Verifica Categoria Tassonomica
            if "expected_category" in tc and filtered.event_category != tc["expected_category"]:
                errors.append(
                    f"Categoria attesa={tc['expected_category']}, ottenuta={filtered.event_category}"
                )

            # 4. Verifica Priorità
            if "expected_priority" in tc and filtered.priority_level != tc["expected_priority"]:
                errors.append(
                    f"Priorità attesa={tc['expected_priority']}, ottenuta={filtered.priority_level}"
                )

            # Salvataggio in news_archive
            cursor = db.execute(
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

            # 5. Verifica Cross-Source Syndication Dedup (Jaccard)
            is_dup = orchestrator._is_cross_source_duplicate(filtered, news_id)
            expected_dup = tc.get("expected_dedup_skip", False)
            if is_dup != expected_dup:
                errors.append(f"Syndication Dedup atteso={expected_dup}, ottenuto={is_dup}")

            if not is_dup:
                prompt_id = orchestrator.prompt_builder.process_news_item(filtered, news_id)
                p_row = db.fetch_one(
                    """SELECT id, news_id, priority_level, system_prompt,
                              user_prompt, expected_json_schema
                       FROM prepared_prompts WHERE id = ?""",
                    (prompt_id,),
                )

                # 6. Verifica presenza Guardrail nel User Prompt
                if "expected_guardrail" in tc:
                    guardrail_str = tc["expected_guardrail"]
                    if guardrail_str not in p_row["user_prompt"]:
                        errors.append(f"Guardrail mancante nel prompt: [{guardrail_str}]")

                # 7. Esecuzione valutazione LLM (o Fallback) e verifica output
                await orchestrator._evaluate_single_prompt_row(p_row)
                e_row = db.fetch_one(
                    "SELECT * FROM llm_evaluations WHERE prompt_id = ?", (prompt_id,)
                )

                if not e_row:
                    errors.append("Nessun record salvato in llm_evaluations")
                else:
                    assets = json.loads(e_row["impacted_assets_json"])
                    first_asset = assets[0] if assets else {}

                    if "expected_llm_event" in tc and e_row["event_type"] != tc["expected_llm_event"]:
                        errors.append(
                            f"LLM event_type atteso={tc['expected_llm_event']}, ottenuto={e_row['event_type']}"
                        )
                    if "expected_llm_action" in tc and first_asset.get("recommended_action") != tc["expected_llm_action"]:
                        errors.append(
                            f"LLM action attesa={tc['expected_llm_action']}, ottenuta={first_asset.get('recommended_action')}"
                        )
                    if "forbidden_llm_action" in tc and first_asset.get("recommended_action") == tc["forbidden_llm_action"]:
                        errors.append(
                            f"LLM ha violato il guardrail raccomandando {tc['forbidden_llm_action']}"
                        )
                    if "expected_llm_horizon" in tc and first_asset.get("expected_horizon") != tc["expected_llm_horizon"]:
                        errors.append(
                            f"LLM horizon atteso={tc['expected_llm_horizon']}, ottenuto={first_asset.get('expected_horizon')}"
                        )
                    if "expected_llm_role" in tc and first_asset.get("role") != tc["expected_llm_role"]:
                        errors.append(
                            f"LLM role atteso={tc['expected_llm_role']}, ottenuto={first_asset.get('role')}"
                        )
                    if "expected_sentiment_sign" in tc:
                        sent = e_row["overall_sentiment"]
                        if tc["expected_sentiment_sign"] == "positive" and sent <= 0:
                            errors.append(f"Sentiment atteso positivo (>0), ottenuto={sent:+.2f}")
                        elif tc["expected_sentiment_sign"] == "negative" and sent >= 0:
                            errors.append(f"Sentiment atteso negativo (<0), ottenuto={sent:+.2f}")

        if not errors:
            passed += 1
            tickers_str = ",".join(filtered.matched_ticker_symbols) if filtered else "SCARTATA (OK)"
            cat_str = f"{filtered.event_category} (P{filtered.priority_level})" if filtered else "-"
            print(f"[PASS] | {test_id:<26} | Ticker: {tickers_str:<16} | Cat: {cat_str}")
        else:
            failed += 1
            print(f"[FAIL] | {test_id:<26} | Motivo: {' ; '.join(errors)}")
            print(f"        Nota: {tc['notes']}")

    await llm_client.close()
    print("=" * 105)
    print(f" [RISULTATO FINALE]: {passed}/{len(test_cases)} PASSATI | {failed} FALLITI")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue il Torture Test di Fase 1")
    parser.add_argument(
        "--file", type=str, default="torture_test_news.json", help="Percorso del file JSON di test"
    )
    parser.add_argument(
        "--llm-model", type=str, default="qwen2.5:7b", help="Modello Ollama da utilizzare"
    )
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Forza l'uso del fallback euristico deterministico senza chiamare Ollama",
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="Non svuotare news_archive/prepared_prompts prima del test",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            json_path=Path(args.file),
            model_name=args.llm_model,
            heuristic_only=args.heuristic_only,
            reset_history=not args.keep_history,
        )
    )