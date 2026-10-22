-- SQLite Pragmas (executed per-connection in db_manager.py)
-- PRAGMA journal_mode = WAL;
-- PRAGMA synchronous = NORMAL;
-- PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    base_symbol TEXT NOT NULL,
    company_name TEXT NOT NULL,
    clean_name TEXT NOT NULL,
    sector TEXT NOT NULL,
    index_country TEXT NOT NULL,
    is_portfolio_open INTEGER DEFAULT 0,
    portfolio_avg_price REAL DEFAULT NULL,
    portfolio_allocation_pct REAL DEFAULT 0.0,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL REFERENCES watchlist(ticker) ON DELETE CASCADE,
    pattern TEXT NOT NULL,
    match_type TEXT NOT NULL,
    is_case_sensitive INTEGER DEFAULT 0,
    requires_financial_context INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS news_sources (
    source_id TEXT PRIMARY KEY,
    feed_type TEXT NOT NULL,
    language TEXT NOT NULL,
    poll_interval_sec INTEGER NOT NULL,
    url TEXT NOT NULL,
    last_polled_at TIMESTAMP,
    last_etag TEXT,
    last_modified TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    is_enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS news_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash TEXT UNIQUE NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    link TEXT NOT NULL,
    published_at TEXT,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_summary TEXT,
    cropped_context TEXT NOT NULL,
    matched_tickers_json TEXT NOT NULL,
    event_category TEXT NOT NULL,
    priority_level INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS prepared_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id INTEGER UNIQUE NOT NULL REFERENCES news_archive(id) ON DELETE CASCADE,
    priority_level INTEGER NOT NULL,
    system_prompt TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    expected_json_schema TEXT NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    status TEXT DEFAULT 'READY_FOR_LLM',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id INTEGER NOT NULL REFERENCES news_archive(id) ON DELETE CASCADE,
    prompt_id INTEGER UNIQUE NOT NULL REFERENCES prepared_prompts(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    is_price_sensitive INTEGER NOT NULL DEFAULT 0,
    overall_sentiment REAL NOT NULL,
    confidence REAL NOT NULL,
    executive_summary_it TEXT NOT NULL,
    impacted_assets_json TEXT NOT NULL,
    inference_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- B-Tree Indices for performance
CREATE INDEX IF NOT EXISTS idx_keywords_ticker ON keywords(ticker);
CREATE INDEX IF NOT EXISTS idx_keywords_match_type ON keywords(match_type);
CREATE INDEX IF NOT EXISTS idx_news_archive_url_hash ON news_archive(url_hash);
CREATE INDEX IF NOT EXISTS idx_news_archive_source_id ON news_archive(source_id);
CREATE INDEX IF NOT EXISTS idx_news_archive_priority ON news_archive(priority_level);
CREATE INDEX IF NOT EXISTS idx_news_archive_ingested ON news_archive(ingested_at);
CREATE INDEX IF NOT EXISTS idx_prepared_prompts_status ON prepared_prompts(status);
CREATE INDEX IF NOT EXISTS idx_prepared_prompts_priority ON prepared_prompts(priority_level);
CREATE INDEX IF NOT EXISTS idx_watchlist_portfolio ON watchlist(is_portfolio_open);
CREATE INDEX IF NOT EXISTS idx_watchlist_active ON watchlist(is_active);
CREATE INDEX IF NOT EXISTS idx_llm_evaluations_news_id ON llm_evaluations(news_id);
CREATE INDEX IF NOT EXISTS idx_llm_evaluations_prompt_id ON llm_evaluations(prompt_id);
CREATE INDEX IF NOT EXISTS idx_llm_evaluations_event_type ON llm_evaluations(event_type);
CREATE INDEX IF NOT EXISTS idx_llm_evaluations_price_sensitive ON llm_evaluations(is_price_sensitive);