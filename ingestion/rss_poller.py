"""Asynchronous RSS Poller with persistent HTTP sessions, anti-ban jitter, and freshness filtering.

Features:
- Persistent HTTP session pooling (curl_cffi Chrome 120 TLS impersonation + aiohttp fallback)
- ETag/Last-Modified conditional fetching
- User-Agent rotation
- Per-source poll_interval_sec enforcement with random jitter (±15%)
- Bounded concurrency via asyncio.Semaphore to prevent Google News rate-limiting
- Full-content RSS tag extraction (<content:encoded> fallback to <summary>)
- Non-blocking XML feed parsing via asyncio.to_thread
- Stale news TTL filter (discards articles older than MAX_NEWS_AGE_HOURS)
- Pre-warming of LRUDeduplicator from SQLite news_archive at startup
- Google News title suffix normalization before SHA-256 hashing
"""

import asyncio
import hashlib
import logging
import random
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import feedparser

from config.settings import (
    ACCEPT_HEADER,
    ACCEPT_LANGUAGE,
    AUTO_CIRCUIT_BREAKER_THRESHOLD,
    BACKOFF_MULTIPLIER,
    GNEWS_BATCH_SIZE,
    GNEWS_TIME_WINDOW,
    HTTP_TIMEOUT_SEC,
    LRU_DEDUP_SIZE,
    MAX_CONCURRENT_RSS_REQUESTS,
    MAX_CONSECUTIVE_FAILURES,
    MAX_NEWS_AGE_HOURS,
    POLL_JITTER_PCT,
    USER_AGENTS,
)
from database.db_manager import DatabaseManager
from models.schemas import RawNewsItem

logger = logging.getLogger(__name__)

# Try curl_cffi for TLS fingerprint impersonation, fallback to aiohttp
try:
    from curl_cffi.requests import AsyncSession as CurlAsyncSession
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

# Strip trailing source attribution on Google News titles (e.g. "Title - MilanoFinanza")
_GNEWS_SOURCE_SUFFIX_RE = re.compile(r'\s+[-–—|]\s+[^-–—|]{2,40}$')


class LRUDeduplicator:
    """LRU set for O(1) deduplication of the last N url_hashes."""

    def __init__(self, maxsize: int = LRU_DEDUP_SIZE):
        self._cache: OrderedDict[str, None] = OrderedDict()
        self._maxsize = maxsize

    def contains(self, key: str) -> bool:
        if key in self._cache:
            self._cache.move_to_end(key)
            return True
        return False

    def add(self, key: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = None
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


def compute_url_hash(title: str = "", link: str = "") -> str:
    """Compute SHA-256 hash for deduplication.

    For Google News (long variable token URLs), normalizes the title and strips
    trailing publisher attribution (' - Reuters', ' - MilanoFinanza') before hashing.
    For other sources, hashes the link.
    """
    if 'news.google.com' in link:
        cleaned_title = _GNEWS_SOURCE_SUFFIX_RE.sub('', title.strip())
        normalized = re.sub(r'\s+', ' ', cleaned_title.lower()).strip()
    else:
        normalized = link.strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def parse_rss_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Parse RFC-2822 or ISO-8601 date strings from RSS feeds into a UTC datetime object."""
    if not date_str or not date_str.strip():
        return None
    raw = date_str.strip()

    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    try:
        normalized_iso = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_richest_entry_body(entry: object) -> str:
    """Extract the longest available body from an RSS/Atom entry.

    Checks <content:encoded> (entry.content) in addition to <summary> and <description>
    to mitigate short editorial RSS teaser stubs.
    """
    candidates: List[str] = []

    content_list = getattr(entry, 'content', None)
    if isinstance(content_list, list):
        for item in content_list:
            val = item.get('value', '') if isinstance(item, dict) else getattr(item, 'value', '')
            if val and isinstance(val, str):
                candidates.append(val.strip())

    for attr in ('summary', 'description'):
        val = getattr(entry, attr, '')
        if val and isinstance(val, str):
            candidates.append(val.strip())

    if not candidates:
        return ""
    return max(candidates, key=len)


class RSSPoller:
    """Asynchronous RSS feed poller with persistent connection pooling and freshness filtering."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        max_concurrent_requests: int = MAX_CONCURRENT_RSS_REQUESTS,
        max_news_age_hours: float = MAX_NEWS_AGE_HOURS,
    ):
        self.db = db or DatabaseManager()
        self.dedup = LRUDeduplicator()
        self._ua_index = 0
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.max_news_age_hours = max_news_age_hours
        self._curl_session: Optional['CurlAsyncSession'] = None
        self._aiohttp_session: Optional['aiohttp.ClientSession'] = None

    def warm_dedup_cache(self) -> int:
        """Pre-populate the in-memory LRU deduplicator from SQLite news_archive."""
        try:
            rows = self.db.fetch_all(
                f"SELECT url_hash FROM news_archive ORDER BY id ASC LIMIT {LRU_DEDUP_SIZE}"
            )
            for r in rows:
                self.dedup.add(r["url_hash"])
            return len(rows)
        except Exception as exc:
            logger.debug(f"Could not warm dedup cache: {exc}")
            return 0

    async def close(self) -> None:
        """Close persistent HTTP sessions cleanly."""
        if self._curl_session is not None:
            try:
                await self._curl_session.close()
            except Exception:
                pass
            self._curl_session = None

        if self._aiohttp_session is not None:
            try:
                await self._aiohttp_session.close()
            except Exception:
                pass
            self._aiohttp_session = None

    def _get_next_ua(self) -> str:
        """Rotate through User-Agent strings."""
        ua = USER_AGENTS[self._ua_index % len(USER_AGENTS)]
        self._ua_index += 1
        return ua

    def _build_headers(
        self, etag: Optional[str] = None, last_modified: Optional[str] = None
    ) -> Dict[str, str]:
        """Build HTTP headers with anti-ban measures and conditional fetching."""
        headers = {
            'User-Agent': self._get_next_ua(),
            'Accept': ACCEPT_HEADER,
            'Accept-Language': ACCEPT_LANGUAGE,
            'Cache-Control': 'no-cache',
        }
        if etag:
            headers['If-None-Match'] = etag
        if last_modified:
            headers['If-Modified-Since'] = last_modified
        return headers

    async def _get_curl_session(self) -> Optional['CurlAsyncSession']:
        """Get or lazily initialize the persistent curl_cffi AsyncSession."""
        if not _HAS_CURL_CFFI:
            return None
        if self._curl_session is None:
            self._curl_session = CurlAsyncSession(impersonate="chrome120", max_clients=16)
        return self._curl_session

    async def _get_aiohttp_session(self, timeout: int) -> Optional['aiohttp.ClientSession']:
        """Get or lazily initialize the persistent aiohttp ClientSession."""
        if not _HAS_AIOHTTP:
            return None
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            connector = aiohttp.TCPConnector(limit=16, ttl_dns_cache=300)
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=client_timeout, connector=connector
            )
        return self._aiohttp_session

    async def _fetch_feed_curl(
        self, url: str, headers: Dict[str, str], timeout: int = HTTP_TIMEOUT_SEC
    ) -> Optional[Tuple[int, str, Dict[str, str]]]:
        """Fetch feed using persistent curl_cffi session with Chrome TLS impersonation."""
        session = await self._get_curl_session()
        if session is None:
            return None
        try:
            response = await session.get(
                url, headers=headers, timeout=timeout, allow_redirects=True
            )
            resp_headers = dict(response.headers) if response.headers else {}
            return response.status_code, response.text, resp_headers
        except Exception as e:
            logger.debug(f"curl_cffi failed for {url}: {e}")
            return None

    async def _fetch_feed_aiohttp(
        self, url: str, headers: Dict[str, str], timeout: int = HTTP_TIMEOUT_SEC
    ) -> Optional[Tuple[int, str, Dict[str, str]]]:
        """Fetch feed using persistent aiohttp session as fallback."""
        session = await self._get_aiohttp_session(timeout)
        if session is None:
            return None
        try:
            async with session.get(url, headers=headers) as response:
                text = await response.text()
                resp_headers = dict(response.headers)
                return response.status, text, resp_headers
        except Exception as e:
            logger.debug(f"aiohttp failed for {url}: {e}")
            return None

    async def fetch_feed(
        self, url: str, etag: Optional[str] = None, last_modified: Optional[str] = None
    ) -> Optional[Tuple[int, str, Dict[str, str]]]:
        """Fetch a feed URL with TLS impersonation (curl_cffi) or aiohttp fallback."""
        headers = self._build_headers(etag, last_modified)

        async with self._semaphore:
            if "news.google.com" in url:
                await asyncio.sleep(random.uniform(0.15, 0.65))

            result = await self._fetch_feed_curl(url, headers)
            if result is not None:
                return result

            return await self._fetch_feed_aiohttp(url, headers)

    def _is_entry_stale(self, published_str: str) -> bool:
        """Check if an entry's publication timestamp is older than max_news_age_hours."""
        if not published_str or self.max_news_age_hours <= 0:
            return False

        pub_dt = parse_rss_datetime(published_str)
        if pub_dt is None:
            return False

        now_utc = datetime.now(timezone.utc)
        age = now_utc - pub_dt
        return age > timedelta(hours=self.max_news_age_hours)

    def _parse_feed_entries(
        self, xml_text: str, source_id: str, enforce_freshness: bool = True
    ) -> List[RawNewsItem]:
        """Parse feed XML into RawNewsItem objects, discarding stale items."""
        items: List[RawNewsItem] = []
        feed = feedparser.parse(xml_text)
        stale_skipped = 0

        for entry in feed.entries:
            title = getattr(entry, 'title', '') or ''
            link = getattr(entry, 'link', '') or ''
            if not title and not link:
                continue

            published = getattr(entry, 'published', '') or getattr(entry, 'updated', '') or ''

            if enforce_freshness and self._is_entry_stale(published):
                stale_skipped += 1
                continue

            summary = _extract_richest_entry_body(entry)
            url_hash = compute_url_hash(title, link)

            if self.dedup.contains(url_hash):
                continue
            self.dedup.add(url_hash)

            items.append(RawNewsItem(
                source_id=source_id,
                title=title.strip(),
                link=link.strip(),
                published_at=published,
                raw_summary=summary,
                url_hash=url_hash,
            ))

        if stale_skipped > 0:
            logger.debug(
                f"[{source_id}] Scartati {stale_skipped} articoli obsoleti (> {self.max_news_age_hours}h)."
            )

        return items

    def _is_source_due_for_poll(
        self, last_polled_at: Optional[str], poll_interval_sec: int
    ) -> bool:
        """Determine if enough time (with ±15% jitter) has elapsed since last_polled_at."""
        if not last_polled_at:
            return True

        last_dt = parse_rss_datetime(last_polled_at)
        if last_dt is None:
            return True

        jitter_factor = random.uniform(1.0 - POLL_JITTER_PCT, 1.0 + POLL_JITTER_PCT)
        effective_interval = max(10.0, poll_interval_sec * jitter_factor)
        elapsed_sec = (datetime.now(timezone.utc) - last_dt).total_seconds()
        return elapsed_sec >= effective_interval

    async def poll_source(
        self, source_id: str, url: str, feed_type: str,
        etag: Optional[str] = None, last_modified: Optional[str] = None
    ) -> List[RawNewsItem]:
        """Poll a single RSS source and return new items."""
        result = await self.fetch_feed(url, etag, last_modified)

        if result is None:
            self._record_failure(source_id)
            return []

        status_code, body, resp_headers = result
        now_str = datetime.now(timezone.utc).isoformat()

        if status_code == 304:
            self.db.execute(
                "UPDATE news_sources SET last_polled_at = ? WHERE source_id = ?",
                (now_str, source_id)
            )
            return []

        if status_code >= 400:
            logger.warning(f"Unexpected status {status_code} from {source_id}")
            self._record_failure(source_id, status_code=status_code)
            return []

        if status_code != 200:
            logger.warning(f"Unexpected status {status_code} from {source_id}")
            return []

        new_etag = resp_headers.get('ETag', resp_headers.get('etag', ''))
        new_lm = resp_headers.get('Last-Modified', resp_headers.get('last-modified', ''))

        self.db.execute(
            """UPDATE news_sources 
               SET last_polled_at = ?, last_etag = ?, last_modified = ?, consecutive_failures = 0
               WHERE source_id = ?""",
            (now_str, new_etag or None, new_lm or None, source_id)
        )

        # Offload CPU-bound XML feedparser to thread pool to keep asyncio loop responsive
        return await asyncio.to_thread(self._parse_feed_entries, body, source_id, True)

    def _record_failure(self, source_id: str, status_code: Optional[int] = None) -> None:
        """Increment failure counter and apply backoff or circuit-breaker if threshold reached."""
        row = self.db.fetch_one(
            "SELECT consecutive_failures, poll_interval_sec FROM news_sources WHERE source_id = ?",
            (source_id,)
        )
        if row is None:
            return

        failures = row['consecutive_failures'] + 1
        interval = row['poll_interval_sec']

        if failures >= AUTO_CIRCUIT_BREAKER_THRESHOLD:
            interval = min(interval * BACKOFF_MULTIPLIER, 3600)
            logger.error(
                f"[CIRCUIT BREAKER] Sorgente {source_id} ha raggiunto {failures} fallimenti "
                f"consecutivi (ultimo HTTP status: {status_code}). Polling backoff a {interval}s."
            )
        elif failures >= MAX_CONSECUTIVE_FAILURES or status_code in (404, 410):
            interval = min(interval * BACKOFF_MULTIPLIER, 3600)

        self.db.execute(
            """UPDATE news_sources 
               SET consecutive_failures = ?, poll_interval_sec = ?, 
                   last_polled_at = ?
               WHERE source_id = ?""",
            (failures, interval, datetime.now(timezone.utc).isoformat(), source_id)
        )

    def generate_portfolio_priority_feeds(self) -> List[Dict[str, str]]:
        """Generate dynamic Google News RSS URLs for portfolio positions."""
        rows = self.db.fetch_all(
            "SELECT ticker, company_name, clean_name FROM watchlist WHERE is_portfolio_open = 1 AND is_active = 1"
        )

        if not rows:
            return []

        feeds = []
        batch = []
        batch_idx = 0

        for row in rows:
            clean = row['clean_name']
            batch.append(f'"{clean}"')

            if len(batch) >= GNEWS_BATCH_SIZE:
                feed = self._build_gnews_url(batch, batch_idx)
                feeds.append(feed)
                batch = []
                batch_idx += 1

        if batch:
            feed = self._build_gnews_url(batch, batch_idx)
            feeds.append(feed)

        return feeds

    def _build_gnews_url(self, company_names: List[str], batch_idx: int) -> Dict[str, str]:
        """Build a Google News RSS search URL for a batch of company names."""
        query = '+OR+'.join(quote_plus(name) for name in company_names)
        url = (
            f"https://news.google.com/rss/search?q=({query})"
            f"+when:{GNEWS_TIME_WINDOW}&hl=it&gl=IT&ceid=IT:it"
        )
        return {
            'source_id': f'DYNAMIC_PORTFOLIO_{batch_idx}',
            'url': url,
            'feed_type': 'DYNAMIC_PORTFOLIO',
            'language': 'it',
            'poll_interval_sec': 60,
        }

    async def poll_all_active(self) -> List[RawNewsItem]:
        """Poll active sources that are due for polling concurrently with bounded semaphore."""
        sources = self.db.fetch_all(
            "SELECT * FROM news_sources WHERE is_enabled = 1"
        )

        dynamic_feeds = self.generate_portfolio_priority_feeds()
        all_items: List[RawNewsItem] = []
        tasks = []

        for source in sources:
            if not self._is_source_due_for_poll(
                source['last_polled_at'], source['poll_interval_sec']
            ):
                continue

            tasks.append(
                self.poll_source(
                    source_id=source['source_id'],
                    url=source['url'],
                    feed_type=source['feed_type'],
                    etag=source['last_etag'],
                    last_modified=source['last_modified'],
                )
            )

        for df in dynamic_feeds:
            tasks.append(
                self.poll_source(
                    source_id=df['source_id'],
                    url=df['url'],
                    feed_type=df['feed_type'],
                )
            )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Poll task failed: {result}")
                continue
            if isinstance(result, list):
                all_items.extend(result)

        return all_items