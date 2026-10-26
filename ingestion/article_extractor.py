"""Asynchronous Article Full-Text Extractor for RSS feeds with thin descriptions (v1.4).

Features:
- Uses selectolax.lexbor.LexborHTMLParser (compatible with selectolax >= 0.3 / 1.0)
  with fallback to legacy selectolax.parser and pure-regex <p>/<article> extraction.
- Triggers when clean_summary < FULL_TEXT_MIN_SUMMARY_CHARS (1200 chars) or
  when title/summary token similarity >= 0.65.
- Native asynchronous Google News URL resolution via SOCS GDPR consent bypass cookie
  and Google 'batchexecute' (Fbv4je) RPC decoding inside curl_cffi Chrome 120.
"""

import asyncio
import base64
import html
import json
import logging
import re
from typing import Optional
from urllib.parse import quote, urlparse

from config.settings import FULL_TEXT_MIN_SUMMARY_CHARS

logger = logging.getLogger(__name__)

# Try curl_cffi for TLS fingerprint impersonation
try:
    from curl_cffi.requests import AsyncSession as CurlAsyncSession
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

# Try selectolax.lexbor first (modern backend), fallback to legacy selectolax.parser
try:
    from selectolax.lexbor import LexborHTMLParser
    _HAS_SELECTOLAX = True
except ImportError:
    try:
        from selectolax.parser import HTMLParser as LexborHTMLParser
        _HAS_SELECTOLAX = True
    except ImportError:
        _HAS_SELECTOLAX = False

# Minimum content length to consider extraction successful
_MIN_EXTRACTED_CHARS = 200

# Persistent Google SOCS cookie to bypass consent.google.com GDPR wall in EU
_GOOGLE_SOCS_COOKIE = "CAESHAgBEhJnd3NfMjAyNDA3MTUtMF9SQzIaAmVuIAEaBgiA_LmwBg"

# Noise HTML elements to strip before text extraction
_NOISE_TAGS = frozenset({
    'script', 'style', 'nav', 'header', 'footer', 'aside', 'form',
    'noscript', 'svg', 'iframe', 'button', 'input', 'select', 'textarea',
})

# Cookie/consent banner class patterns to remove
_COOKIE_BANNER_RE = re.compile(
    r'(?:cookie|consent|gdpr|privacy.?banner|onetrust|cookiebot|paywall|newsletter)',
    re.IGNORECASE,
)

# Inline cookie consent sentences or editorial correction footers embedded inside normal <p> tags
_INLINE_JUNK_PARAGRAPH_RE = re.compile(
    r'(?:Ich\s+stimme\s+der\s+Verwendung\s+von\s+Cookies\s+zu|'
    r'Bitte\s+lesen\s+und\s+akzeptieren\s+Sie\s+die\s+Datenschutzinformation|'
    r'Avertissement\s+l[ée]gal\s+Contactez-nous\s+pour\s+toute\s+demande\s+de\s+correction|'
    r'Kontaktieren\s+Sie\s+uns,\s+wenn\s+Sie\s+eine\s+Korrektur\s+wünschen|'
    r'Borsa\s+Italiana\s+non\s+ha\s+responsabilit[àa]\s+per\s+il\s+contenuto|'
    r'Sarai\s+automaticamente\s+diretto\s+al\s+link\s+in\s+cinque\s+secondi)',
    re.IGNORECASE,
)
_INTERSTITIAL_DISCLAIMER_RE = _INLINE_JUNK_PARAGRAPH_RE

# Semantic article body selectors in priority order
_ARTICLE_SELECTORS = [
    'article',
    '[itemprop="articleBody"]',
    '.news-text',
    '.article-text',
    '.news-body',
    '#tlb-news-content',
    '.detail-text',
    '.corpo-articolo',
    '[role="main"]',
    'main',
    '.article-body',
    '.article-content',
    '.post-content',
    '.entry-content',
    '.story-body',
    '#article-body',
    '.main-content',
]

_WORD_TOKEN_RE = re.compile(r'\b[a-zA-Z0-9À-ÿ]{2,}\b')

# Regex patterns for Google News RPC token extraction (data-n-a-sg & data-n-a-ts)
_GNEWS_SG_RE = re.compile(r'data-n-a-sg="([^"]+)"')
_GNEWS_TS_RE = re.compile(r'data-n-a-ts="([^"]+)"')
_GNEWS_RPC_URL_RE = re.compile(r'\["garturlres","(https?://[^"]+)"')

# Pure-Python regex fallbacks if selectolax is unavailable or fails
_SCRIPT_STYLE_STRIP_RE = re.compile(
    r'<(script|style|nav|header|footer|aside|noscript|svg|iframe)[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)
_PARAGRAPH_TAG_RE = re.compile(r'<p\b[^>]*>(.*?)</p>', re.DOTALL | re.IGNORECASE)
_HTML_TAG_STRIP_RE = re.compile(r'<[^>]+>')


def _compute_token_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard token similarity between two texts in [0.0, 1.0]."""
    tokens_a = set(_WORD_TOKEN_RE.findall(text_a.lower()))
    tokens_b = set(_WORD_TOKEN_RE.findall(text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def needs_full_text_extraction(
    title: str,
    clean_summary: str,
    min_chars: int = FULL_TEXT_MIN_SUMMARY_CHARS,
) -> bool:
    """Determine if full-text extraction from the web page is needed.

    Returns True when:
    - clean_summary is shorter than min_chars (default 1200 chars), OR
    - Token similarity between title and clean_summary >= 0.65 (title repetition)
    """
    if len(clean_summary) < min_chars:
        return True
    similarity = _compute_token_similarity(title, clean_summary)
    return similarity >= 0.65


def _regex_extract_paragraphs(html_content: str) -> str:
    """Fallback HTML paragraph extractor using pure regular expressions."""
    if not html_content:
        return ""
    cleaned_html = _SCRIPT_STYLE_STRIP_RE.sub(' ', html_content)
    paragraphs = _PARAGRAPH_TAG_RE.findall(cleaned_html)
    texts = []
    for raw_p in paragraphs:
        txt = html.unescape(_HTML_TAG_STRIP_RE.sub(' ', raw_p))
        txt = re.sub(r'\s+', ' ', txt).strip()
        if len(txt) > 35 and not _COOKIE_BANNER_RE.search(txt[:80]):
            texts.append(txt)
    result = ' '.join(texts)
    return result if len(result) >= _MIN_EXTRACTED_CHARS else ""


def _extract_text_from_html(html_content: str) -> str:
    """Parse HTML and extract article body text using LexborHTMLParser or regex fallback."""
    if not html_content:
        return ""

    if _HAS_SELECTOLAX:
        try:
            tree = LexborHTMLParser(html_content)

            for tag in _NOISE_TAGS:
                for node in tree.css(tag):
                    node.decompose()

            for node in tree.css('[class], [id]'):
                attrs = ''
                for attr_name in ('class', 'id'):
                    val = node.attributes.get(attr_name, '')
                    if val:
                        attrs += ' ' + val
                if attrs and _COOKIE_BANNER_RE.search(attrs):
                    node.decompose()

            for selector in _ARTICLE_SELECTORS:
                try:
                    nodes = tree.css(selector)
                    if nodes:
                        container = nodes[0]
                        paragraphs = container.css('p, li')
                        if paragraphs:
                            texts = []
                            for p in paragraphs:
                                p_text = p.text(separator=' ').strip()
                                min_len = 20 if p.tag == 'li' else 30
                                if len(p_text) > min_len and not _INTERSTITIAL_DISCLAIMER_RE.search(p_text):
                                    texts.append(p_text)
                            if texts:
                                result = re.sub(r'\s+', ' ', ' '.join(texts)).strip()
                                if len(result) >= _MIN_EXTRACTED_CHARS:
                                    return result
                        # Fallback for publishers (like Teleborsa) that place article text directly inside container divs/spans
                        direct_text = re.sub(r'\s+', ' ', container.text(separator=' ').strip())
                        if (
                            len(direct_text) >= _MIN_EXTRACTED_CHARS
                            and not _INTERSTITIAL_DISCLAIMER_RE.search(direct_text)
                        ):
                            return direct_text
                except Exception:
                    continue

            all_p = tree.css('p, li')
            if all_p:
                texts = []
                for p in all_p:
                    p_text = p.text(separator=' ').strip()
                    min_len = 20 if p.tag == 'li' else 30
                    if len(p_text) > min_len and not _INTERSTITIAL_DISCLAIMER_RE.search(p_text):
                        texts.append(p_text)
                if texts:
                    result = re.sub(r'\s+', ' ', ' '.join(texts)).strip()
                    if len(result) >= _MIN_EXTRACTED_CHARS:
                        return result
        except Exception as exc:
            logger.debug(f"Selectolax parsing error, falling back to regex: {exc}")

    return _regex_extract_paragraphs(html_content)


def _try_decode_google_news_url(url: str) -> Optional[str]:
    """Attempt offline base64 decode for legacy Google News CBMi URLs."""
    try:
        parsed = urlparse(url)
        if 'news.google.com' not in parsed.netloc:
            return None

        for part in parsed.path.split('/'):
            if part.startswith('CBMi'):
                encoded = part[4:]
                padding = 4 - (len(encoded) % 4)
                if padding != 4:
                    encoded += '=' * padding
                try:
                    decoded = base64.urlsafe_b64decode(encoded).decode('utf-8', errors='ignore')
                    if decoded.startswith('http'):
                        return decoded.split('\x00')[0].strip()
                    http_idx = decoded.find('http')
                    if http_idx >= 0:
                        candidate = decoded[http_idx:].split('\x00')[0].strip()
                        if candidate.startswith(('http://', 'https://')) and '.' in candidate:
                            return candidate
                except Exception:
                    pass
        return None
    except Exception:
        return None


def _extract_gnews_article_id(url: str) -> Optional[str]:
    """Extract the CBMi... article ID token from a Google News URL."""
    try:
        parsed = urlparse(url)
        for part in parsed.path.split('/'):
            if part.startswith('CBMi'):
                return part
        return None
    except Exception:
        return None


class ArticleExtractor:
    """Asynchronous article full-text extractor with native Google News RPC resolution."""

    def __init__(self, timeout: float = 6.0):
        self._timeout = timeout
        self._session: Optional['CurlAsyncSession'] = None

    async def _get_session(self) -> Optional['CurlAsyncSession']:
        """Get or lazily initialize the persistent curl_cffi AsyncSession."""
        if not _HAS_CURL_CFFI:
            return None
        if self._session is None:
            self._session = CurlAsyncSession(impersonate="chrome120", max_clients=8)
        return self._session

    async def close(self) -> None:
        """Close the persistent HTTP session."""
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    async def _resolve_google_news_url_async(self, url: str) -> str:
        """Resolve a Google News URL to the canonical publisher URL asynchronously.

        1. Attempts fast offline base64 decoding for legacy URLs.
        2. For modern opaque CBMi tokens, fetches the splash page with the SOCS GDPR
           cookie to extract (data-n-a-ts, data-n-a-sg) and calls Google's internal
           'batchexecute' (Fbv4je) endpoint via curl_cffi Chrome 120.
        """
        offline_decoded = _try_decode_google_news_url(url)
        if offline_decoded:
            return offline_decoded

        article_id = _extract_gnews_article_id(url)
        session = await self._get_session()
        if not article_id or session is None:
            return url

        splash_url = f"https://news.google.com/rss/articles/{article_id}?oc=5"
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,it;q=0.8,de;q=0.7,fr;q=0.6',
            'Cookie': f'SOCS={_GOOGLE_SOCS_COOKIE}; CONSENT=YES+cb.20240715-00-p0.en+FX+111',
        }

        try:
            resp = await session.get(
                splash_url,
                headers=headers,
                timeout=self._timeout,
                allow_redirects=True,
            )
            # If Google redirected directly to the publisher domain, return it
            if resp.url and 'google.com' not in resp.url:
                return str(resp.url)

            html_body = resp.text or ""
            sg_match = _GNEWS_SG_RE.search(html_body)
            ts_match = _GNEWS_TS_RE.search(html_body)
            if not sg_match or not ts_match:
                return url

            sig = sg_match.group(1)
            ts = ts_match.group(1)

            inner_req = (
                f'["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",'
                f'null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,'
                f'null,0,0,null,0],"{article_id}",{ts},"{sig}"]'
            )
            rpc_payload = f'f.req={quote(json.dumps([[["Fbv4je", inner_req, None, "generic"]]]))}'

            rpc_headers = {
                'User-Agent': headers['User-Agent'],
                'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'Referer': 'https://news.google.com/',
                'Origin': 'https://news.google.com',
                'Cookie': headers['Cookie'],
            }
            rpc_resp = await session.post(
                "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je",
                data=rpc_payload,
                headers=rpc_headers,
                timeout=self._timeout,
            )
            if rpc_resp.status_code == 200 and rpc_resp.text:
                unescaped = rpc_resp.text.replace('\\"', '"').replace('\\/', '/')
                url_match = _GNEWS_RPC_URL_RE.search(unescaped)
                if url_match:
                    resolved = url_match.group(1)
                    logger.debug(f"[GNEWS DECODED] Risolto URL Google News -> {resolved}")
                    return resolved
        except Exception as exc:
            logger.debug(f"Google News RPC resolution failed for {article_id}: {exc}")

        return url

    async def fetch_full_text(self, url: str) -> str:
        """Fetch the full article text from a URL (resolving Google News links if needed)."""
        session = await self._get_session()
        if session is None:
            return ""

        target_url = url
        if 'news.google.com' in url:
            target_url = await self._resolve_google_news_url_async(url)
            if 'news.google.com' in target_url or 'consent.google.com' in target_url:
                return ""

        try:
            response = await session.get(
                target_url,
                headers={
                    'User-Agent': (
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/120.0.0.0 Safari/537.36'
                    ),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9,it;q=0.8,de;q=0.7,fr;q=0.6',
                    'Cookie': f'SOCS={_GOOGLE_SOCS_COOKIE}',
                },
                timeout=self._timeout,
                allow_redirects=True,
            )

            if response.status_code != 200:
                logger.debug(
                    f"Full-text fetch failed for {target_url}: HTTP {response.status_code}"
                )
                return ""

            html_content = response.text
            if not html_content or len(html_content) < 400:
                return ""

            extracted = _extract_text_from_html(html_content)
            if extracted:
                logger.debug(
                    f"[EXTRACTED] Extracted {len(extracted)} chars of full text from {target_url}"
                )
            return extracted

        except Exception as e:
            logger.debug(f"Full-text extraction failed for {target_url}: {e}")
            return ""

    async def extract_if_needed(
        self,
        title: str,
        clean_summary: str,
        url: str,
    ) -> str:
        """Conditionally extract full text if the RSS summary is insufficient (< 1200 chars)."""
        if not needs_full_text_extraction(title, clean_summary):
            return clean_summary

        full_text = await self.fetch_full_text(url)
        if full_text and len(full_text) > len(clean_summary):
            logger.info(
                f"[FULL-TEXT] Extracted {len(full_text)} chars "
                f"(was {len(clean_summary)} chars from RSS)"
            )
            return full_text

        return clean_summary
