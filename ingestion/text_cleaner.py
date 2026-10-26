"""Text cleaning and Context Window Cropping module.

Uses selectolax (C-level Lexbor HTML parser) with fallback to regex-based stripping.
Implements intelligent Head + Match + Tail context cropping and sentence-safe truncation.
"""

import html
import re
import unicodedata
from typing import List

from config.settings import MAX_CONTEXT_CHARS, CONTEXT_WINDOW_SENTENCES

# Try modern C-backed Lexbor HTML parser first, fallback to legacy parser or regex
try:
    from selectolax.lexbor import LexborHTMLParser
    _HAS_SELECTOLAX = True
except ImportError:
    try:
        from selectolax.parser import HTMLParser as LexborHTMLParser
        _HAS_SELECTOLAX = True
    except ImportError:
        _HAS_SELECTOLAX = False

# Boilerplate suffixes to remove from titles/text
_BOILERPLATE_PATTERNS = [
    re.compile(r'\s*[-–—|]\s*(Reuters|Bloomberg|ANSA|MilanoFinanza|Milano Finanza|Teleborsa|'
               r'SoldiOnline(?:\.it)?|FinanzaOnline|Investing\.com(?:\s+Deutsch|\s+Italia|\s+UK)?|'
               r'MarketWatch|CNBC|Financial Times|Handelsblatt|Les Echos|Il Sole 24 Ore|'
               r'Borsa Italiana|Corriere della Sera|La Repubblica|Der Spiegel|Le Monde|'
               r'The Guardian|BBC|Sky News|Yahoo Finance|Yahoo Finanza|Google Finance|'
               r'AD HOC NEWS|ad-hoc-news\.de|finanzen\.at|finanzen\.ch|finanzen\.net|'
               r'La Nazione|TradingView|EQS News|MarketScreener(?:\s+Deutschland|\s+Schweiz|\s+Italia)?|'
               r'Börse Express|boerse\.de|Wallstreet Online|Money\.it|Websim|Boursorama|'
               r'Zonebourse|Boursier(?:\.com)?|Frankfurter Rundschau|MarketBeat|TipRanks|Simply Wall St)\s*$', re.IGNORECASE),
    # Trailing publisher name without dash at end of Google News description
    re.compile(r'\s+(Teleborsa|AD HOC NEWS|ad-hoc-news\.de|finanzen\.at|finanzen\.ch|finanzen\.net|'
               r'Yahoo Finanza|La Nazione|TradingView|SoldiOnline(?:\.it)?|Money\.it|Websim|'
               r'Börse Express|boerse\.de|Wallstreet Online|MarketScreener(?:\s+Deutschland|\s+Schweiz)?|'
               r'Investing\.com(?:\s+Deutsch|\s+Italia)?|Boursorama|Frankfurter Rundschau|MarketBeat)\s*$', re.IGNORECASE),
    # Remove any residual inline domain mention of ad-hoc-news.de so it never triggers 'ad-hoc' taxonomy
    re.compile(r'\bad-hoc-news\.de\b', re.IGNORECASE),
    # Copyright footers
    re.compile(r'\s*©.*$', re.IGNORECASE),
    re.compile(r'\s*Copyright\s+\d{4}.*$', re.IGNORECASE),
    # "Read more" / "Continue reading" junk
    re.compile(r'\s*(Read more|Continue reading|Leggi tutto|Continua a leggere|'
               r'Weiterlesen|Lire la suite)\s*\.?\s*$', re.IGNORECASE),
    # Standard Forward-Looking Statements legal disclaimer blocks at the end of PRs
    re.compile(
        r'\s*(?:Disclaimer\s*:\s*Forward[- ]Looking\s+Statements?|'
        r'Avertissement\s*:\s*d[ée]clarations\s+prospectives)\s+'
        r'(?:This\s+press\s+release\s+contains|Ce\s+communiqu[ée]\s+de\s+presse\s+contient).*$',
        re.IGNORECASE | re.DOTALL,
    ),
]

_TOKEN_OVERLAP_RE = re.compile(r'\b[a-zA-Z0-9À-ÿ]{3,}\b')

# Sentence splitting regex protecting numeric/corporate abbreviations (Mio., Mrd., Dr., ca., vs., Inc., Ltd.)
_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?])'
    r'(?<!\b\d\.)(?<!\b\d{2}\.)'
    r'(?<!\bMio\.)(?<!\bMrd\.)(?<!\bmio\.)(?<!\bmrd\.)'
    r'(?<!\bca\.)(?<!\bvs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bNr\.)'
    r'(?<!\bCo\.)(?<!\bInc\.)(?<!\bLtd\.)(?<!\bcca\.)'
    r'\s+(?=[A-Z\dÀ-ÿ"„\'])'
)


def strip_html(text: str) -> str:
    """Remove all HTML tags and decode entities using selectolax or regex fallback."""
    if not text:
        return ""

    if _HAS_SELECTOLAX:
        try:
            tree = LexborHTMLParser(text)
            stripped = tree.text(separator=' ') or ""
        except Exception:
            stripped = _regex_strip_html(text)
    else:
        stripped = _regex_strip_html(text)

    # Decode HTML entities
    stripped = html.unescape(stripped)
    return stripped.strip()


def _regex_strip_html(text: str) -> str:
    """Fallback HTML stripping using regex."""
    # Remove script and style blocks
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form and clean up special characters."""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    # Replace various dash types with standard hyphen
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    # Replace fancy quotes with standard ones
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Remove common news feed boilerplate suffixes and prefixes."""
    if not text:
        return ""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub('', text)
    return text.strip()


def clean_text(raw_text: str) -> str:
    """Full cleaning pipeline: strip HTML -> normalize Unicode -> remove boilerplate."""
    text = strip_html(raw_text)
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def combine_title_and_summary(clean_title: str, clean_summary: str) -> str:
    """Combine cleaned title and summary while avoiding Google News title duplication.

    If clean_summary is empty, contained within clean_title, or shares >= 75%
    Jaccard word-token overlap with clean_title, returns only clean_title.
    """
    title_str = (clean_title or "").strip()
    summary_str = (clean_summary or "").strip()

    if not summary_str:
        return title_str
    if not title_str:
        return summary_str

    norm_title = re.sub(r'\W+', ' ', title_str.lower()).strip()
    norm_summary = re.sub(r'\W+', ' ', summary_str.lower()).strip()

    if norm_summary == norm_title or norm_summary in norm_title:
        return title_str

    title_tokens = set(_TOKEN_OVERLAP_RE.findall(title_str.lower()))
    summary_tokens = set(_TOKEN_OVERLAP_RE.findall(summary_str.lower()))

    if title_tokens and summary_tokens:
        intersection = len(title_tokens & summary_tokens)
        union = len(title_tokens | summary_tokens)
        jaccard = intersection / union if union > 0 else 0.0
        if jaccard >= 0.75 and len(summary_str) <= len(title_str) + 45:
            return title_str

    if title_str.endswith(('.', '!', '?')):
        return f"{title_str} {summary_str}"
    return f"{title_str}. {summary_str}"


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []
    sentences = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _sentence_safe_truncate(text: str, max_chars: int) -> str:
    """Truncate text without cutting numbers or words in half.

    Prefers the last full sentence boundary ('.', '!', '?') within the second half
    of the budget; otherwise falls back to the last word boundary (' ').
    """
    if len(text) <= max_chars:
        return text

    budget = max(0, max_chars - 4)
    candidate = text[:budget]

    # Look for a clean sentence ending in the second half of the allowed budget
    min_boundary_pos = budget // 2
    for punct in ('. ', '! ', '? ', '."', '.'):
        pos = candidate.rfind(punct)
        if pos >= min_boundary_pos:
            cut_idx = pos + (1 if punct.startswith('.') else len(punct))
            return candidate[:cut_idx].rstrip() + " ..."

    # Fallback: cut at the last whitespace to never mutilate numbers (e.g. "€150M" -> "€15...")
    last_space = candidate.rfind(' ')
    if last_space >= min_boundary_pos:
        return candidate[:last_space].rstrip() + " ..."

    return candidate + " ..."


def crop_context(
    text: str,
    match_terms: List[str],
    max_chars: int = MAX_CONTEXT_CHARS,
    window: int = CONTEXT_WINDOW_SENTENCES,
) -> str:
    """Progressive Budget Context Window Cropping (Head + Progressive Match Expansion + Tail).

    - If len(text) <= max_chars (8000), returns 100% of the text untouched.
    - If len(text) > max_chars, preserves the opening Head sentences, the closing Tail
      sentences, and progressively expands the sentence radius around match_terms
      until the available max_chars budget is densely filled (avoiding any step-cliff
      collapse to a few sentences), finishing with sentence-safe truncation.
    """
    if len(text) <= max_chars:
        return text

    sentences = split_sentences(text)
    if not sentences or len(sentences) <= 1:
        return _sentence_safe_truncate(text, max_chars)

    match_indices = set()
    lower_terms = [t.lower() for t in match_terms if t]

    for idx, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        for term in lower_terms:
            if term in sentence_lower:
                match_indices.add(idx)
                break

    if not match_indices:
        match_indices = {0}

    selected_indices = set()

    # 1. Always preserve opening lead (first 2 sentences) and closing fine print (last 2 sentences)
    if max_chars >= 1200 and len(sentences) >= 4:
        selected_indices.add(0)
        selected_indices.add(1)
        selected_indices.add(len(sentences) - 1)
        if len(sentences) >= 6:
            selected_indices.add(len(sentences) - 2)

    # 2. Initial window around every match
    for idx in match_indices:
        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        for i in range(start, end):
            selected_indices.add(i)

    def _build_selected_text(indices_set: set) -> str:
        ordered = sorted(indices_set)
        parts = []
        prev = -2
        for i in ordered:
            if i > prev + 1 and prev >= 0:
                parts.append('[...]')
            parts.append(sentences[i])
            prev = i
        return ' '.join(parts)

    # 3. Progressive radius expansion: keep adding adjacent sentences around matches
    # and from the lead until we fill at least 90% of max_chars (or exhaust all sentences)
    radius = window + 1
    max_radius = len(sentences)
    target_fill = int(max_chars * 0.98)

    while len(selected_indices) < len(sentences) and radius <= max_radius:
        current_text = _build_selected_text(selected_indices)
        if len(current_text) >= target_fill:
            break

        added_any = False
        for idx in sorted(match_indices):
            for candidate_idx in (idx + radius, idx - radius):
                if 0 <= candidate_idx < len(sentences) and candidate_idx not in selected_indices:
                    selected_indices.add(candidate_idx)
                    added_any = True
                    if len(_build_selected_text(selected_indices)) >= target_fill:
                        break
            if len(_build_selected_text(selected_indices)) >= target_fill:
                break

        if not added_any:
            for seq_idx in range(len(sentences)):
                if seq_idx not in selected_indices:
                    selected_indices.add(seq_idx)
                    break

        radius += 1

    result = _build_selected_text(selected_indices)
    if len(result) > max_chars:
        result = _sentence_safe_truncate(result, max_chars)

    return result