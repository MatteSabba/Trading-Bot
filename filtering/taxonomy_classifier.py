"""Multilingual (EN/IT/DE/FR) weighted lexical taxonomy classifier.

Assigns event_category and priority_level to news items using a weighted
multi-score algorithm across all categories (including European-specific catalysts
like Accelerated Bookbuilding, Ad-hoc-Mitteilungen, and Capital Increases),
with headline/incipit boost and regulatory risk precedence.
Supports portfolio priority override.
"""

import re
from typing import List, Optional, Set, Tuple

from database.db_manager import DatabaseManager
from config.settings import (
    BOARD_CONTRACT_EXCLUSION_PATTERNS,
    CALENDAR_INVITATION_PATTERNS,
    ROUTINE_BUYBACK_PATTERNS,
    ROUTINE_SHORT_DISCLOSURE_PATTERNS,
)

# B2B idioms that contain accounting words ('revenue', 'contract') but describe products/services
_NON_FINANCIAL_IDIOMS_RE = re.compile(
    r'\b(?:revenue\s+cycle(?:\s+management)?|revenue\s+operations|revenue\s+sharing|'
    r'contract\s+manufacturing|contract\s+logistics|contract\s+research|contract\s+catering|'
    r'in\s+order\s+to|out\s+of\s+order)\b',
    re.IGNORECASE,
)


# ── Taxonomy Definitions ──
# Each category: (category_name, keywords_set, priority_level)
_TAXONOMY: List[Tuple[str, Set[str], int]] = [
    (
        "M&A_CORPORATE",
        {
            # English
            "takeover", "takeovers", "merger", "mergers", "acquisition", "acquisitions",
            "acquire", "acquires", "acquired", "buyout", "buyouts", "spin-off", "delisting",
            "joint venture", "definitive agreement", "board unanimously", "unanimously approved",
            "stake", "accelerated bookbuilding", "bookbuilding", "private placement",
            "capital increase", "rights issue", "demerger",
            "rule 2.7", "rule 2.4", "firm intention to make an offer",
            "possible offer", "recommended cash offer", "recommended offer",
            # Italian
            "opa", "opas", "fusione", "fusioni", "acquisizione", "acquisizioni", "scissione",
            "collocamento accelerato", "aumento di capitale", "cessione", "cessioni",
            # German
            "übernahme", "übernahmen", "fusion", "kapitalerhöhung", "übernehmen", "übernahmeangebot",
            # French
            "rachat", "augmentation de capital", "fusion",
            # Spanish
            "adquisición", "adquisiciones", "fusión", "fusiones", "ampliación de capital",
        },
        1,
    ),
    (
        "EARNINGS_GUIDANCE",
        {
            # English
            "q1", "q2", "q3", "q4", "h1", "fy", "ebitda", "ebit", "earnings",
            "net profit", "net income", "revenue", "revenues", "quarterly sales",
            "nine-month sales", "full-year sales", "annual sales", "lower sales", "higher sales",
            "sales growth", "sales rose", "sales fell", "organic growth", "dividend", "dividends",
            "margin", "margins", "guidance", "outlook", "profit warning", "trading update",
            "preliminary results", "eqs-adhoc", "eqs-ad-hoc", "ad-hoc-mitteilung",
            "ad-hoc announcement", "share buyback", "share repurchase",
            # Italian
            "trimestrale", "trimestrali", "semestrale", "semestrali", "bilancio", "risultati",
            "utile", "utili", "utile netto", "ricavi", "fatturato", "dividendo", "dividendi",
            "margine", "margini", "preconsuntivo", "preconsuntivi", "azioni proprie",
            # German
            "quartalszahlen", "prognose", "vorläufige zahlen", "geschäftsbericht", "dividende",
            "umsatz", "konzernumsatz", "erlöse", "gewinnwarnung", "umsatzwarnung",
            # French
            "résultats", "chiffres provisoires", "dividende", "chiffre d'affaires",
            "prévision", "prévisions", "relèvement de prévision", "relève ses prévisions",
            "abaisse ses prévisions",
            # Spanish
            "resultados", "beneficio", "beneficios", "ingresos", "dividendo", "dividendos",
        },
        1,
    ),
    (
        "CONTRACT_ORDER",
        {
            # English
            "contract", "contracts", "order intake", "new order", "new orders",
            "orders", "backlog", "awarded", "tender",
            # Italian
            "commessa", "commesse", "contratto", "contratti", "appalto", "appalti", "aggiudica",
            "ordine", "ordini", "nuovo ordine", "nuovi ordini", "portafoglio ordini",
            # German
            "auftrag", "aufträge", "großauftrag",
            # French
            "contrat", "contrats", "commande", "commandes",
            # Spanish
            "adjudicación", "adjudicaciones", "pedido", "pedidos", "contrato", "contratos",
        },
        2,
    ),
    (
        "REGULATORY_LEGAL",
        {
            # English
            "ema", "fda", "antitrust", "patent", "patents", "approval", "approvals",
            "investigation", "investigations", "fines", "fined", "regulatory fine",
            "antitrust fine", "heavy fine", "penalty", "golden power",
            # Italian
            "consob", "multa", "multe", "brevetto", "brevetti", "indagine", "indagini",
            "sanzione", "sanzioni",
            # German
            "bafin", "kartellamt", "kartellamtsentscheid", "untersuchung", "bußgeld",
            # French
            "amf", "fca", "enquête", "amende",
            # Spanish
            "cnmv", "multa", "multas", "investigación", "sanción", "sanciones",
        },
        2,
    ),
    (
        "ANALYST_RATING",
        {
            # English
            "upgrade", "upgrades", "downgrade", "downgrades", "target price",
            "buy rating", "outperform", "overweight", "underweight",
            "raises target", "cuts target", "price target", "initiates coverage",
            "reiterates buy",
            # Italian
            "raccomandazione", "raccomandazioni", "prezzo obiettivo", "target price",
            "alza il target", "taglia il target", "conferma buy",
            "avvia copertura", "giudizio buy", "giudizio hold",
            # German
            "kursziel", "kaufempfehlung",
            "hob ziel", "senkt ziel", "senkte ziel", "senken das kursziel", "erhöht das kursziel", "senkt das kursziel",
            "startet mit", "stuft", "bestätigt", "kaufen", "halten", "verkaufen",
            # French
            "objectif de cours",
            "relève son objectif", "abaisse son objectif", "entame le suivi",
            # Spanish
            "precio objetivo", "recomendación",
            "eleva el precio objetivo", "recorta el precio objetivo",
        },
        3,
    ),
]

# High-severity regulatory enforcement terms that override weak M&A/General matches
_HARD_REGULATORY_TERMS = re.compile(
    r'\b(?:antitrust|investigation|indagine|consob|bafin|amf|multa|fines|fined|regulatory\s+fine|antitrust\s+fine|sanzione|golden\s+power)\b',
    re.IGNORECASE,
)

_BOARD_CONTRACT_EXCLUSION_RE = re.compile(
    r'(?:' + '|'.join(BOARD_CONTRACT_EXCLUSION_PATTERNS) + r')',
    re.IGNORECASE,
)

_CALENDAR_INVITATION_RE = re.compile(
    r'(?:' + '|'.join(CALENDAR_INVITATION_PATTERNS) + r')',
    re.IGNORECASE,
)

_ROUTINE_SHORT_DISCLOSURE_RE = re.compile(
    r'(?:' + '|'.join(ROUTINE_SHORT_DISCLOSURE_PATTERNS) + r')',
    re.IGNORECASE,
)

_ROUTINE_BUYBACK_DISCLOSURE_RE = re.compile(
    r'(?:' + '|'.join(ROUTINE_BUYBACK_PATTERNS) + r')',
    re.IGNORECASE,
)

# Generic market wrap headlines (e.g., "Borsa: Milano in rialzo...", "Borsa: l'Europa...")
_MARKET_WRAP_HEADLINE_RE = re.compile(
    r'^\s*borsa\s*:\s*(?:milano|l[\'’]europa|europa|wall\s+street|tokyo|asia|cambi)\b',
    re.IGNORECASE,
)

# Hard financial performance figures that prove an article contains actual results, not just an invitation
_ACTUAL_EARNINGS_FIGURES_RE = re.compile(
    r'(?:\b(?:ebitda|ebit|net\s+profit|utile\s+netto|guidance\s+raised|guidance\s+cut|profit\s+warning)\b'
    r'|\b\d+[,.]?\d*\s*(?:%|percent|prozent|mln|mrd|mio|million|billion|miliardi|milioni)\b)',
    re.IGNORECASE,
)

# German and Nordic compound roots (>= 6 chars) that can fuse into composite words
# e.g., Umsatzprognose, Kartellamtsentscheid, Konzernumsatz, Jahresprognose
_GERMAN_COMPOUND_ROOTS = {
    "prognose", "umsatz", "umsätze", "gewinn", "ergebnis", "dividende",
    "übernahme", "übernehm", "kartellamt", "auftrag", "aufträge",
    "quartalszahl", "geschäftsbericht", "kapitalerhöhung", "gewinnwarnung",
}

# Pre-compile taxonomy patterns:
# - Short terms (< 6 chars) use strict word boundaries \bkw\b to prevent false matches (e.g. \bopa\b)
# - Long German compound roots use \b\w*?(?:root)\w*?\b to capture agglutinated words (e.g. Umsatzprognose)
_COMPILED_TAXONOMY: List[Tuple[str, re.Pattern, int]] = []
for _cat, _keywords, _prio in _TAXONOMY:
    sorted_kw = sorted(_keywords, key=len, reverse=True)
    regex_parts = []
    for kw in sorted_kw:
        low_kw = kw.lower()
        if low_kw in _GERMAN_COMPOUND_ROOTS:
            regex_parts.append(rf'\b\w*?{re.escape(low_kw)}\w*?\b')
        else:
            regex_parts.append(rf'\b{re.escape(kw)}\b')
    pattern_str = '(?:' + '|'.join(regex_parts) + ')'
    _COMPILED_TAXONOMY.append((_cat, re.compile(pattern_str, re.IGNORECASE), _prio))


class TaxonomyClassifier:
    """Multilingual news event classifier with weighted multi-scoring and portfolio override."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self._portfolio_tickers: Set[str] = set()
        self._loaded = False

    def load_portfolio_state(self) -> None:
        """Load the set of tickers with open portfolio positions."""
        rows = self.db.fetch_all(
            "SELECT ticker FROM watchlist WHERE is_portfolio_open = 1"
        )
        self._portfolio_tickers = {row['ticker'] for row in rows}
        self._loaded = True

    def classify(
        self, text: str, matched_tickers: List[str]
    ) -> Tuple[str, int]:
        """Classify the event category and priority level of a news text using weighted scoring.

        Args:
            text: Combined title + summary text.
            matched_tickers: List of ticker symbols found in the text.

        Returns:
            Tuple of (event_category, priority_level).
        """
        if not self._loaded:
            self.load_portfolio_state()

        event_category = "GENERAL"
        priority_level = 3

        if text:
            # Mask out non-financial B2B idioms (e.g., 'revenue cycle management' -> 'b2b_service')
            scoring_text = _NON_FINANCIAL_IDIOMS_RE.sub('b2b_service', text)
            incipit = scoring_text[:160]
            best_score = 0.0
            scores_by_cat = {}
            prio_by_cat = {}

            for idx, (cat_name, cat_pattern, cat_priority) in enumerate(_COMPILED_TAXONOMY):
                all_matches = [m.group().lower() for m in cat_pattern.finditer(scoring_text)]
                if not all_matches:
                    continue

                # Guard 1: Do not classify Executive/Management Board contract renewals as commercial CONTRACT_ORDER
                if cat_name == "CONTRACT_ORDER" and _BOARD_CONTRACT_EXCLUSION_RE.search(scoring_text):
                    continue

                # Guard 2: Do not classify Webcast/Conference Call calendar invitations as EARNINGS_GUIDANCE
                # unless actual financial figures or guidance actions are present
                if (
                    cat_name == "EARNINGS_GUIDANCE"
                    and _CALENDAR_INVITATION_RE.search(scoring_text)
                    and not _ACTUAL_EARNINGS_FIGURES_RE.search(scoring_text)
                ):
                    continue

                # Guard 3: Routine Consob/EU short-selling (PNC) daily position updates are GENERAL (Prio 3),
                # not REGULATORY_LEGAL enforcement/investigations
                if cat_name == "REGULATORY_LEGAL" and _ROUTINE_SHORT_DISCLOSURE_RE.search(scoring_text):
                    continue

                # Guard 4: Routine weekly treasury share buyback disclosures are administrative GENERAL (Prio 3),
                # not primary M&A or Earnings catalysts
                if cat_name in ("M&A_CORPORATE", "EARNINGS_GUIDANCE") and _ROUTINE_BUYBACK_DISCLOSURE_RE.search(scoring_text):
                    continue

                # Guard 5: Generic market wrap headlines ("Borsa: Milano...", "Borsa: l'Europa...") are GENERAL (Prio 3)
                if cat_name in ("M&A_CORPORATE", "EARNINGS_GUIDANCE") and _MARKET_WRAP_HEADLINE_RE.search(incipit):
                    continue

                unique_matches = set(all_matches)
                score = float(len(unique_matches)) + 0.25 * (len(all_matches) - len(unique_matches))

                if cat_pattern.search(incipit):
                    score += 1.5

                score += (len(_COMPILED_TAXONOMY) - idx) * 0.01

                scores_by_cat[cat_name] = score
                prio_by_cat[cat_name] = cat_priority

                if score > best_score:
                    best_score = score
                    event_category = cat_name
                    priority_level = cat_priority

            # Regulatory precedence override: if hard antitrust/investigation keywords appear
            # in the headline/incipit, elevate to REGULATORY_LEGAL with Priority 1
            if (
                "REGULATORY_LEGAL" in scores_by_cat
                and _HARD_REGULATORY_TERMS.search(incipit)
                and scores_by_cat.get("REGULATORY_LEGAL", 0.0) >= scores_by_cat.get(event_category, 0.0) - 0.5
            ):
                event_category = "REGULATORY_LEGAL"
                priority_level = prio_by_cat["REGULATORY_LEGAL"]

        # Portfolio override: force priority 1 if any matched ticker is in portfolio
        if matched_tickers and self._portfolio_tickers:
            for ticker in matched_tickers:
                if ticker in self._portfolio_tickers:
                    priority_level = 1
                    break

        return event_category, priority_level