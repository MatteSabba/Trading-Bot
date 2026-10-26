import os
from pathlib import Path

# ── Project Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "trading_bot.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"
WATCHLIST_FILE = BASE_DIR / "eu_small_mid_caps.txt"
NEWS_SOURCES_FILE = BASE_DIR / "eu_news_sources.txt"

# ── HTTP / Anti-Ban & Rate-Limiting Settings ──
HTTP_TIMEOUT_SEC = 10
MAX_CONSECUTIVE_FAILURES = 5
AUTO_CIRCUIT_BREAKER_THRESHOLD = 10
BACKOFF_MULTIPLIER = 2
MAX_CONCURRENT_RSS_REQUESTS = 8
POLL_JITTER_PCT = 0.15  # ±15% random jitter on per-source polling interval

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.2; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

ACCEPT_HEADER = "application/rss+xml, application/rdf+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5"
ACCEPT_LANGUAGE = "en-US,en;q=0.9,it-IT;q=0.8,it;q=0.7"

# ── Context Window Cropping & Freshness TTL ──
MAX_CONTEXT_CHARS = 8000  # Expanded to 8000 to preserve full transcripts, tables and deal terms
FULL_TEXT_MIN_SUMMARY_CHARS = 1200  # Trigger full-text web extraction when RSS summary < 1200 chars
CONTEXT_WINDOW_SENTENCES = 2  # initial sentence radius around match before progressive expansion
MAX_NEWS_AGE_HOURS = 4  # Discard stale RSS items older than 4 hours during live ingestion
STALE_NEWS_MINUTES_THRESHOLD = 60  # Flag news older than 60 mins as likely priced-in
SEMANTIC_DEDUP_WINDOW_MINUTES = 45  # Extended from 15 to 45 min to catch delayed cross-source/cross-language syndication
SEMANTIC_DEDUP_JACCARD_THRESHOLD = 0.55  # Title token similarity threshold

# ── Deduplication ──
LRU_DEDUP_SIZE = 15000

# ── Google News Dynamic Feed ──
GNEWS_BATCH_SIZE = 5  # max companies per Google News query
GNEWS_TIME_WINDOW = "1h"

# ── Short/Ambiguous Ticker Blacklist ──
AMBIGUOUS_SYMBOLS = {
    "IP", "SOL", "AM", "BB", "ERA", "SK", "RF", "TE", "FR", "AF",
    "BOY", "DOM", "BAG", "ASC", "COA", "HOT", "LEG", "PAT", "ACT",
    "DOU", "VIS", "MEL", "DIA", "BG", "DOC", "POST", "STR", "SUN",
    "HEM", "JM", "SF", "ISS", "TOM", "FRO", "NAS", "VEI", "GE",
    "SO", "MN", "CE", "AV", "EM", "UD", "FM", "ACE", "IRE", "TIP",
    "BAN", "TES", "LR", "FF", "SAX", "FIE", "HAS", "PAG", "AO",
    "BAR", "REC", "FLOW",
}

# ── Stopwords e nomi corti ad alta collisione che devono preservare il suffisso societario ──
AMBIGUOUS_CLEAN_NAMES = {
    "nel", "ses", "sto", "sol", "man", "air", "del", "sur",
    "per", "con", "tra", "fra", "solar", "post", "next", "iss",
    "cvs", "sig",
    "workplace", "workspace", "general", "national", "global", "first", "standard",
}

# ── Legal & Descriptive Corporate Suffix Regex for clean_name generation ──
LEGAL_SUFFIXES_PATTERN = (
    r"(?:\s+&\s*C\.?\b|"
    r"\b(?:SpA|S\.p\.A\.|AG|SE|SA|plc|NV|AB|Oyj|ASA|A/S|Group|Holding|Holdings|"
    r"International|Ltd|Corp|Corporation|KGaA|GmbH|N\.V\.|S\.A\.|"
    r"Football\s+Club|Entertainment|MediaGroup|e-solutions|Lagonda)\b)"
)

# ── Additional Brand Aliases for Multi-Word European Companies ──
COMPANY_BRAND_ALIASES = {
    "SFER.MI": ["Ferragamo"],
    "MN.MI": ["Mondadori"],
    "CE.MI": ["Credem"],
    "MAIRE.MI": ["Maire"],
    "LHA.DE": ["Lufthansa"],
    "BDB.MI": ["Banco Desio"],
    "CLI.MI": ["Centrale del Latte"],
    "DIS.MI": ["d'Amico"],
    "JDW.L": ["Wetherspoon"],
    "DOM.L": ["Domino's"],
    "BAMNB.AS": ["BAM Groep"],
    "VPK.AS": ["Vopak"],
    "IWG.L": ["International Workplace"],
}

# ── Financial Context Vocabulary (multilingual + European currencies & units) ──
FINANCIAL_CONTEXT_WORDS = {
    # English
    "shares", "stock", "board", "ebitda", "ebit", "revenue", "profit", "takeover",
    "merger", "acquisition", "dividend", "guidance", "contract", "order",
    "earnings", "quarterly", "annual", "fiscal", "investor", "trading",
    "bookbuilding", "abb", "buyback", "repurchase", "placement", "rns", "ad-hoc",
    # Italian
    "azioni", "borsa", "utile", "ricavi", "cda", "acquisizione",
    "trimestrale", "commessa", "dividendo", "fatturato", "collocamento",
    "aumento di capitale", "azioni proprie", "piazza affari",
    # German
    "aktie", "umsatz", "gewinn", "übernahme", "quartalszahlen", "prognose",
    "eqs-adhoc", "hauptversammlung",
    # French
    "chiffre d'affaires", "bénéfice", "résultats", "rachat",
    # Spanish
    "acciones", "bolsa", "beneficio", "ingresos", "ebitda", "dividendo",
    "adquisición", "fusión", "resultados", "contrato", "cnmv",
    # European Currencies, Symbols & Units (EUR, GBP, CHF, Nordics SEK/NOK/DKK)
    "€", "$", "£", "eur", "gbp", "pence", "chf", "sek", "nok", "dkk",
    "meur", "mchf", "msek", "mnok", "mdkk", "mln", "bln", "bn",
    "million", "billion", "milioni", "miliardi", "mio", "mrd",
    "milliard", "milliards",
}

# ── Strict Equity / Corporate Financial Vocabulary (required when B2C/Sports noise is detected) ──
HARD_FINANCIAL_CONTEXT_WORDS = {
    "shares", "stock", "equity", "shareholders", "ebitda", "ebit", "net profit",
    "net income", "dividend", "guidance", "takeover", "merger", "buyback",
    "capital increase", "accelerated bookbuilding", "delisting", "consensus",
    "target price", "quarterly results", "ad-hoc", "eqs-adhoc", "rns",
    "azioni", "azionisti", "borsa", "piazza affari", "xetra", "euronext",
    "utile netto", "dividendo", "cda", "consiglio di amministrazione", "opa",
    "aumento di capitale", "collocamento accelerato", "trimestrale", "semestrale",
    "bilancio", "prezzo obiettivo", "consob", "bafin", "amf", "aktie",
    "hauptversammlung", "bénéfice net",
}

# ── Domain Noise Blacklist (Sports, Motorsport, Fashion, Retail Promotions) ──
DOMAIN_NOISE_BLACKLIST = {
    # Sports & Football competitions
    "formula 1", "f1", "motogp", "gran premio", "gp", "scuderia", "pilota",
    "campionato", "serie a", "bundesliga", "premier league", "la liga", "ligue 1",
    "champions league", "europa league", "conference league", "coppa italia",
    "calciomercato", "calciatore", "allenatore", "partita", "rigore", "tifosi",
    "sfilata", "fashion week", "passerella", "collezione autunno", "collezione primavera",
    "black friday", "cyber monday", "incidente stradale",
    # German sports, match previews & betting noise (prevents Prognose on football clubs)
    "sportwetten", "wettfreunde", "wett-tipp", "tipp", "quoten", "wettquoten", "wetten",
    "anstoßzeit", "aufstellungen", "spieltag", "k.o.-phase", "gruppenphase", "heimvorteil",
    "spitzenspiel", "tabellenführer", "auswärtsspiel", "heimspiel", "allianz stadium",
    "pflichtspiel", "halbzeit", "stürmer", "torwart", "schiedsrichter",
    # English/International betting & match preview noise
    "match preview", "predicted lineups", "betting tips", "odds", "bookmaker",
    "kick-off", "matchday", "head-to-head", "starting xi",
    "pronostico", "pronostici", "scommesse", "probabili formazioni", "capolista",
    "trasferta", "giornata di campionato", "fase a gironi",
}

# ── Common words & B2C/Sports brands requiring mandatory financial context & case sensitivity ──
COMMON_NAME_WORDS = {
    # Dictionary words that are also European company names (comprehensive watchlist audit)
    "reply", "basicnet", "fiera milano", "basic-fit", "premier foods",
    "man group", "hunting", "hays", "pagegroup", "greggs", "solar",
    "catena", "elis", "new wave", "novo", "fii", "clari",
    "frontline", "workplace", "workspace", "workspace group", "international workplace",
    "international workplace group", "rational", "believe", "mony", "marshalls",
    "savills", "costain", "grainger", "bellway", "elementis", "chesnara", "clarkson",
    "firstgroup", "next", "focus", "synergie", "alten", "spie", "barco", "fagron",
    "signify", "arcadis", "postnl", "tomtom", "sulzer", "forbo", "komax", "tecan",
    "valmet", "kemira", "fiskars", "harvia", "sanoma", "raisio", "subsea",
    "storebrand", "europris", "hafnia", "kitron", "pandox", "fabege", "castellum", "loomis",
    # High consumer / motorsport / sports / retail exposure brands (both full and cleaned forms)
    "brembo", "geox", "technogym", "piaggio", "piaggio & c", "danieli", "danieli & c",
    "de' longhi", "salvatore ferragamo", "ferragamo", "mondadori",
    "juventus", "juventus football club", "borussia dortmund",
    "hugo boss", "puma", "fnac darty", "ubisoft", "ubisoft entertainment",
    "aston martin", "aston martin lagonda", "rcs", "rcs mediagroup",
    "txt", "txt e-solutions", "dr. martens", "wh smith", "currys", "asos",
    "hellofresh", "douglas", "tui", "sixt",
}

# ── High-collision English/Italian dictionary words requiring HARD equity context ──
STRICT_DICTIONARY_COMPANY_NAMES = {
    "frontline", "workplace", "workspace", "rational", "believe", "hunting",
    "reply", "catena", "mony", "marshalls", "savills", "costain", "grainger",
    "elementis", "firstgroup", "next", "solar", "elis", "spie", "alten",
}

# ── Exchange Prefix Codes for EXPLICIT_SYMBOL pattern ──
EXCHANGE_PREFIXES = [
    "MIL", "BIT", "XETRA", "ETR", "EPA", "LON", "LSE", "AMS",
    "EBR", "BME", "SWX", "VIE", "STO", "CPH", "HEL", "OSL",
]


# ── Passive Noise Patterns (Form 8.3/8.5 UK, SEO retrospective articles, voting rights) ──
# These patterns identify articles with zero operational catalyst that pollute the pipeline.
# IMPORTANT: Do NOT block Rule 2.7 (Firm Intention to Make an Offer) or Rule 2.4 (Possible Offer)!
PASSIVE_NOISE_PATTERNS = [
    # UK Takeover Panel & EU passive regulatory holdings disclosures (Form 8.3, 8.5, TR-1, PDMR, WpHG)
    r'\bform\s+8\.(?:3|5)\b',
    r'\brule\s+8\.(?:3|5)\b',
    r'\btr-1\b.*?\bnotification of major holdings\b',
    r'\bholding\(s\)\s+in\s+company\b',
    r'\brelease\s+according\s+to\s+article\s+40,\s*section\s+1\s+of\s+the\s+wphg\b',
    r'\btotal voting rights\b',
    r'\bdirector/pdmr\s+shareholding\b',
    r'\bministry\s+of\s+finance\s+on\s+behalf\s+of\s+the\s+state\s+of\s+norway\b',
    r'\bwilmington,\s+(?:delaware|de),\s+united\s+states\s+of\s+america\b',
    # German (SEO retrospective return calculators - all variants including Investition/bedeutet/angefallen)
    r'so\s+viel\s+(?:gewinn|verlust)\s+(?:hätte|wäre)\s+(?:ein|eine|bei\s+einem)\s+.*?(?:investment|investition)',
    r'so\s+viel\s+hätten\s+anleger\s+mit\s+(?:einem\s+investment|einer\s+investition)',
    r'von\s+vor\s+\d+\s+jahren?\s+(?:eingebracht|eingefahren|abgeworfen|verloren|gewonnen|bedeutet|angefallen)',
    r'vor\s+(?:einem|\d+)\s+jahren?\s+(?:abgeworfen|verloren|gewonnen|bedeutet|angefallen)',
    # German & Multilingual automated technical chart / price-level noise before earnings
    r'\bliegt\s+\d+[,.]?\d*\s*(?:prozent|%)\s+unter\s+(?:dem\s+)?(?:jahreshoch|allzeithoch|52-wochen-hoch)\b',
    r'\b(?:notiert|kostet|fällt|steigt|legt)\s+.*?\bvor\s+(?:den\s+|dem\s+)?(?:quartalszahlen|geschäftszahlen|zahlen|quartalsbericht|bericht\s+zum)\b',
    r'\bvor\s+(?:den\s+|dem\s+)?(?:quartalszahlen|quartalsbericht)\s+(?:im\s+fokus|unter\s+druck|leichter|fester)\b',
    r'\bsorgt\s+derzeit\s+für\s+gesprächsstoff\s+an\s+der\s+börse\b',
    r'\bstehen\s+vor\s+einer\s+richtungsweisenden\s+woche\b',
    r'\b(?:scambia|costa|cede|sale)\s+.*?\bprima\s+dei\s+(?:conti|risultati\s+trimestrali)\b',
    # English (SEO retrospective return calculators)
    r'if\s+you\s+had\s+invested\s+.*?\s+years?\s+ago',
    r'how\s+much\s+a\s+.*?\s+investment\s+.*?\s+years?\s+ago',
    # Italian (SEO retrospective return calculators)
    r'quanto\s+avresti\s+guadagnato\s+investendo',
    r'se\s+avessi\s+investito\s+.*?\s+anni\s+fa',
    # French (SEO retrospective return calculators)
    r'combien\s+auriez-vous\s+gagné\s+en\s+investissant',
    # Spanish (SEO retrospective return calculators)
    r'cuánto\s+habrías\s+ganado\s+invirtiendo',
    # Quotation-only pages without news (Boursorama / LSE / Reuters / Teleborsa static quote pages)
    r'\bcours\s+action\s+[A-Z0-9.\-]+\s*,\s*cotation\s+bourse\b',
    r'\b(?:equity|stock|share)\s+price\s+and\s+quote\b',
    r'\bstock\s+price\s*(?:&|and)\s*latest\s+news\b',
    r'\b(?:aktienkurs\s+und\s+chart|quotazione\s+azioni\s+e\s+grafico)\b',
    r'\bperformance\s+negativa\s+per\s+.*?\b_TLB-AUTO\b',
    # Sports betting & football match predictions
    r'\b(?:tipp|ki-prognose|wettquoten|sportwetten|wettfreunde)\b',
    r'\b(?:prognose|aufstellungen|anstoßzeit)\b.*?\b(?:champions\s+league|europa\s+league|serie\s+a|bundesliga|spitzenspiel|tabellenführer|allianz\s+stadium)\b',
    r'\b(?:juventus|borussia\s+dortmund)\s+gegen\s+[a-zA-ZÀ-ÿ\s]+\bprognose\b',
]

# Override safeguard: real takeover offer announcements must NOT be blocked
PASSIVE_NOISE_OVERRIDE_PATTERNS = [
    r'\brule\s+2\.7\b',
    r'\brule\s+2\.4\b',
    r'\bfirm\s+intention\s+to\s+make\s+an\s+offer\b',
    r'\bpossible\s+offer\b',
    r'\brecommended\s+(?:cash\s+)?offer\b',
]

# ── Financial Intermediary Tickers (comprehensive watchlist audit of 26 brokers/asset managers) ──
# When these tickers appear in analyst-action context, they are the ISSUER, not the SUBJECT.
FINANCIAL_INTERMEDIARY_TICKERS = {
    "EQUI.MI",  # Equita Group (IT)
    "MB.MI",    # Mediobanca (IT)
    "BGN.MI",   # Banca Generali (IT)
    "AZM.MI",   # Azimut Holding (IT)
    "IF.MI",    # Banca Ifis (IT)
    "TIP.MI",   # Tamburi Investment Partners (IT)
    "DWS.DE",   # DWS Group (DE)
    "DBAN.DE",  # Deutsche Beteiligungs (DE)
    "MLP.DE",   # MLP SE (DE)
    "FTK.DE",   # flatexDEGIRO (DE)
    "EMG.L",    # Man Group (UK)
    "III.L",    # 3i Group (UK)
    "TCAP.L",   # TP ICAP Group / TP ICAP Midcap (UK)
    "ABDN.L",   # abrdn plc (UK)
    "ASHM.L",   # Ashmore Group (UK)
    "INVP.L",   # Investec (UK)
    "RAT.L",    # Rathbones Group (UK)
    "JUP.L",    # Jupiter Fund Management (UK)
    "AJB.L",    # AJ Bell (UK)
    "IGG.L",    # IG Group Holdings (UK)
    "CMCX.L",   # CMC Markets (UK)
    "BPT.L",    # Bridgepoint Group (UK)
    "AMUN.PA",  # Amundi (FR)
    "RF.PA",    # Eurazeo (FR)
    "MF.PA",    # Wendel (FR)
    "VLK.AS",   # Van Lanschot Kempen (NL)
    "FLOW.AS",  # Flow Traders (NL)
}

# Patterns indicating the intermediary is ISSUING a rating/recommendation on a third party
ANALYST_ISSUER_ACTION_PATTERNS = [
    # Italian (supports "alza target price", "abbassa target price", "upgrade a Buy da Equita", etc.)
    r'(?:alza|alzano|taglia|tagliano|abbassa|abbassano|riduce|riducono|incrementa|ritocca)\s+(?:il\s+)?target\s+price',
    r'target\s+price\s+(?:a|pari\s+a)\s+\d+[,.]?\d*\s*euro',
    r'conferma\s+(?:il\s+giudizio\s+|il\s+)?(?:buy|hold|neutral|outperform|sell|underperform)',
    r'\b(?:upgrade|downgrade)\s+(?:a\s+(?:buy|outperform|overweight|hold|neutral|sell|underperform)\s+)?(?:da|di)\b',
    r'\bmigliorato\s+a\s+["\'«“]?(?:buy|outperform|hold|neutral)',
    r'\b(?:equita|mediobanca|intermonte|banca\s+akros|kepler)\s+vede\b',
    r'promozione\s+di',
    r'avvia\s+(?:la\s+)?copertura',
    r'secondo\s+gli\s+analisti\s+di',
    r'raccomandazione\s+(?:buy|hold|sell)',
    # English
    r'raises?\s+target',
    r'cuts?\s+target',
    r'lowers?\s+target',
    r'initiates?\s+coverage',
    r'reiterates?\s+(?:buy|hold|neutral|outperform|sell)',
    r'form\s+8\.3\s*-',
    # German
    r'(?:hob|hebt|senkte?|senken|bestätigt)\s+(?:das\s+)?(?:ziel|kursziel)',
    r'startet\s+mit',
    r'stuft\s+.*?\s+ein',
]

# Patterns indicating the news is ABOUT the intermediary's own corporate governance/earnings
SELF_CORPORATE_GOVERNANCE_MARKERS = [
    r'cda\s+di\s+equita',
    r'bilancio\s+(?:di\s+)?equita',
    r'bilancio\s+(?:di\s+)?mediobanca',
    r'cda\s+di\s+mediobanca',
    r'\butile\s+netto\s+di\b',
    r'\bdividendo\b',
    r'\bassemblea\s+degli\s+azionisti\b',
    r'\brisultati\s+(?:trimestrali|semestrali|annuali)\b',
    r'\bbilancio\s+(?:semestrale|annuale|consolidato)\b',
]

# ── LLM Priority Gate ──
# Only generate LLM prompts for news with priority_level <= this value.
# 1 = M&A/Earnings/Portfolio only, 2 = + Contract/Regulatory, 3 = all including Analyst/General
DEFAULT_MAX_LLM_PRIORITY = 2

# ── Taxonomy False-Positive Exclusion Patterns (v1.4) ──
# 1. Executive / Management Board employment contract renewals (not commercial client contracts)
BOARD_CONTRACT_EXCLUSION_PATTERNS = [
    r'\b(?:extends?|renews?|prolongs?|verlängert|rinnova|proroga)\b.*?\b(?:executive\s+board|management\s+board|supervisory\s+board|vorstandsvertrag|vorstandsmandat|vorstand|consigliere\s+delegato|amministratore\s+delegato|board\s+member)\b',
    r'\b(?:executive\s+board|management\s+board|vorstandsvertrag|vorstand)\b.*?\b(?:contract\s+extension|contract\s+renewal|vertragsverlängerung|verlängert)\b',
]

# 2. Calendar / Webcast / Conference Call invitations without financial figures
CALENDAR_INVITATION_PATTERNS = [
    r'\b(?:invitation\s+to|invites?\s+investors\s+to|webcast|conference\s+call|telefonkonferenz|einladung\s+zur|save\s+the\s+date|avviso\s+di\s+convocazione)\b.*?\b(?:interim\s+report|quarterly\s+report|q[1-4]\s+report|quartalsbericht|presentazione\s+dei\s+risultati)\b',
    r'\b(?:to\s+host|will\s+hold|wird\s+.*?\s+abhalten)\s+(?:a\s+)?(?:webcast|conference\s+call)\b',
]

# 3. Routine daily short-selling / PNC position disclosures (archive as GENERAL P3, not REGULATORY P2)
ROUTINE_SHORT_DISCLOSURE_PATTERNS = [
    r'\bposizioni\s+nette\s+corte\b',
    r'\b(?:riduce|aumenta|lima|taglia|incrementa)\s+(?:lo\s+)?short\b',
    r'\bposizioni\s+short\b',
    r'\bshort\s+interest\b',
]

# 4. Routine weekly treasury share buyback disclosures (downgrade to GENERAL P3 & flag in PromptBuilder)
ROUTINE_BUYBACK_PATTERNS = [
    r'comunicazione\s+in\s+merito\s+all[\'’]acquisto\s+di\s+azioni\s+proprie',
    r'informativa\s+sull[\'’]acquisto\s+di\s+azioni\s+proprie',
    r'riepilogo\s+settimanale\s+azioni\s+proprie',
    r'transaction[s]?\s+in\s+own\s+shares',
    r'\b(?:rilevate|acquistate|acquista)\s+[\d.,]+\s+azioni\s+(?:ordinarie\s+)?(?:proprie|nel\s+buyback)\b',
    r'\bporta\s+al\s+\d+[,.]\d+%\s+la\s+quota\s+di\s+azioni\s+proprie\b',
    r'\bnell[\'’]ambito\s+del\s+programma\s+di\s+(?:acquisto\s+di\s+azioni\s+proprie|buyback)\b',
    r'\bcontinues\s+buybacks?,\s+acquires\s+[\d.,]+\s+treasury\s+shares\b',
    r'\brepurchases\s+of\s+shares\s+by\b.*?\bduring\s+week\s+\d+\b',
]
