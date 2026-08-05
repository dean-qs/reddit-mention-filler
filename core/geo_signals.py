"""Lightweight regex/lookup signals fed into the geolocation prompt as
*evidence*, not a verdict — the LLM weighs these alongside the actual text.
Deliberately small and easy to extend; false positives are fine here since
the model treats these as hints, not ground truth.
"""
import re

# Specific British/American spelling-variant word pairs — an explicit list rather
# than a suffix pattern, since a broad "-ise"/"-our" regex also matches ordinary
# words (surprise, wise, tour) that aren't dialect markers at all and would fire
# on nearly every row, making the signal useless.
_BRITISH_WORDS = re.compile(
    r"\b(colour|colours|favourite|neighbour|neighbours|humour|honour|behaviour|"
    r"organise|organised|organising|organisation|recognise|recognised|realise|realised|"
    r"apologise|criticise|analyse|analysed|"
    r"centre|centres|theatre|litre|litres|metre|metres|"
    r"defence|licence|programme|travelled|travelling|cancelled|"
    r"jewellery|aluminium|mould|grey|tyre|tyres|pyjamas|kerb)\b",
    re.I,
)
_AMERICAN_WORDS = re.compile(
    r"\b(color|colors|favorite|neighbor|neighbors|humor|honor|behavior|"
    r"organize|organized|organizing|organization|recognize|recognized|realize|realized|"
    r"apologize|criticize|analyze|analyzed|"
    r"center|centers|theater|liter|liters|meter|meters|"
    r"defense|license|program|traveled|traveling|canceled|"
    r"jewelry|aluminum|mold|gray|tire|tires|pajamas|curb)\b",
    re.I,
)

# Numeric formatting conventions — distinctive enough (multi-digit grouping +
# decimal suffix) to avoid firing on an ordinary lone decimal like "1.234".
_EU_STYLE_NUMBER = re.compile(r"\b\d{1,3}(?:\.\d{3})+,\d{1,2}\b")   # "1.234,56" — most of Europe/Latin America
_US_STYLE_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+\.\d{1,2}\b")  # "1,234.56" — US/UK

# Spanish dialect tells — Iberian (vosotros forms, Spain-specific vocabulary) vs.
# the LatAm counterparts. "ustedes" alone is weak evidence (Spain uses it too, just
# formally-only) so it's included but shouldn't be read as strong on its own.
_IBERIAN_SPANISH = re.compile(
    r"\b(vosotros|vosotras|vuestro|vuestra|vuestros|vuestras|"
    r"ordenador|ordenadores|m[oó]vil|m[oó]viles|coche|coches)\b|"
    r"\w+[áé]is\b",  # vosotros-form verb endings (habláis, coméis) — accented, so it
                      # doesn't also catch unrelated words (e.g. Portuguese "mais")
    re.I,
)
_LATAM_SPANISH = re.compile(
    r"\b(ustedes|computadora|computadoras|computador|celular|celulares|"
    r"carro|carros|platicar)\b",
    re.I,
)

# (human-readable label, regex) — other British vs. American English tells.
_MARKERS = [
    ("British spelling (colour/organise/centre/etc.)", _BRITISH_WORDS),
    ("American spelling (color/organize/center/etc.)", _AMERICAN_WORDS),
    ("whilst / amongst", re.compile(r"\b(whilst|amongst)\b", re.I)),
    ("British transit/daily-life terms (queue, postcode, tube, petrol, boot/bonnet)",
     re.compile(r"\b(queue|postcode|petrol|tube station|boot of the car|bonnet)\b", re.I)),
    ("mum (vs. mom)", re.compile(r"\bmum\b", re.I)),
    ("British currency (£, quid, pence)", re.compile(r"(£|\bquid\b|\bpence\b)", re.I)),
    ("American currency/units ($, miles, Fahrenheit)", re.compile(r"(\$\d|\bmiles?\b|\bfahrenheit\b)", re.I)),
    ("mom (vs. mum)", re.compile(r"\bmom\b", re.I)),
    ("American transit/daily-life terms (sidewalk, trunk of the car, gas station, zip code)",
     re.compile(r"\b(sidewalk|trunk of the car|gas station|zip code)\b", re.I)),
    ("European/Latin-American-style decimal formatting (1.234,56)", _EU_STYLE_NUMBER),
    ("US/UK-style decimal formatting (1,234.56)", _US_STYLE_NUMBER),
    ("Iberian Spanish (vosotros/vuestro, ordenador, móvil, coche, -áis/-éis verb forms)", _IBERIAN_SPANISH),
    ("Latin American Spanish (ustedes, computadora, celular, carro)", _LATAM_SPANISH),
]

# Small, easily-extended set of subreddits that are strong geography hints on their own.
_SUBREDDIT_HINTS = {
    "casualuk": "United Kingdom", "askuk": "United Kingdom", "unitedkingdom": "United Kingdom",
    "ukpolitics": "United Kingdom", "london": "United Kingdom",
    "ireland": "Ireland", "askireland": "Ireland",
    "canada": "Canada", "onguardforthee": "Canada", "personalfinancecanada": "Canada", "toronto": "Canada",
    "australia": "Australia", "askanaustralian": "Australia", "melbourne": "Australia", "sydney": "Australia",
    "newzealand": "New Zealand",
    "india": "India", "indiaspeaks": "India", "bangalore": "India", "mumbai": "India",
    "askanamerican": "United States",
    "nyc": "United States (New York)", "losangeles": "United States (California)",
    "chicago": "United States (Illinois)", "texas": "United States (Texas)",
}


def detect_signals(subreddit, title, text):
    """Return a short list of plain-English evidence strings, or [] if nothing fired."""
    combined = f"{title or ''} {text or ''}"
    signals = [label for label, rx in _MARKERS if rx.search(combined)]
    hint = _SUBREDDIT_HINTS.get((subreddit or "").lower())
    if hint:
        signals.append(f"subreddit r/{subreddit} is commonly associated with {hint}")
    return signals
