"""Text cleaning and per-row metadata helpers shared by every module.

Ported from the original process_batch.py — pure functions, no I/O.
"""
import re

ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EXCEL_CELL_LIMIT = 32767
URL_RE = re.compile(r'https?://[^\s\)\]"\']+')

# The Bulk Mentions export's own placeholder text for withheld Reddit content —
# used to tell "not filled yet" apart from "filled, but legitimately empty".
LICENSING_PLACEHOLDER = "licensing restrictions"

_SUBREDDIT_RE = re.compile(r'reddit\.com/r/([A-Za-z0-9_]+)/', re.I)


def subreddit_from_url(url):
    m = _SUBREDDIT_RE.search(str(url or ""))
    return m.group(1) if m else ""


def looks_unfilled(text):
    """True if `text` is empty or still the export's withheld-content placeholder."""
    t = (text or "").strip()
    return not t or LICENSING_PLACEHOLDER in t.lower()


def rough_token_estimate(text):
    """Cheap word-count-based approximation — good enough for a pre-run estimate,
    not for billing (real cost is computed from actual API usage after running)."""
    return max(1, int(len((text or "").split()) * 1.3))


def norm(url):
    return str(url).strip().split("?")[0].rstrip("/")


def md_to_text(md):
    """Render Reddit markdown to the plain text a reader sees on the page."""
    t = md
    t = (t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
          .replace("&#x200B;", "").replace("&nbsp;", " ").replace("​", ""))
    t = re.sub(r"(?m)^[ \t]*(?:>[ \t]?)+", "", t)                      # blockquote markers
    t = re.sub(r"(?m)^#{1,6}[ \t]+", "", t)                            # headings
    t = re.sub(r"!?\[([^\]]*)\]\(([^)]+)\)", r"\1", t)                 # [text](url) -> text
    t = re.sub(r"\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*", r"\1", t, flags=re.S)  # ***bold italic***
    t = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"\1", t, flags=re.S)    # **bold**
    t = re.sub(r"\*(?=\S)([^*\n]+?)(?<=\S)\*", r"\1", t)               # *italic*
    t = re.sub(r"__(?=\S)(.+?)(?<=\S)__", r"\1", t, flags=re.S)        # __bold__ (single _ left alone)
    t = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"\1", t, flags=re.S)        # ~~strikethrough~~
    t = re.sub(r"`([^`\n]+)`", r"\1", t)                               # `code`
    t = re.sub(r"\^\((.*?)\)", r"\1", t)                               # ^(superscript)
    t = re.sub(r"\\([\\*_~^`\[\]()#>!.-])", r"\1", t)                  # escaped \* \_ etc.
    t = re.sub(r"[ \t]+(?=\n)", "", t)                                 # markdown hard-break spaces
    return t


def extract_links(raw_text):
    """Pull linked URLs out of the raw (pre-markdown-stripped) text."""
    if not raw_text:
        return [], []
    found = [u.rstrip('.,;:!?)') for u in URL_RE.findall(raw_text)]
    seen, uniq = set(), []
    for u in found:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    domains = []
    for u in uniq:
        m = re.match(r'https?://(?:www\.)?([^/]+)', u)
        if m:
            domains.append(m.group(1))
    return uniq, sorted(set(domains))


def parse_score(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return int(str(v).replace(",", "").strip())
    except ValueError:
        return None


def row_type(r):
    cid = (r or {}).get("comment_id") or ""
    if cid.startswith("t1_"):
        return "Comment"
    if cid.startswith("t3_"):
        return "Post"
    return ""


def categorize_status(r):
    """Turn the raw fetch status into one clean reason code (no live-fallback status in v0)."""
    status = (r or {}).get("status", "") or ""
    if status.startswith("OK"):
        return "OK"
    if status == "UNPARSEABLE_URL":
        return "UNPARSEABLE_URL"
    if status == "NOT_IN_ARCHIVE":
        return "NOT_IN_ARCHIVE"
    return status or "NO_RESULT"


EXTRA_COLS = [
    "Score", "Type", "Edited",
    "Linked URLs", "Link Domain", "Fetch Status",
]


def build_extra_cols(r):
    if r is None:
        return {c: ("NO_RESULT" if c == "Fetch Status" else "") for c in EXTRA_COLS}
    urls, domains = extract_links(r.get("text") or "")
    score = parse_score(r.get("score"))
    return {
        "Score": score if score is not None else 0,
        "Type": row_type(r),
        "Edited": "Yes" if r.get("edited") else "No",
        "Linked URLs": "; ".join(urls),
        "Link Domain": "; ".join(domains),
        "Fetch Status": categorize_status(r),
    }
