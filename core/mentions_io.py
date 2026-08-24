"""Read a Bulk Mentions export (xlsx/csv), fill it from fetched Reddit data,
and write the output workbook — all in memory (bytes in, bytes out).

xlsx reads go through python-calamine and xlsx writes through xlsxwriter's
constant-memory mode — NOT openpyxl's normal mutable-cell mode, which is
catastrophically slow/memory-hungry for wide real-world exports. Measured on
a real 73,384-row x 193-column Brandwatch export: openpyxl's normal
`load_workbook()` took ~220s and ~5.7GB RAM; `read_only=True` cut memory to
~350MB but still took ~150-220s (the per-row XML parsing overhead barely
changes with fewer columns kept). python-calamine read the same file in
~12s at ~900MB, and xlsxwriter's constant_memory mode wrote it back out in
~17s — verified byte-for-byte round-trip identical. This is what was
actually causing "hangs, never offers a download" on larger real files:
Streamlit Community Cloud's containers have roughly 1GB of RAM, and even
locally, several minutes of silent blocking with no progress feedback reads
as a hang.

xlsxwriter needs `strings_to_urls=False` — by default it auto-detects
URL-looking strings and converts them to real hyperlinks, and Excel caps a
sheet at 65,530 hyperlinks; past that it silently drops the cell rather
than erroring. A Url column alone exceeds that on a file this size.
"""
import csv
import datetime
import io
import re
from collections import defaultdict

import python_calamine
import xlsxwriter

from .text_utils import ILLEGAL, EXCEL_CELL_LIMIT, EXTRA_COLS, build_extra_cols, categorize_status, md_to_text, norm, parse_score, row_type

REQUIRED_COLS = ("Date", "Url", "Full Text", "Author", "Title")


class BadExport(Exception):
    pass


def _read_xlsx_rows(file_bytes):
    """Every row as plain Python lists, via python-calamine — see module
    docstring for why not openpyxl."""
    wb = python_calamine.CalamineWorkbook.from_filelike(io.BytesIO(file_bytes))
    sheet = wb.get_sheet_by_index(0)
    return sheet.to_python()


def _read_csv_rows(file_bytes):
    csv.field_size_limit(50_000_000)
    text = file_bytes.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text)))


def locate_header(rows):
    """Find the 'Query Id' header row -> (index, col_indexes). Works with or without preamble."""
    hdr_i = next((i for i, r in enumerate(rows[:30]) if r and r[0] == "Query Id"), None)
    if hdr_i is None:
        raise BadExport("Could not find the 'Query Id' header row — is this a Bulk Mentions export?")
    header = [c if c is not None else "" for c in rows[hdr_i]]
    try:
        cols = {name: header.index(name) for name in REQUIRED_COLS}
    except ValueError as e:
        raise BadExport(f"Missing expected column: {e}")
    return hdr_i, cols


# ---------------------------------------------------------------------------
# Writing — every output path (fill, enrichment) funnels through here so the
# xlsxwriter setup (constant memory, no auto-hyperlinks, date formatting)
# only needs to be right in one place.
# ---------------------------------------------------------------------------

def _write_cell(ws, r, c, val, date_fmt):
    if val is None or val == "":
        return
    if isinstance(val, bool):
        ws.write_boolean(r, c, val)
    elif isinstance(val, (int, float)):
        ws.write_number(r, c, val)
    elif isinstance(val, (datetime.datetime, datetime.date)):
        ws.write_datetime(r, c, val, date_fmt)
    else:
        ws.write_string(r, c, ILLEGAL.sub("", str(val)))


def _write_sheet(ws, rows, date_fmt):
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            _write_cell(ws, r, c, val, date_fmt)


def _safe_sheet_name(name):
    return (re.sub(r'[\[\]:*?/\\]', "-", name) or "Sheet")[:31]


def _write_workbook_bytes(main_rows, extra_sheets=None):
    """main_rows: list of rows, row 0 is the header. extra_sheets: optional
    list of (sheet_name, rows) pairs, same shape."""
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True, "constant_memory": True, "strings_to_urls": False})
    date_fmt = wb.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})
    _write_sheet(wb.add_worksheet("Sheet0"), main_rows, date_fmt)
    for name, rows in (extra_sheets or []):
        _write_sheet(wb.add_worksheet(_safe_sheet_name(name)), rows, date_fmt)
    wb.close()
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Initial parse (upload -> URL list + full row data, ready for filling)
# ---------------------------------------------------------------------------

class ParsedExport:
    """Holds everything needed to both estimate (count URLs) and later fill/write."""

    def __init__(self, is_csv, urls, *, preamble, header, data_rows, cols):
        self.is_csv = is_csv
        self.urls = urls
        self.preamble = preamble
        self.header = header
        self.data_rows = data_rows
        self.cols = cols


def parse_export(file_bytes, filename):
    """Parse an uploaded Bulk Mentions export: locate the header, extract
    every data row (not just URLs) so fill_export never has to re-read the
    file from scratch."""
    is_csv = filename.lower().endswith(".csv")
    rows = _read_csv_rows(file_bytes) if is_csv else _read_xlsx_rows(file_bytes)
    hdr_i, cols = locate_header(rows)
    preamble, header, data_rows = rows[:hdr_i], rows[hdr_i], rows[hdr_i + 1:]
    header = [c if c is not None else "" for c in header]
    data_rows = [list(row) for row in data_rows]

    urls = [str(r[cols["Url"]]).strip() for r in data_rows if len(r) > cols["Url"] and r[cols["Url"]]]
    if not urls:
        raise BadExport("No URLs found — is this a Bulk Mentions export?")
    return ParsedExport(is_csv, urls, preamble=preamble, header=header, data_rows=data_rows, cols=cols)


def read_header(file_bytes, filename):
    """Return the full header row (every column name, not just the required
    5) of a Bulk Mentions export — csv or xlsx, filled or not. Used to
    detect which columns an earlier module (in this pass or a prior one)
    already added, e.g. Sentiment Coding's "LLM Sentiment: <entity>" columns."""
    is_csv = filename.lower().endswith(".csv")
    rows = _read_csv_rows(file_bytes) if is_csv else _read_xlsx_rows(file_bytes)
    hdr_i, _ = locate_header(rows)
    return [c if c is not None else "" for c in rows[hdr_i]]


def iter_column_values(file_bytes, filename, column_name):
    """Read one named column's raw values across every data row of a Bulk
    Mentions export (csv or xlsx, filled or not) — for a lightweight preview/
    discovery step that doesn't need the full parse/fill machinery. Returns
    [] if the column isn't present in this export."""
    is_csv = filename.lower().endswith(".csv")
    rows = _read_csv_rows(file_bytes) if is_csv else _read_xlsx_rows(file_bytes)
    hdr_i, _ = locate_header(rows)
    header = [c if c is not None else "" for c in rows[hdr_i]]
    if column_name not in header:
        return []
    idx = header.index(column_name)
    return [r[idx] if len(r) > idx else "" for r in rows[hdr_i + 1:]]


def _build_rollup_rows(author_stats):
    rows = [["Author", "Total Mentions", "Posts", "Comments", "Total Score", "Avg Score", "First Date", "Last Date"]]
    for author, a in sorted(author_stats.items(), key=lambda kv: kv[1]["count"], reverse=True):
        avg = round(a["score_sum"] / a["score_n"], 1) if a["score_n"] else ""
        rows.append([
            author, a["count"], a["posts"], a["comments"],
            a["score_sum"] if a["score_n"] else "",
            avg,
            a["first"].strftime("%Y-%m-%d") if a["first"] else "",
            a["last"].strftime("%Y-%m-%d") if a["last"] else "",
        ])
    return rows


def fill_export(parsed: ParsedExport, fetch_results):
    """Fill Date/Full Text/metadata into the export and return (output_bytes, stats).

    fetch_results must be in the same order as parsed.urls (output of reddit_fetch.fetch_archive).
    """
    by_url = {norm(r["url"]): r for r in fetch_results}
    filled = truncated = 0
    unmatched = []  # (row_num, url, reason)
    author_stats = defaultdict(lambda: {"count": 0, "posts": 0, "comments": 0,
                                         "score_sum": 0, "score_n": 0, "first": None, "last": None})

    def lookup(url):
        return by_url.get(norm(url))

    def matched(r):
        return r if r and (r.get("status") or "").startswith("OK") else None

    def render_text(r):
        nonlocal truncated
        text = md_to_text(ILLEGAL.sub("", r.get("text") or ""))
        if len(text) > EXCEL_CELL_LIMIT:
            text = text[:EXCEL_CELL_LIMIT - 15] + " [...TRUNCATED]"
            truncated += 1
        return text

    def render_title(r):
        nonlocal truncated
        post_title = md_to_text(ILLEGAL.sub("", r.get("post_title") or ""))
        if len(post_title) > EXCEL_CELL_LIMIT:
            post_title = post_title[:EXCEL_CELL_LIMIT - 15] + " [...TRUNCATED]"
            truncated += 1
        return post_title

    def parse_author(r):
        return md_to_text(ILLEGAL.sub("", r.get("author") or ""))

    def parse_dt(r):
        if not r.get("created"):
            return None
        return datetime.datetime.fromisoformat(r["created"].replace("Z", "+00:00")).replace(tzinfo=None)

    def update_rollup(r, dt):
        author = r.get("author") or "[unknown/deleted]"
        a = author_stats[author]
        a["count"] += 1
        t = row_type(r)
        if t == "Post":
            a["posts"] += 1
        elif t == "Comment":
            a["comments"] += 1
        score = parse_score(r.get("score"))
        if score is not None:
            a["score_sum"] += score
            a["score_n"] += 1
        if dt:
            if a["first"] is None or dt < a["first"]:
                a["first"] = dt
            if a["last"] is None or dt > a["last"]:
                a["last"] = dt

    cols = parsed.cols
    i_date, i_url, i_ft = cols["Date"], cols["Url"], cols["Full Text"]
    i_author, i_title = cols.get("Author"), cols.get("Title")

    out_rows = []
    for idx, row in enumerate(parsed.data_rows):
        if len(row) <= i_ft:
            out_rows.append(list(row) + [""] * len(EXTRA_COLS))
            continue
        url = str(row[i_url]).strip() if row[i_url] is not None else ""
        r = lookup(url) if url else None
        base = [ILLEGAL.sub("", c) if isinstance(c, str) else c for c in row]
        m = matched(r)
        if not m:
            if url:
                unmatched.append((len(parsed.preamble) + 2 + idx, url, categorize_status(r)))
            out_rows.append(base + list(build_extra_cols(r).values()))
            continue
        out = list(base)
        out[i_ft] = render_text(m)
        if i_author is not None and len(out) > i_author:
            out[i_author] = parse_author(m)
        if i_title is not None and len(out) > i_title:
            out[i_title] = render_title(m)
        dt = parse_dt(m)
        if dt:
            out[i_date] = dt
        out_rows.append(out + list(build_extra_cols(m).values()))
        update_rollup(m, dt)
        filled += 1

    header_row = list(parsed.header) + EXTRA_COLS
    main_rows = list(parsed.preamble) + [header_row] + out_rows
    rollup_rows = _build_rollup_rows(author_stats)
    output_bytes = _write_workbook_bytes(main_rows, extra_sheets=[("Author Rollup", rollup_rows)])

    stats = {
        "filled": filled,
        "total": len(parsed.urls),
        "truncated": truncated,
        "unmatched": unmatched,
        "author_count": len(author_stats),
    }
    return io.BytesIO(output_bytes), stats


# ---------------------------------------------------------------------------
# Post-fill enrichment (Sentiment / Geolocation / Theme Summary / Driver
# Analysis) — read the already-filled sheet, append columns, write once.
# ---------------------------------------------------------------------------

class Sheet:
    """An in-memory, already-filled Bulk Mentions sheet that a module reads
    from and appends columns to. Plain Python lists throughout — see module
    docstring for why this replaced a live openpyxl worksheet."""

    def __init__(self, preamble, header, rows):
        self.preamble = preamble
        self.header = list(header)
        self.rows = rows  # list of lists, mutable, parallel to header
        self._reindex()

    def _reindex(self):
        self.col_index = {name: i for i, name in enumerate(self.header) if name}

    def ensure_columns(self, names):
        added = [n for n in names if n not in self.col_index]
        for n in added:
            self.col_index[n] = len(self.header)
            self.header.append(n)
        if added:
            pad = [""] * len(added)
            for row in self.rows:
                row.extend(pad)

    def row_dict(self, row):
        return {name: (row[i] if i < len(row) else None) for name, i in self.col_index.items()}

    def set_by_index(self, row_idx, name, value):
        self.rows[row_idx][self.col_index[name]] = value

    def iter_rows(self):
        """Yield (row_index, {column_name: value}) for every data row."""
        for i, row in enumerate(self.rows):
            yield i, self.row_dict(row)

    def to_bytes(self, extra_sheets=None):
        """extra_sheets: optional list of (sheet_name, header_row, data_rows)."""
        main_rows = list(self.preamble) + [self.header] + self.rows
        extras = [(name, [header] + list(data)) for name, header, data in (extra_sheets or [])]
        return _write_workbook_bytes(main_rows, extra_sheets=extras)


def load_sheet_for_enrichment(file_bytes):
    """Load an already-filled xlsx (Mention Filler's output, or one the user
    uploaded pre-filled) into a Sheet for a module to read/append columns to."""
    rows = _read_xlsx_rows(file_bytes)
    hdr_i = next((i for i, r in enumerate(rows[:30]) if r and r[0] == "Query Id"), None)
    if hdr_i is None:
        raise BadExport("Could not find the 'Query Id' header row — is this a filled Bulk Mentions export?")
    preamble, header, data_rows = rows[:hdr_i], rows[hdr_i], rows[hdr_i + 1:]
    header = [c if c is not None else "" for c in header]
    data_rows = [list(row) for row in data_rows]
    return Sheet(preamble, header, data_rows)


def unmatched_csv_bytes(unmatched):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Row", "Url", "Reason"])
    w.writerows(unmatched)
    return buf.getvalue().encode("utf-8")
