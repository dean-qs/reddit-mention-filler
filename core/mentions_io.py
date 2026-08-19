"""Read a Bulk Mentions export (xlsx/csv), fill it from fetched Reddit data,
and write the output workbook — all in memory (BytesIO in, BytesIO out).

Ported from process_batch.py, with the on-disk workdir cache and CLI plumbing
removed: a hosted multi-user app can't safely share a cache directory keyed
only by filename, so each run just does the read -> fetch -> fill pass fresh.
"""
import csv
import datetime
import io
from collections import defaultdict

import openpyxl
from openpyxl.cell import WriteOnlyCell

from .text_utils import ILLEGAL, EXCEL_CELL_LIMIT, EXTRA_COLS, build_extra_cols, categorize_status, md_to_text, norm, parse_score, row_type

REQUIRED_COLS = ("Date", "Url", "Full Text", "Author", "Title")


class BadExport(Exception):
    pass


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


def _read_csv_rows(file_bytes):
    csv.field_size_limit(50_000_000)
    text = file_bytes.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text)))


class ParsedExport:
    """Holds everything needed to both estimate (count URLs) and later fill/write."""

    def __init__(self, is_csv, urls, *, preamble=None, header=None, data_rows=None, cols=None,
                 header_row_1based=None):
        self.is_csv = is_csv
        self.urls = urls
        self.preamble = preamble
        self.header = header
        self.data_rows = data_rows
        self.cols = cols
        self.header_row_1based = header_row_1based


def parse_export(file_bytes, filename):
    """Parse an uploaded Bulk Mentions export and extract its URL list."""
    is_csv = filename.lower().endswith(".csv")
    if is_csv:
        rows = _read_csv_rows(file_bytes)
        hdr_i, cols = locate_header(rows)
        preamble, header, data_rows = rows[:hdr_i], rows[hdr_i], rows[hdr_i + 1:]
        urls = [r[cols["Url"]].strip() for r in data_rows if len(r) > cols["Url"] and r[cols["Url"]].strip()]
        if not urls:
            raise BadExport("No URLs found — is this a Bulk Mentions export?")
        return ParsedExport(True, urls, preamble=preamble, header=header, data_rows=data_rows, cols=cols)
    else:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True)
        ws = wb.worksheets[0]
        head = [list(row) for row in ws.iter_rows(min_row=1, max_row=30, values_only=True)]
        hdr_i, cols = locate_header(head)
        header_row_1based = hdr_i + 1
        urls = []
        for row in ws.iter_rows(min_row=header_row_1based + 1, values_only=True):
            v = row[cols["Url"]] if len(row) > cols["Url"] else None
            if v:
                urls.append(str(v).strip())
        wb.close()
        if not urls:
            raise BadExport("No URLs found — is this a Bulk Mentions export?")
        return ParsedExport(False, urls, cols=cols, header_row_1based=header_row_1based)


def read_header(file_bytes, filename):
    """Return the full header row (every column name, not just the required
    5) of a Bulk Mentions export — csv or xlsx, filled or not. Used to
    detect which columns an earlier module (in this pass or a prior one)
    already added, e.g. Sentiment Coding's "LLM Sentiment: <entity>" columns."""
    is_csv = filename.lower().endswith(".csv")
    if is_csv:
        rows = _read_csv_rows(file_bytes)
        hdr_i, _ = locate_header(rows)
        return [c if c is not None else "" for c in rows[hdr_i]]
    else:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True)
        ws = wb.worksheets[0]
        head = [list(row) for row in ws.iter_rows(min_row=1, max_row=30, values_only=True)]
        hdr_i, _ = locate_header(head)
        header = [c if c is not None else "" for c in head[hdr_i]]
        wb.close()
        return header


def iter_column_values(file_bytes, filename, column_name):
    """Read one named column's raw values across every data row of a Bulk
    Mentions export (csv or xlsx, filled or not) — for a lightweight preview/
    discovery step that doesn't need the full parse/fill machinery. Returns
    [] if the column isn't present in this export."""
    is_csv = filename.lower().endswith(".csv")
    if is_csv:
        rows = _read_csv_rows(file_bytes)
        hdr_i, _ = locate_header(rows)
        header = [c if c is not None else "" for c in rows[hdr_i]]
        if column_name not in header:
            return []
        idx = header.index(column_name)
        return [r[idx] if len(r) > idx else "" for r in rows[hdr_i + 1:]]
    else:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True)
        ws = wb.worksheets[0]
        head = [list(row) for row in ws.iter_rows(min_row=1, max_row=30, values_only=True)]
        hdr_i, _ = locate_header(head)
        header = [c if c is not None else "" for c in head[hdr_i]]
        if column_name not in header:
            wb.close()
            return []
        idx = header.index(column_name)
        values = [row[idx] if len(row) > idx else "" for row in ws.iter_rows(min_row=hdr_i + 2, values_only=True)]
        wb.close()
        return values


def write_rollup_sheet(wb, author_stats):
    ws = wb.create_sheet("Author Rollup")
    ws.append(["Author", "Total Mentions", "Posts", "Comments", "Total Score", "Avg Score", "First Date", "Last Date"])
    for author, a in sorted(author_stats.items(), key=lambda kv: kv[1]["count"], reverse=True):
        avg = round(a["score_sum"] / a["score_n"], 1) if a["score_n"] else ""
        ws.append([
            author, a["count"], a["posts"], a["comments"],
            a["score_sum"] if a["score_n"] else "",
            avg,
            a["first"].strftime("%Y-%m-%d") if a["first"] else "",
            a["last"].strftime("%Y-%m-%d") if a["last"] else "",
        ])


def fill_export(parsed: ParsedExport, fetch_results, file_bytes=None):
    """Fill Date/Full Text/metadata into the export and return (output_bytes, stats).

    fetch_results must be in the same order as parsed.urls (output of reddit_fetch.fetch_archive).
    file_bytes is required (and used) only for the xlsx path, to preserve the original workbook.
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

    out_buf = io.BytesIO()

    if parsed.is_csv:
        wb = openpyxl.Workbook(write_only=True)
        ws = wb.create_sheet("Sheet0")
        for row in parsed.preamble:
            ws.append(row)
        ws.append(list(parsed.header) + EXTRA_COLS)
        cols = parsed.cols
        i_date, i_url, i_ft = cols["Date"], cols["Url"], cols["Full Text"]
        for idx, row in enumerate(parsed.data_rows):
            if len(row) <= i_ft:
                ws.append(list(row) + [""] * len(EXTRA_COLS))
                continue
            url = row[i_url].strip()
            r = lookup(url) if url else None
            base = [ILLEGAL.sub("", c) if isinstance(c, str) else c for c in row]
            m = matched(r)
            if not m:
                if url:
                    unmatched.append((len(parsed.preamble) + 2 + idx, url, categorize_status(r)))
                ws.append(base + list(build_extra_cols(r).values()))
                continue
            out = list(base)
            out[i_ft] = render_text(m)
            if "Author" in cols and len(out) > cols["Author"]:
                out[cols["Author"]] = parse_author(m)
            if "Title" in cols and len(out) > cols["Title"]:
                out[cols["Title"]] = render_title(m)
            dt = parse_dt(m)
            if dt:
                c = WriteOnlyCell(ws, value=dt)
                c.number_format = "yyyy-mm-dd hh:mm:ss"
                out[i_date] = c
            ws.append(out + list(build_extra_cols(m).values()))
            update_rollup(m, dt)
            filled += 1
        write_rollup_sheet(wb, author_stats)
        wb.save(out_buf)
    else:
        if file_bytes is None:
            raise ValueError("file_bytes is required to fill an xlsx export")
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes))
        ws = wb.worksheets[0]
        cols = parsed.cols
        col_date, col_url, col_ft, col_auth, col_title = (cols[n] + 1 for n in REQUIRED_COLS)
        header_row_1based = parsed.header_row_1based
        extra_start = ws.max_column + 1
        for j, name_ in enumerate(EXTRA_COLS):
            ws.cell(row=header_row_1based, column=extra_start + j, value=name_)
        for row in range(header_row_1based + 1, ws.max_row + 1):
            url = ws.cell(row=row, column=col_url).value
            extras = build_extra_cols(None)
            if url:
                r = lookup(url)
                m = matched(r)
                if m:
                    dt = parse_dt(m)
                    if dt:
                        c = ws.cell(row=row, column=col_date, value=dt)
                        c.number_format = "yyyy-mm-dd hh:mm:ss"
                    ws.cell(row=row, column=col_ft, value=render_text(m))
                    ws.cell(row=row, column=col_auth, value=parse_author(m))
                    ws.cell(row=row, column=col_title, value=render_title(m))
                    extras = build_extra_cols(m)
                    update_rollup(m, dt)
                    filled += 1
                else:
                    unmatched.append((row, str(url), categorize_status(r)))
                    extras = build_extra_cols(r)
            for j, name_ in enumerate(EXTRA_COLS):
                ws.cell(row=row, column=extra_start + j, value=extras[name_])
        write_rollup_sheet(wb, author_stats)
        wb.save(out_buf)

    out_buf.seek(0)
    stats = {
        "filled": filled,
        "total": len(parsed.urls),
        "truncated": truncated,
        "unmatched": unmatched,
        "author_count": len(author_stats),
    }
    return out_buf, stats


def load_sheet_for_enrichment(file_bytes):
    """Open an already-filled xlsx (Mention Filler's output, or one the user
    uploaded pre-filled) for a module that reads existing columns and appends
    new ones. Returns (workbook, worksheet, header_row_1based, col_index)
    where col_index maps column name -> 1-based column number for every
    column currently in the sheet (original + anything appended so far).
    """
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes))
    ws = wb.worksheets[0]
    header_row_1based = None
    for r in range(1, min(31, ws.max_row + 1)):
        if ws.cell(row=r, column=1).value == "Query Id":
            header_row_1based = r
            break
    if header_row_1based is None:
        raise BadExport("Could not find the 'Query Id' header row — is this a filled Bulk Mentions export?")
    col_index = {}
    for c in range(1, ws.max_column + 1):
        name = ws.cell(row=header_row_1based, column=c).value
        if name:
            col_index[name] = c
    return wb, ws, header_row_1based, col_index


def ensure_columns(ws, header_row_1based, col_index, new_col_names):
    """Append any of new_col_names not already present; return the (possibly
    updated) col_index including their positions."""
    next_col = ws.max_column + 1
    for name in new_col_names:
        if name not in col_index:
            ws.cell(row=header_row_1based, column=next_col, value=name)
            col_index[name] = next_col
            next_col += 1
    return col_index


def iter_data_rows(ws, header_row_1based, col_index):
    """Yield (row_number, {column_name: value}) for every data row."""
    for row_num in range(header_row_1based + 1, ws.max_row + 1):
        row_dict = {name: ws.cell(row=row_num, column=c).value for name, c in col_index.items()}
        yield row_num, row_dict


def unmatched_csv_bytes(unmatched):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Row", "Url", "Reason"])
    w.writerows(unmatched)
    return buf.getvalue().encode("utf-8")
