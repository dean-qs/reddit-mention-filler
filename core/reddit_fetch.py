"""Bulk-fetch Reddit comment/post text from the Arctic Shift archive.

Pure-Python port of reddit_archive_grabber.js — same batching, retry/backoff,
and bisect-on-persistent-failure behavior, no Node/browser involved. This is
the free, keyless, ~1,400-URL/minute path that covers the large majority of
any Bulk Mentions export (per the original README, archive misses are
typically 0-2 per 1,000 URLs).
"""
import re
import time

import requests

API = "https://arctic-shift.photon-reddit.com/api"
BATCH = 100
DELAY_S = 1.0
UA = "quadrant-reddit-mention-filler/1.0"

_COMMENT_RE = re.compile(r"/comments/([a-z0-9]+)/[^/]+/([a-z0-9]+)$", re.I)
_POST_RE = re.compile(r"/comments/([a-z0-9]+)(/[^/]+)?$", re.I)


def classify(url):
    """Classify a Reddit URL as a comment permalink or a post link."""
    clean = url.split("?")[0].rstrip("/")
    m = _COMMENT_RE.search(clean)
    if m:
        return {"kind": "comment", "postId": "t3_" + m.group(1), "id": "t1_" + m.group(2)}
    m = _POST_RE.search(clean)
    if m:
        return {"kind": "post", "postId": "t3_" + m.group(1), "id": "t3_" + m.group(1)}
    return {"kind": "unknown", "postId": None, "id": None}


def _fetch_json(session, url, attempt=1):
    try:
        resp = session.get(url, headers={"User-Agent": UA}, timeout=30)
    except requests.RequestException as e:
        if attempt > 6:
            raise
        wait = min(60, 2 * 2 ** attempt)
        time.sleep(wait)
        return _fetch_json(session, url, attempt + 1)
    if resp.status_code in (429, 422) or resp.status_code >= 500:
        if attempt > 6:
            raise RuntimeError(f"HTTP {resp.status_code} after {attempt} attempts")
        wait = min(60, 2 * 2 ** attempt)
        time.sleep(wait)
        return _fetch_json(session, url, attempt + 1)
    resp.raise_for_status()
    return resp.json()


def _fetch_chunk(session, endpoint, chunk, found, on_log=None):
    try:
        ids_param = ",".join(chunk)
        data = _fetch_json(session, f"{API}/{endpoint}/ids?ids={ids_param}")
        prefix = "t1_" if endpoint == "comments" else "t3_"
        for item in data.get("data") or []:
            found[prefix + item["id"]] = item
    except Exception as e:
        if len(chunk) == 1:
            if on_log:
                on_log(f"  giving up on {chunk[0]}: {e}")
            return
        mid = -(-len(chunk) // 2)
        if on_log:
            on_log(f"  chunk of {len(chunk)} failed ({e}) — bisecting")
        _fetch_chunk(session, endpoint, chunk[:mid], found, on_log)
        time.sleep(DELAY_S)
        _fetch_chunk(session, endpoint, chunk[mid:], found, on_log)


def _bulk_fetch(session, endpoint, ids, label, on_progress=None, on_log=None):
    found = {}
    for i in range(0, len(ids), BATCH):
        _fetch_chunk(session, endpoint, ids[i:i + BATCH], found, on_log)
        if on_progress:
            on_progress(label, min(i + BATCH, len(ids)), len(ids), len(found))
        if i + BATCH < len(ids):
            time.sleep(DELAY_S)
    return found


def fetch_archive(urls, on_progress=None, on_log=None):
    """Fetch every URL from the Arctic Shift archive.

    Returns a list of dicts (same order as `urls`), one per URL:
      {url, comment_id, author, created, score, post_title, edited, status, text}
    status is one of: OK, OK_POST, NOT_IN_ARCHIVE, UNPARSEABLE_URL
    """
    items = [{"url": u, **classify(u)} for u in urls]
    comment_ids = list(dict.fromkeys(x["id"] for x in items if x["kind"] == "comment"))
    post_only_ids = list(dict.fromkeys(x["id"] for x in items if x["kind"] == "post"))
    title_ids = list(dict.fromkeys(x["postId"] for x in items if x["kind"] == "comment"))
    all_post_ids = list(dict.fromkeys(post_only_ids + title_ids))

    with requests.Session() as session:
        comments = _bulk_fetch(session, "comments", comment_ids, "comments", on_progress, on_log)
        posts = _bulk_fetch(session, "posts", all_post_ids, "posts", on_progress, on_log)

    results = []
    for it in items:
        row = {"url": it["url"], "comment_id": it["id"], "author": None, "created": None,
               "score": None, "post_title": None, "edited": False, "status": "", "text": ""}
        if it["kind"] == "unknown":
            row["status"] = "UNPARSEABLE_URL"
        elif it["kind"] == "comment":
            c = comments.get(it["id"])
            p = posts.get(it["postId"])
            if p:
                row["post_title"] = p.get("title")
            if c:
                row["author"] = c.get("author")
                row["created"] = _iso(c.get("created_utc"))
                row["score"] = c.get("score")
                row["text"] = c.get("body")
                row["edited"] = bool(c.get("edited"))
                row["status"] = "OK"
            else:
                row["status"] = "NOT_IN_ARCHIVE"
        else:
            p = posts.get(it["id"])
            if p:
                row["author"] = p.get("author")
                row["created"] = _iso(p.get("created_utc"))
                row["score"] = p.get("score")
                row["post_title"] = p.get("title")
                row["text"] = p.get("selftext") or p.get("url") or ""
                row["edited"] = bool(p.get("edited"))
                row["status"] = "OK_POST"
            else:
                row["status"] = "NOT_IN_ARCHIVE"
        results.append(row)
    return results


def _iso(created_utc):
    if created_utc is None:
        return None
    import datetime
    return datetime.datetime.fromtimestamp(created_utc, tz=datetime.timezone.utc).isoformat()
