"""Bulk-fetch Reddit comment/post text from the Arctic Shift archive.

Pure-Python port of reddit_archive_grabber.js's batching and retry/backoff
behavior, no Node/browser involved — but fetched CONCURRENTLY (a bounded
thread pool) instead of one 100-id chunk at a time with a blocking sleep
between each. The original sequential version is safe in a terminal script
(no time limit on how long a CLI run takes), but on a Streamlit-hosted app a
large export's fetch could run for 20-30+ minutes of continuous sequential
HTTP calls tied to one browser session — long enough for a laptop sleep,
network blip, or the platform's own session recycling to drop the
connection mid-run, which reads to the user as "it finished but there's no
download button" (the page just reloads to its pre-run state). Concurrency
cuts that window to a couple of minutes for even a 70k+ row export. The
per-chunk retry/backoff/bisect-on-failure behavior is unchanged; a bounded
worker pool (not one request at a time) is the actual throttle now instead
of a fixed inter-request sleep.
"""
import concurrent.futures
import re
import time

import requests

API = "https://arctic-shift.photon-reddit.com/api"
BATCH = 100
FETCH_WORKERS = 8
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


def _fetch_chunk(session, endpoint, chunk, on_log=None):
    """Fetch one chunk of ids, bisecting and retrying on failure. Returns a
    dict local to this call — thread-safe by construction, no shared state
    to lock, since callers merge each future's own dict after it completes."""
    found = {}
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
            return found
        mid = -(-len(chunk) // 2)
        if on_log:
            on_log(f"  chunk of {len(chunk)} failed ({e}) — bisecting")
        found.update(_fetch_chunk(session, endpoint, chunk[:mid], on_log))
        found.update(_fetch_chunk(session, endpoint, chunk[mid:], on_log))
    return found


def _chunks(ids):
    return [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]


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

    comments, posts = {}, {}
    dest = {"comments": comments, "posts": posts}
    tasks = [("comments", c) for c in _chunks(comment_ids)] + [("posts", c) for c in _chunks(all_post_ids)]
    total_ids = len(comment_ids) + len(all_post_ids)
    done_total = 0

    with requests.Session() as session:
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=FETCH_WORKERS)
        session.mount("https://", adapter)
        if tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
                futures = {pool.submit(_fetch_chunk, session, endpoint, chunk, on_log): (endpoint, chunk)
                           for endpoint, chunk in tasks}
                for fut in concurrent.futures.as_completed(futures):
                    endpoint, chunk = futures[fut]
                    dest[endpoint].update(fut.result())
                    done_total += len(chunk)
                    if on_progress:
                        # done/total are combined across comments+posts so the caller's
                        # progress bar advances monotonically even though both endpoints'
                        # chunks now complete interleaved, not one endpoint fully before
                        # the other.
                        on_progress(endpoint, min(done_total, total_ids), total_ids,
                                     len(comments) + len(posts))

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
