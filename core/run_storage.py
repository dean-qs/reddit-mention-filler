"""Server-side backup of run outputs, independent of any one browser
session's st.session_state — so a dropped connection (network blip,
Streamlit Cloud's own reconnect timing out, laptop sleep) after a run
finishes doesn't leave the result unrecoverable. Local disk on the running
container: survives a page reload or a brand-new browser session (unlike
session_state, which is per-session), but does NOT survive an app
reboot/redeploy (the container filesystem is ephemeral) — a short-term
safety net, not permanent archival. Download the real result promptly;
treat this as a rescue path, not storage.

Checkpointed, not just saved once at the very end: app.py calls save_run()
with the SAME run_id after every module completes (not only at the very
end), so if the connection drops mid-pipeline, whatever finished before
that point is still recoverable — not just a fully-successful run. Each
call overwrites the previous checkpoint's files (same run_id -> same
directory) rather than keeping one copy per module, so disk usage stays
bounded to roughly one output's worth per run, not N.
"""
import datetime
import json
import re
import time
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
MAX_RUNS_KEPT = 20


def new_run_id():
    # Directory names sort lexicographically for list_runs()'s most-recent-
    # first ordering and _prune_old_runs()'s retention cutoff, so the id
    # must be monotonically increasing as a STRING. A human-readable prefix
    # plus a modulo'd sub-second suffix isn't safe for that (mod of a
    # monotonic counter isn't itself monotonic) — the full zero-padded
    # nanosecond counter is what actually guarantees correct ordering, with
    # the readable prefix kept only for browsability.
    ns = time.time_ns()
    human = time.strftime("%Y%m%d-%H%M%S", time.localtime(ns / 1e9))
    return f"{human}-{ns:020d}"


def _safe_name(name):
    return re.sub(r'[^A-Za-z0-9._ -]', "_", name)[:150] or "file"


def save_run(run_id, filename, output_bytes, extra_files=None, summary_lines=None):
    """Best-effort checkpoint — callers should wrap this in try/except and
    never let a storage failure break the actual run."""
    RUNS_DIR.mkdir(exist_ok=True)
    run_dir = RUNS_DIR / _safe_name(run_id)
    run_dir.mkdir(exist_ok=True)
    for f in run_dir.iterdir():
        if f.is_file():
            f.unlink()
    (run_dir / _safe_name(filename)).write_bytes(output_bytes)
    for extra_name, extra_bytes in (extra_files or {}).items():
        (run_dir / _safe_name(extra_name)).write_bytes(extra_bytes)
    manifest = {
        "filename": filename,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "summary_lines": summary_lines or [],
        "extra_files": list((extra_files or {}).keys()),
    }
    (run_dir / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _prune_old_runs()
    return run_dir


def _prune_old_runs():
    if not RUNS_DIR.exists():
        return
    run_dirs = sorted((d for d in RUNS_DIR.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    for old_dir in run_dirs[MAX_RUNS_KEPT:]:
        for f in old_dir.iterdir():
            f.unlink()
        old_dir.rmdir()


def list_runs():
    """Most-recent-first: [{"dir", "filename", "saved_at", "summary_lines", "extra_files"}]."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for run_dir in sorted((d for d in RUNS_DIR.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True):
        manifest_path = run_dir / "_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        runs.append({"dir": run_dir, **manifest})
    return runs


def load_file(run_dir, filename):
    path = Path(run_dir) / _safe_name(filename)
    return path.read_bytes() if path.exists() else None
