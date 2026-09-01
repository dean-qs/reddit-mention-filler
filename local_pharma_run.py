"""One-off LOCAL run: Mention Filler -> pharma/biotech Sentiment Coding ->
12-theme Theme Summary, for a file too large for the hosted Streamlit app's
~1GB memory ceiling right now. Reuses this repo's own core/modules code
directly (no Streamlit server needed) so behavior matches the app exactly.

Usage:
    python local_pharma_run.py "path/to/export.xlsx"

Requires OPENAI_API_KEY set as an environment variable (never pasted into
chat or committed) before running.

Bypasses two hosted-app-only safety rails that don't make sense for a
deliberate, supervised local run on your own machine/API key: the row-count
cost cap (core.cost_caps, default 20,000 rows/run) and the per-module
concurrency limit (tuned for the shared hosted app, not a one-off big job).
"""
import concurrent.futures
import sys
import time
from pathlib import Path

from openai import OpenAI

from core.mentions_io import BadExport, fill_export, load_sheet_for_enrichment, parse_export
from core.reddit_fetch import fetch_archive
from core.text_utils import looks_unfilled
import core.llm_client as llm_client
from core.llm_client import call_json
import core.cost_caps as cost_caps
import modules.theme_summary as theme_summary_mod
from modules.theme_summary import ThemeSummaryModule

SENTIMENT_WORKERS = 60
INPUT_PRICE = 0.15 / 1_000_000   # gpt-4o-mini
OUTPUT_PRICE = 0.60 / 1_000_000

PHARMA_SYSTEM_PROMPT = (
    "You are analyzing Reddit mentions for a pharmaceutical/biotech social-listening study. "
    "First, determine whether any SPECIFIC pharmaceutical or biotechnology COMPANY is mentioned "
    "in the text (e.g. Pfizer, Genentech, Moderna, Roche, AbbVie, Amgen, Novartis, Merck, Eli "
    "Lilly, Sanofi, GSK, Bristol Myers Squibb, Regeneron, Gilead, Biogen, and similar — a branded "
    "drug counts if its maker is identifiable from context or general knowledge).\n\n"
    "If one or more such companies are mentioned, put the SINGLE most prominently discussed one "
    "in 'Detected Entity' and classify sentiment specifically TOWARD THAT COMPANY.\n\n"
    "If no specific pharma/biotech company is mentioned, set 'Detected Entity' to "
    "'Healthcare System/Patient Access/Biopharma Industry' and classify sentiment toward the "
    "healthcare system, patient access to medications/care, or the biopharma industry generally, "
    "as expressed in the mention.\n\n"
    "Respond with 'Detected Entity' (a company name, or the fallback topic above exactly as "
    "given), 'LLM Sentiment' (exactly one of Positive, Neutral, Negative), and "
    "'LLM Sentiment Rationale' (a short clause, under 15 words)."
)

PHARMA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "pharma_sentiment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "Detected Entity": {"type": "string"},
                "LLM Sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
                "LLM Sentiment Rationale": {"type": "string"},
            },
            "required": ["Detected Entity", "LLM Sentiment", "LLM Sentiment Rationale"],
            "additionalProperties": False,
        },
    },
}


def run_pharma_sentiment(client, file_bytes, progress_every=2000):
    sheet = load_sheet_for_enrichment(file_bytes)
    sheet.ensure_columns(["Detected Entity", "LLM Sentiment", "LLM Sentiment Rationale"])
    if "Full Text" not in sheet.col_index:
        raise BadExport("No 'Full Text' column — run Mention Filler first.")

    all_rows = list(sheet.iter_rows())
    eligible = [(row_num, row) for row_num, row in all_rows if not looks_unfilled(row.get("Full Text"))]
    print(f"  {len(eligible):,} eligible rows of {len(all_rows):,} total", flush=True)

    def work(row_num, row):
        text = str(row.get("Full Text") or "")[:4000]
        title = row.get("Title") or ""
        user_message = f"Post Title: {title}\nFull Text: {text}"
        content, usage = call_json(client, PHARMA_SCHEMA, PHARMA_SYSTEM_PROMPT, user_message)
        return row_num, content, usage

    if not eligible:
        return sheet.to_bytes(), {"n_failed": 0, "cost": 0.0}

    results_by_row = {}
    n_failed = 0
    done = 0
    usage_totals = {"input": 0, "output": 0}
    t0 = time.time()

    first_row_num, first_row = eligible[0]
    try:
        _, first_content, first_usage = work(first_row_num, first_row)
    except Exception as e:
        raise RuntimeError(f"First sentiment test call failed — check API key/billing: {e}") from e
    results_by_row[first_row_num] = first_content
    usage_totals["input"] += first_usage.prompt_tokens or 0
    usage_totals["output"] += first_usage.completion_tokens or 0
    done = 1

    remaining = eligible[1:]
    with concurrent.futures.ThreadPoolExecutor(max_workers=SENTIMENT_WORKERS) as pool:
        futures = [pool.submit(work, rn, row) for rn, row in remaining]
        for fut in concurrent.futures.as_completed(futures):
            try:
                row_num, content, usage = fut.result()
                results_by_row[row_num] = content
                usage_totals["input"] += usage.prompt_tokens or 0
                usage_totals["output"] += usage.completion_tokens or 0
            except Exception:
                n_failed += 1
            done += 1
            if done % progress_every == 0 or done == len(eligible):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(eligible) - done) / rate / 60 if rate else 0
                print(f"  sentiment {done:,}/{len(eligible):,} ({n_failed} failed) — "
                      f"{rate:.1f} rows/s, ETA {eta:.1f} min", flush=True)

    eligible_row_nums = {rn for rn, _ in eligible}
    for row_num, row in all_rows:
        result = results_by_row.get(row_num)
        if result is None:
            entity, sentiment, rationale = ("ERROR", "ERROR", "") if row_num in eligible_row_nums else ("", "", "")
        else:
            entity = result.get("Detected Entity", "")
            sentiment = result.get("LLM Sentiment", "")
            rationale = result.get("LLM Sentiment Rationale", "")
        sheet.set_by_index(row_num, "Detected Entity", entity)
        sheet.set_by_index(row_num, "LLM Sentiment", sentiment)
        sheet.set_by_index(row_num, "LLM Sentiment Rationale", rationale)

    cost = usage_totals["input"] * INPUT_PRICE + usage_totals["output"] * OUTPUT_PRICE
    print(f"  sentiment done: {len(eligible) - n_failed:,} succeeded, {n_failed:,} failed, cost ${cost:.2f}", flush=True)

    # Build the breakdown sheet defensively: `sheet` already has every row's
    # sentiment written in via set_by_index above (that's the expensive part,
    # already paid for and done) — if anything below this point throws, we
    # must still save that, not lose ~$10/an hour of real API work to a bug
    # in a tally. iter_rows() is called fresh here (not from the `all_rows`
    # snapshot above) since that snapshot predates the set_by_index writes.
    extra_sheets = None
    try:
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for _, row in sheet.iter_rows():
            s = row.get("LLM Sentiment")
            if s in counts:
                counts[s] += 1
        total = sum(counts.values())
        net = round((counts["Positive"] - counts["Negative"]) / total, 3) if total else ""
        extra_sheets = [
            ("Sentiment Breakdown", ["Positive", "Negative", "Neutral", "Total", "Net Sentiment"],
             [[counts["Positive"], counts["Negative"], counts["Neutral"], total, net]]),
        ]
    except Exception as e:
        print(f"  WARNING: breakdown tally failed ({e!r}) — saving per-row results without it", flush=True)

    out_bytes = sheet.to_bytes(extra_sheets=extra_sheets)
    return out_bytes, {"n_failed": n_failed, "cost": cost}


def main():
    args = sys.argv[1:]
    resume_from_filled = "--resume-from-filled" in args
    resume_from_sentiment = "--resume-from-sentiment" in args
    if resume_from_filled:
        args.remove("--resume-from-filled")
    if resume_from_sentiment:
        args.remove("--resume-from-sentiment")
    if not args or (resume_from_filled and resume_from_sentiment):
        print("Usage: python local_pharma_run.py [--resume-from-filled | --resume-from-sentiment] <path-to-export.xlsx>\n"
              "  --resume-from-filled: given file is Mention Filler's output — skip straight to\n"
              "  sentiment + theme summary (no need to redo the archive fetch).\n"
              "  --resume-from-sentiment: given file already has sentiment columns — skip straight\n"
              "  to theme summary (no need to redo the archive fetch OR the sentiment LLM pass).")
        sys.exit(1)
    path = Path(args[0])
    filename = path.name

    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        sys.exit("No OPENAI_API_KEY found. Either set it as an environment variable in this "
                 "terminal, or copy .streamlit/secrets.toml.example to .streamlit/secrets.toml "
                 "and paste your key in there (that file is gitignored — never committed, never "
                 "shared in chat).")
    client = OpenAI(api_key=api_key)
    llm_client._client = client  # pre-seed so theme_summary's internal get_client() reuses this client

    # Local-run-only overrides — the hosted app's row cap and worker counts are
    # tuned for the shared, memory-constrained Streamlit Cloud container, not a
    # deliberate one-off big job on your own machine/API key.
    # NOTE: theme_summary.py did `from core.cost_caps import max_rows_per_run`,
    # which bound its OWN copy of that name at import time — patching the
    # cost_caps module's attribute afterward doesn't reach that copy. Patch
    # the name inside theme_summary_mod's own namespace instead.
    cost_caps.max_rows_per_run = lambda: 10 ** 9
    theme_summary_mod.max_rows_per_run = lambda: 10 ** 9
    theme_summary_mod.MAX_WORKERS = SENTIMENT_WORKERS

    t_start = time.time()

    # base_stem drives every output filename below — strip a trailing
    # " - filled" / " - filled+sentiment" so resuming doesn't produce
    # "... - filled - filled+sentiment.xlsx"-style double-naming.
    base_stem = path.stem
    for suffix in (" - filled+sentiment", " - filled"):
        if base_stem.endswith(suffix):
            base_stem = base_stem[:-len(suffix)]
            break

    if resume_from_sentiment:
        print(f"Resuming from already-sentiment-coded file: {path} (skipping Stages 1-2)", flush=True)
        sentiment_bytes = path.read_bytes()
    elif resume_from_filled:
        print(f"Resuming from already-filled file: {path} (skipping Stage 1)", flush=True)
        filled_bytes = path.read_bytes()
    else:
        print(f"Reading {path} ...", flush=True)
        file_bytes = path.read_bytes()

        # ---- Stage 1: Mention Filler ----
        print("\n=== Stage 1: Mention Filler ===", flush=True)
        t0 = time.time()
        parsed = parse_export(file_bytes, filename)
        print(f"  {len(parsed.urls):,} urls, streaming={parsed.is_streaming}, parsed in {time.time()-t0:.1f}s", flush=True)

        def fetch_progress(label, done, total, found):
            if total and (done % 20000 < 800 or done == total):
                print(f"  fetch {label}: {done:,}/{total:,} ({found:,} found)", flush=True)

        t1 = time.time()
        results = fetch_archive(parsed.urls, on_progress=fetch_progress)
        print(f"  fetched in {time.time()-t1:.1f}s", flush=True)

        t2 = time.time()
        out_buf, stats = fill_export(parsed, results, file_bytes=file_bytes)
        filled_bytes = out_buf.getvalue()
        print(f"  filled {stats['filled']:,}/{stats['total']:,} rows in {time.time()-t2:.1f}s "
              f"({stats['author_count']:,} authors)", flush=True)

        filled_path = path.with_name(base_stem + " - filled.xlsx")
        filled_path.write_bytes(filled_bytes)
        print(f"  saved: {filled_path}", flush=True)

    if not resume_from_sentiment:
        # ---- Stage 2: pharma/biotech sentiment ----
        print("\n=== Stage 2: Pharma/Biotech Sentiment ===", flush=True)
        t3 = time.time()
        sentiment_bytes, sentiment_stats = run_pharma_sentiment(client, filled_bytes)
        print(f"  sentiment stage in {time.time()-t3:.1f}s", flush=True)

        sentiment_path = path.with_name(base_stem + " - filled+sentiment.xlsx")
        sentiment_path.write_bytes(sentiment_bytes)
        print(f"  saved: {sentiment_path}", flush=True)
    else:
        sentiment_path = path

    # ---- Stage 3: Theme Summary (12 themes, sample 1000) ----
    print("\n=== Stage 3: Theme Summary (12 themes, sample 1000) ===", flush=True)
    t4 = time.time()
    theme_module = ThemeSummaryModule()
    theme_params = {"theme_source": "auto", "n_themes": 12, "sample_size": 1000, "manual_themes_text": ""}

    def theme_progress(frac, msg):
        print(f"  [{frac*100:5.1f}%] {msg}", flush=True)

    result = theme_module.run(None, sentiment_bytes, sentiment_path.name, theme_params, progress_cb=theme_progress)
    print(f"  theme stage in {time.time()-t4:.1f}s", flush=True)
    for line in result.summary_lines:
        print(f"  - {line}", flush=True)

    final_path = path.with_name(base_stem + " - final.xlsx")
    final_path.write_bytes(result.output_bytes)
    print(f"\nDONE. Final file: {final_path}", flush=True)
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
