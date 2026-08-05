"""Runs any number of LLMEnrichmentModules together as ONE chat completion
call per data row, instead of one call per module per row.

This is the piece that makes "run sentiment + geolocation in the same
prompt" work while keeping the two modules' code independent: each module
only ever describes its own slice (instructions, schema fields, row inputs,
how to read its answer back out of the combined JSON) via the
LLMEnrichmentModule interface (modules/base.py). This file merges those
slices into one prompt/schema, fans the calls out in parallel with retry,
and hands each module its own columns back.

The JSON schema is rebuilt PER ROW, not once for the whole run: a module can
return different (or zero) properties depending on the row (e.g. multi-entity
Sentiment Coding only asks about entities its regex prefilter actually found
in that row's text). If every module opts out of a row entirely, the row
skips the API call altogether — real cost savings, not just a hint. The
system prompt stays fixed across the run either way, so OpenAI's automatic
prompt-prefix caching still applies to it.

Follows the sync+parallel+checkpoint-free pattern for in-session bulk LLM
calls (thread pool, exponential backoff, running cost tally from real token
usage) — see the llm-bulk-api skill for the general pattern this mirrors.
"""
import concurrent.futures
import io

from core.cost_caps import CostCapExceeded, max_rows_per_run
from core.llm_client import call_json, get_client
from core.mentions_io import BadExport, ensure_columns, iter_data_rows, load_sheet_for_enrichment
from core.text_utils import looks_unfilled

# gpt-4o-mini, as of Aug 2026 — https://devtk.ai/en/models/gpt-4o-mini/
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_CACHED_INPUT = 0.075
PRICE_PER_1M_OUTPUT = 0.60

MAX_WORKERS = 20
SYSTEM_PREAMBLE = (
    "You are enriching Reddit mentions, one at a time, for a social-listening "
    "analysis. For each row you are given some context and must respond with "
    "ONLY the requested JSON fields — no other commentary, no markdown."
)


def _build_schema_for_row(modules, params_by_key, row):
    """Merge each module's (possibly row-dependent) schema fragment. A module
    that returns an empty fragment for this row is asking nothing of it."""
    properties, required = {}, []
    for mod in modules:
        frag = mod.json_schema_fragment(row, params_by_key[mod.key])
        if not frag or not frag.get("properties"):
            continue
        properties.update(frag["properties"])
        required.extend(frag.get("required", list(frag["properties"].keys())))
    if not properties:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "enrichment_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _build_system_prompt(modules, params_by_key):
    parts = [SYSTEM_PREAMBLE]
    for mod in modules:
        parts.append(f"\n## {mod.label}\n{mod.system_prompt_fragment(params_by_key[mod.key])}")
    return "\n".join(parts)


def _build_user_message(modules, params_by_key, row):
    sections = []
    for mod in modules:
        ctx = mod.row_context(row, params_by_key[mod.key])
        body = "\n".join(f"{k}: {v}" for k, v in ctx.items())
        sections.append(f"### {mod.label} input\n{body}")
    return "\n\n".join(sections)


def _write_defaults(modules, params_by_key, row, result):
    """result is None for a row that was never sent to the API (every module
    opted out), or a dict of only the fields that WERE requested that row."""
    out = {}
    for mod in modules:
        out.update(mod.columns_from_result(row, result, params_by_key[mod.key]))
    return out


def run_llm_modules(modules, params_by_key, file_bytes, filename, progress_cb=None):
    """Runs every module in `modules` (all LLMEnrichmentModule) as one combined
    call per eligible row, writes all their output columns into the same
    workbook, and returns (output_bytes, output_filename, cost_summary).
    """
    wb, ws, header_row_1based, col_index = load_sheet_for_enrichment(file_bytes)

    new_cols = []
    for mod in modules:
        new_cols += mod.output_columns(params_by_key[mod.key])
    col_index = ensure_columns(ws, header_row_1based, col_index, new_cols)

    if "Full Text" not in col_index:
        raise BadExport("No 'Full Text' column found — run Mention Filler first, or upload an already-filled export.")

    all_rows = list(iter_data_rows(ws, header_row_1based, col_index))
    eligible = [(row_num, row) for row_num, row in all_rows if not looks_unfilled(row.get("Full Text"))]
    skipped_unfilled = len(all_rows) - len(eligible)

    cap = max_rows_per_run()
    if len(eligible) > cap:
        raise CostCapExceeded(
            f"This run would send {len(eligible):,} rows to OpenAI — above the configured cap of "
            f"{int(cap):,} (MAX_LLM_ROWS_PER_RUN in the app's Secrets). Split the batch into smaller "
            f"pieces, or raise the cap in Secrets if this run is intentional."
        )

    system_prompt = _build_system_prompt(modules, params_by_key)
    client = get_client()

    results_by_row = {}      # row_num -> parsed JSON dict (only for rows actually called)
    no_signal_row_nums = set()  # eligible rows every module opted out of — never called
    usage_totals = {"input": 0, "cached_input": 0, "output": 0}
    n_failed = 0
    done = 0

    def work(row_num, row):
        schema = _build_schema_for_row(modules, params_by_key, row)
        if schema is None:
            return row_num, "no_signal", None, None
        user_message = _build_user_message(modules, params_by_key, row)
        content, usage = call_json(client, schema, system_prompt, user_message)
        return row_num, "ok", content, usage

    def _record(row_num, status, content, usage):
        if status == "no_signal":
            no_signal_row_nums.add(row_num)
            return
        results_by_row[row_num] = content
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        usage_totals["cached_input"] += cached
        usage_totals["input"] += max(0, (usage.prompt_tokens or 0) - cached)
        usage_totals["output"] += usage.completion_tokens or 0

    if eligible:
        # Test the first row that actually needs an API call synchronously before
        # fanning out the rest — a bad API key/billing issue fails once immediately
        # instead of retrying 3x across every row in the batch before it's noticed.
        preflight_done = False
        remaining = list(eligible)
        while remaining and not preflight_done:
            row_num, row = remaining.pop(0)
            schema = _build_schema_for_row(modules, params_by_key, row)
            if schema is None:
                no_signal_row_nums.add(row_num)
                done += 1
                continue
            try:
                _, status, content, usage = work(row_num, row)
            except Exception as e:
                raise RuntimeError(
                    f"The first test call to OpenAI failed — check your API key/billing "
                    f"before running the full batch. Details: {e}"
                ) from e
            _record(row_num, status, content, usage)
            done += 1
            preflight_done = True
            if progress_cb:
                progress_cb(done / len(eligible), f"Enriching rows: {done:,}/{len(eligible):,}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(work, row_num, row) for row_num, row in remaining]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    row_num, status, content, usage = fut.result()
                    _record(row_num, status, content, usage)
                except Exception:
                    n_failed += 1
                done += 1
                if progress_cb:
                    progress_cb(done / len(eligible), f"Enriching rows: {done:,}/{len(eligible):,}")

    eligible_row_nums = {row_num for row_num, _ in eligible}
    for row_num, row in all_rows:
        if row_num in results_by_row:
            values = _write_defaults(modules, params_by_key, row, results_by_row[row_num])
        elif row_num in no_signal_row_nums:
            values = _write_defaults(modules, params_by_key, row, None)
        elif row_num in eligible_row_nums:
            values = {c: "ERROR" for mod in modules for c in mod.output_columns(params_by_key[mod.key])}
        else:
            values = {c: "" for mod in modules for c in mod.output_columns(params_by_key[mod.key])}
        for c, v in values.items():
            if c in col_index:
                ws.cell(row=row_num, column=col_index[c], value=v)

    out_buf = io.BytesIO()
    wb.save(out_buf)
    out_buf.seek(0)

    cost = (
        usage_totals["input"] / 1_000_000 * PRICE_PER_1M_INPUT
        + usage_totals["cached_input"] / 1_000_000 * PRICE_PER_1M_CACHED_INPUT
        + usage_totals["output"] / 1_000_000 * PRICE_PER_1M_OUTPUT
    )
    cost_summary = {
        **usage_totals, "cost_usd": cost, "n_rows": len(eligible), "n_failed": n_failed,
        "n_skipped": skipped_unfilled, "n_no_signal": len(no_signal_row_nums),
    }
    return out_buf.getvalue(), filename, cost_summary
