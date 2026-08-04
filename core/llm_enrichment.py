"""Runs any number of LLMEnrichmentModules together as ONE chat completion
call per data row, instead of one call per module per row.

This is the piece that makes "run sentiment + geolocation in the same
prompt" work while keeping the two modules' code independent: each module
only ever describes its own slice (instructions, schema fields, row inputs,
how to read its answer back out of the combined JSON) via the
LLMEnrichmentModule interface (modules/base.py). This file merges those
slices into one prompt/schema, fans the calls out in parallel with retry,
and hands each module its own columns back.

Follows the sync+parallel+checkpoint-free pattern for in-session bulk LLM
calls (thread pool, exponential backoff, running cost tally from real token
usage) — see the llm-bulk-api skill for the general pattern this mirrors.
"""
import concurrent.futures
import io
import json
import time

from core.llm_client import MODEL, get_client
from core.mentions_io import BadExport, ensure_columns, iter_data_rows, load_sheet_for_enrichment
from core.text_utils import looks_unfilled

# gpt-4o-mini, as of Aug 2026 — https://devtk.ai/en/models/gpt-4o-mini/
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_CACHED_INPUT = 0.075
PRICE_PER_1M_OUTPUT = 0.60

MAX_WORKERS = 20
MAX_RETRIES = 3
SYSTEM_PREAMBLE = (
    "You are enriching Reddit mentions, one at a time, for a social-listening "
    "analysis. For each row you are given some context and must respond with "
    "ONLY the requested JSON fields — no other commentary, no markdown."
)


def _build_schema(modules, params_by_key):
    properties, required = {}, []
    for mod in modules:
        frag = mod.json_schema_fragment(params_by_key[mod.key])
        properties.update(frag["properties"])
        required.extend(frag.get("required", list(frag["properties"].keys())))
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


def _call_one(client, schema, system_prompt, user_message, attempt=1):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=schema,
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content), resp.usage
    except Exception:
        if attempt >= MAX_RETRIES:
            raise
        time.sleep(2 ** attempt)
        return _call_one(client, schema, system_prompt, user_message, attempt + 1)


def run_llm_modules(modules, params_by_key, file_bytes, filename, progress_cb=None):
    """Runs every module in `modules` (all LLMEnrichmentModule) as one combined
    call per eligible row, writes all their output columns into the same
    workbook, and returns (output_bytes, output_filename, summary_lines, cost_summary).
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
    skipped = len(all_rows) - len(eligible)

    schema = _build_schema(modules, params_by_key)
    system_prompt = _build_system_prompt(modules, params_by_key)
    client = get_client()

    results_by_row = {}
    usage_totals = {"input": 0, "cached_input": 0, "output": 0}
    n_failed = 0
    done = 0

    def work(row_num, row):
        user_message = _build_user_message(modules, params_by_key, row)
        content, usage = _call_one(client, schema, system_prompt, user_message)
        return row_num, content, usage

    def _record(row_num, content, usage):
        results_by_row[row_num] = content
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        usage_totals["cached_input"] += cached
        usage_totals["input"] += max(0, (usage.prompt_tokens or 0) - cached)
        usage_totals["output"] += usage.completion_tokens or 0

    if eligible:
        # Test the very first row synchronously before fanning out the rest — a bad
        # API key/billing issue fails once immediately instead of retrying 3x across
        # every row in the batch before the user finds out.
        first_row_num, first_row = eligible[0]
        try:
            _, first_content, first_usage = work(first_row_num, first_row)
        except Exception as e:
            raise RuntimeError(
                f"The first test call to OpenAI failed — check your API key/billing "
                f"before running the full batch. Details: {e}"
            ) from e
        _record(first_row_num, first_content, first_usage)
        done = 1
        if progress_cb:
            progress_cb(done / len(eligible), f"Enriching rows: {done:,}/{len(eligible):,}")

        remaining = eligible[1:]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(work, row_num, row) for row_num, row in remaining]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    row_num, content, usage = fut.result()
                    _record(row_num, content, usage)
                except Exception:
                    n_failed += 1
                done += 1
                if progress_cb:
                    progress_cb(done / len(eligible), f"Enriching rows: {done:,}/{len(eligible):,}")

    eligible_row_nums = {row_num for row_num, _ in eligible}
    for row_num, row in all_rows:
        result = results_by_row.get(row_num)
        for mod in modules:
            cols = mod.output_columns(params_by_key[mod.key])
            if result is not None:
                values = mod.columns_from_result(result, params_by_key[mod.key])
            elif row_num in eligible_row_nums:
                values = {c: "ERROR" for c in cols}
            else:
                values = {c: "" for c in cols}
            for c in cols:
                ws.cell(row=row_num, column=col_index[c], value=values.get(c, ""))

    out_buf = io.BytesIO()
    wb.save(out_buf)
    out_buf.seek(0)

    cost = (
        usage_totals["input"] / 1_000_000 * PRICE_PER_1M_INPUT
        + usage_totals["cached_input"] / 1_000_000 * PRICE_PER_1M_CACHED_INPUT
        + usage_totals["output"] / 1_000_000 * PRICE_PER_1M_OUTPUT
    )
    cost_summary = {**usage_totals, "cost_usd": cost, "n_rows": len(eligible), "n_failed": n_failed, "n_skipped": skipped}
    return out_buf.getvalue(), filename, cost_summary
