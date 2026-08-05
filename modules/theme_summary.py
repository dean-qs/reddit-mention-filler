"""Theme Summary — a dataset-level LLM module (not per-row like Sentiment/
Geolocation, so it's a plain AnalysisModule with its own two-phase flow
rather than an LLMEnrichmentModule plugged into the shared combined-call
coordinator):

  1. Discover: either auto-discover — sample up to `sample_size` filled
     mentions, ask the LLM once for up to `n_themes` recurring themes (name +
     description) — or use a predefined list the user typed in, which skips
     this call (and its cost) entirely.
  2. Tag: ask the LLM, once per eligible row, which theme fits best (or
     "Other / none of the above").

Writes a "Theme" (+ rationale) column on every row plus a "Theme Summary"
sheet: theme, description, count, and a couple of example quotes.
"""
import io
import random
import concurrent.futures

from core.cost_caps import CostCapExceeded, max_rows_per_run
from core.llm_client import call_json, get_client
from core.llm_enrichment import PRICE_PER_1M_CACHED_INPUT, PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT
from core.mentions_io import BadExport, ensure_columns, iter_data_rows, load_sheet_for_enrichment
from core.text_utils import looks_unfilled, rough_token_estimate
from .base import AnalysisModule, Estimate, ModuleResult

SAMPLE_TEXT_CHARS = 300     # per-row truncation for the discovery prompt
TAG_TEXT_CHARS = 1500       # per-row truncation for the tagging prompt
OTHER_THEME = "Other / none of the above"
MAX_WORKERS = 20

DISCOVERY_SYSTEM_PROMPT = (
    "You are analyzing a sample of Reddit mentions from a social-listening dataset. "
    "Identify up to {n_themes} recurring themes/topics that meaningfully group these "
    "mentions — the kind of buckets an analyst would use to summarize what people are "
    "actually talking about. Prefer fewer, clearer themes over many overlapping ones. "
    "Each theme needs a short name (2-5 words) and a one-sentence description.\n\n"
    "Respond with 'themes': an array of {{name, description}} objects."
)

TAG_SYSTEM_PROMPT_TEMPLATE = (
    "Assign this mention to the single best-fitting theme from the list below, or "
    "'{other}' if truly none fit. Respond with 'Theme' (the exact theme name) and "
    "'Theme Rationale' (a short clause, under 15 words).\n\nThemes:\n{theme_list}"
)


def _discovery_schema(n_themes):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "theme_discovery",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "themes": {
                        "type": "array",
                        "maxItems": n_themes,
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                            "required": ["name", "description"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["themes"],
                "additionalProperties": False,
            },
        },
    }


def _tag_schema(theme_names):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "theme_tag",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "Theme": {"type": "string", "enum": theme_names + [OTHER_THEME]},
                    "Theme Rationale": {"type": "string"},
                },
                "required": ["Theme", "Theme Rationale"],
                "additionalProperties": False,
            },
        },
    }


def _usage_cost(usage_totals):
    return (
        usage_totals["input"] / 1_000_000 * PRICE_PER_1M_INPUT
        + usage_totals["cached_input"] / 1_000_000 * PRICE_PER_1M_CACHED_INPUT
        + usage_totals["output"] / 1_000_000 * PRICE_PER_1M_OUTPUT
    )


def _add_usage(totals, usage):
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    totals["cached_input"] += cached
    totals["input"] += max(0, (usage.prompt_tokens or 0) - cached)
    totals["output"] += usage.completion_tokens or 0


def _parse_manual_themes(text):
    """One theme per line: 'ThemeName: description'. Description is optional
    (falls back to empty) but a name is required for the line to count."""
    themes = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, desc = line.partition(":")
        name = name.strip()
        if not name:
            continue
        themes.append({"name": name, "description": desc.strip()})
    return themes


class ThemeSummaryModule(AnalysisModule):
    key = "theme_summary"
    label = "Theme Summary"
    description = "Discover recurring themes in the dataset and tag every mention with one (LLM)."
    uses_paid_api = True

    def render_options(self, st, key_prefix, parsed=None, file_bytes=None, filename=None):
        n_rows = len(parsed.urls) if parsed else 200
        source = st.radio(
            "How should themes be determined?",
            ["Discover automatically", "I'll define them myself"],
            key=f"{key_prefix}_source",
        )
        params = {"theme_source": "auto" if source == "Discover automatically" else "manual",
                  "n_themes": 0, "sample_size": 0, "manual_themes_text": ""}

        if params["theme_source"] == "auto":
            params["n_themes"] = st.slider("Number of themes to identify", 3, 12, 6, key=f"{key_prefix}_n_themes")
            params["sample_size"] = st.slider(
                "Sample size for theme discovery", 20, 500, min(150, n_rows), key=f"{key_prefix}_sample_size",
                help="How many mentions the discovery step reads to propose themes. Every row still "
                     "gets tagged afterward, regardless of this number — this only controls the "
                     "one-time discovery call's cost.",
            )
        else:
            manual_text = st.text_area(
                "One theme per line: ThemeName: description",
                height=140,
                key=f"{key_prefix}_manual_themes",
                placeholder="Addiction/Overuse: concern about excessive or compulsive use\n"
                            "Customer Support: experiences with the support/help team\n"
                            "Bugs/Technical Issues: complaints about bugs or broken updates",
                help="Skips the discovery call entirely (and its cost) — every row is tagged "
                     "directly against this list.",
            )
            params["manual_themes_text"] = manual_text
            themes = _parse_manual_themes(manual_text)
            if themes:
                st.caption(f"{len(themes)} themes defined: {', '.join(t['name'] for t in themes)}")
        return params

    # ------------------------------------------------------------- estimate ---
    def estimate(self, parsed, params, context) -> Estimate:
        n = len(parsed.urls)
        manual = params.get("theme_source") == "manual"
        lines = []

        if manual:
            manual_themes = _parse_manual_themes(params.get("manual_themes_text"))
            discovery_in = discovery_out = 0
            n_themes_for_estimate = len(manual_themes)
            if not manual_themes:
                lines.append("⚠️ No themes defined yet — add at least one 'ThemeName: description' "
                              "line above, or every row will just be tagged 'Other'.")
            else:
                lines.append(f"{len(manual_themes)} themes predefined — skips the discovery call "
                              f"entirely (and its cost).")
        else:
            sample = min(params["sample_size"], n)
            discovery_in = sample * (SAMPLE_TEXT_CHARS // 4 + 10) + 100
            discovery_out = params["n_themes"] * 30 + 20
            n_themes_for_estimate = params["n_themes"]
            lines.append(f"Discovery: one call reading a {sample:,}-row sample to propose up to "
                         f"{params['n_themes']} themes.")

        tag_in_per_row = rough_token_estimate(TAG_SYSTEM_PROMPT_TEMPLATE) + n_themes_for_estimate * 15 + 200
        tag_out_per_row = 20
        total_in = discovery_in + tag_in_per_row * n
        total_out = discovery_out + tag_out_per_row * n
        cost = total_in / 1_000_000 * PRICE_PER_1M_INPUT + total_out / 1_000_000 * PRICE_PER_1M_OUTPUT
        lines.append(f"Tagging: {n:,} calls (one per row) to assign each mention to a theme.")
        lines.append(f"Uses OpenAI gpt-4o-mini — estimated cost: ${cost:,.2f} total, rough.")
        lines.append("Real cost is computed from actual token usage after the run and shown in the results.")
        if not context.get("text_will_be_filled"):
            lines.insert(0, "⚠️ Full Text doesn't look filled yet in this file — run Mention Filler "
                             "first (in this same run, or upload an already-filled export) or this "
                             "module will skip every row.")
        return Estimate(headline=f"Theme discovery + tagging for {n:,} rows", lines=lines, est_cost_usd=cost)

    # ------------------------------------------------------------------ run ---
    def run(self, parsed, file_bytes, filename, params, progress_cb=None) -> ModuleResult:
        wb, ws, header_row_1based, col_index = load_sheet_for_enrichment(file_bytes)
        col_index = ensure_columns(ws, header_row_1based, col_index, ["Theme", "Theme Rationale"])

        if "Full Text" not in col_index:
            raise BadExport("No 'Full Text' column found — run Mention Filler first, or upload an already-filled export.")

        all_rows = list(iter_data_rows(ws, header_row_1based, col_index))
        eligible = [(row_num, row) for row_num, row in all_rows if not looks_unfilled(row.get("Full Text"))]

        cap = max_rows_per_run()
        if len(eligible) > cap:
            raise CostCapExceeded(
                f"This run would send {len(eligible):,} rows to OpenAI — above the configured cap of "
                f"{int(cap):,} (MAX_LLM_ROWS_PER_RUN in the app's Secrets). Split the batch, or raise "
                f"the cap in Secrets if this run is intentional."
            )

        usage_totals = {"input": 0, "cached_input": 0, "output": 0}
        client = get_client()

        if not eligible:
            out_buf = io.BytesIO()
            wb.save(out_buf)
            return ModuleResult(
                output_bytes=out_buf.getvalue(), output_filename=filename,
                summary_lines=["No filled rows to summarize — nothing to do."],
            )

        # ---- Phase 1: determine themes — auto-discover from a sample, or use the predefined list ----
        manual = params.get("theme_source") == "manual"
        sample = []
        if manual:
            if progress_cb:
                progress_cb(0.0, "Using predefined themes (skipping discovery)...")
            themes = _parse_manual_themes(params.get("manual_themes_text"))
            if not themes:
                raise RuntimeError("No themes defined — add at least one 'ThemeName: description' "
                                   "line, or switch to automatic discovery.")
        else:
            if progress_cb:
                progress_cb(0.0, "Discovering themes from a sample...")
            sample = random.sample(eligible, min(params["sample_size"], len(eligible)))
            sample_lines = [
                f"{i+1}. {str(row.get('Full Text') or '')[:SAMPLE_TEXT_CHARS]}" for i, (_, row) in enumerate(sample)
            ]
            discovery_system = DISCOVERY_SYSTEM_PROMPT.format(n_themes=params["n_themes"])
            discovery_user = "\n".join(sample_lines)
            try:
                discovery_result, discovery_usage = call_json(
                    client, _discovery_schema(params["n_themes"]), discovery_system, discovery_user
                )
            except Exception as e:
                raise RuntimeError(f"Theme discovery call to OpenAI failed: {e}") from e
            _add_usage(usage_totals, discovery_usage)
            themes = discovery_result.get("themes") or []
            if not themes:
                raise RuntimeError("Theme discovery returned no themes — try a larger sample or fewer/more themes.")

        theme_names = [t["name"] for t in themes]
        theme_descriptions = {t["name"]: t.get("description", "") for t in themes}

        # ---- Phase 2: tag every eligible row ----
        theme_list_text = "\n".join(f"- {name}: {theme_descriptions[name]}" for name in theme_names)
        tag_system = TAG_SYSTEM_PROMPT_TEMPLATE.format(other=OTHER_THEME, theme_list=theme_list_text)
        tag_schema = _tag_schema(theme_names)

        def tag_one(row_num, row):
            text = str(row.get("Full Text") or "")[:TAG_TEXT_CHARS]
            title = row.get("Title") or ""
            user_message = f"Post Title: {title}\nFull Text: {text}"
            content, usage = call_json(client, tag_schema, tag_system, user_message)
            return row_num, content, usage

        results_by_row = {}
        n_failed = 0
        done = 0

        first_row_num, first_row = eligible[0]
        try:
            _, first_content, first_usage = tag_one(first_row_num, first_row)
        except Exception as e:
            raise RuntimeError(
                f"The first tagging test call to OpenAI failed — check your API key/billing "
                f"before running the full batch. Details: {e}"
            ) from e
        results_by_row[first_row_num] = first_content
        _add_usage(usage_totals, first_usage)
        done = 1
        if progress_cb:
            progress_cb(done / len(eligible), f"Tagging rows: {done:,}/{len(eligible):,}")

        remaining = eligible[1:]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(tag_one, row_num, row) for row_num, row in remaining]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    row_num, content, usage = fut.result()
                    results_by_row[row_num] = content
                    _add_usage(usage_totals, usage)
                except Exception:
                    n_failed += 1
                done += 1
                if progress_cb:
                    progress_cb(done / len(eligible), f"Tagging rows: {done:,}/{len(eligible):,}")

        # ---- write per-row columns + aggregate ----
        theme_counts = {name: 0 for name in theme_names}
        theme_counts[OTHER_THEME] = 0
        theme_quotes = {name: [] for name in list(theme_names) + [OTHER_THEME]}
        eligible_row_nums = {row_num for row_num, _ in eligible}

        for row_num, row in all_rows:
            result = results_by_row.get(row_num)
            if result is None:
                theme, rationale = ("ERROR", "") if row_num in eligible_row_nums else ("", "")
            else:
                theme = result.get("Theme", OTHER_THEME)
                rationale = result.get("Theme Rationale", "")
                if theme in theme_counts:
                    theme_counts[theme] += 1
                    if len(theme_quotes[theme]) < 2:
                        theme_quotes[theme].append(str(row.get("Full Text") or "")[:200])
            ws.cell(row=row_num, column=col_index["Theme"], value=theme)
            ws.cell(row=row_num, column=col_index["Theme Rationale"], value=rationale)

        summary_ws = wb.create_sheet("Theme Summary")
        summary_ws.append(["Theme", "Description", "Count", "Example Quote 1", "Example Quote 2"])
        for name in sorted(theme_counts, key=lambda n: -theme_counts[n]):
            desc = theme_descriptions.get(name, "Doesn't fit any of the discovered themes.")
            quotes = theme_quotes.get(name, [])
            summary_ws.append([name, desc, theme_counts[name],
                                quotes[0] if len(quotes) > 0 else "", quotes[1] if len(quotes) > 1 else ""])

        out_buf = io.BytesIO()
        wb.save(out_buf)
        out_buf.seek(0)

        cost = _usage_cost(usage_totals)
        theme_source_line = (
            f"Used {len(theme_names)} predefined themes: {', '.join(theme_names)}"
            if manual else
            f"Discovered {len(theme_names)} themes from a {len(sample):,}-row sample: {', '.join(theme_names)}"
        )
        summary_lines = [
            theme_source_line,
            f"Tagged {len(eligible) - n_failed:,} of {len(eligible):,} eligible rows"
            + (f", {n_failed:,} failed after retries" if n_failed else ""),
            f"Actual cost: ${cost:,.4f} ({usage_totals['input']:,} input + "
            f"{usage_totals['cached_input']:,} cached + {usage_totals['output']:,} output tokens)",
            "See the 'Theme Summary' sheet for counts and example quotes per theme.",
        ]
        return ModuleResult(output_bytes=out_buf.getvalue(), output_filename=filename, summary_lines=summary_lines)
