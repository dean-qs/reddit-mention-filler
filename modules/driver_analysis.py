"""Driver / Barrier Analysis — thematic analysis subset by entity, to find
which conversation themes are driving positive vs. negative sentiment
toward each brand. Builds on Sentiment Coding's per-entity "LLM Sentiment:
<entity>" columns (multi-entity mode) — run Sentiment Coding first, either
earlier in this same pass or on a file already processed.

For each entity (or "Rest of Field" in owned-vs-competitor mode):
  1. Discover themes from a sample of that entity's mentions — agnostic of
     sentiment, since the same theme can show up in both praise and
     criticism (e.g. one post can be a driver on one theme and a barrier on
     another: "YouTube Kids is amazing, but I worry about circumventing
     controls" touches both). Reuses core/theme_discovery.py.
  2. Tag every mention with EVERY theme that genuinely applies (multi-label)
     plus that mention's sentiment TOWARD THAT THEME specifically — not
     reused from the mention's one overall entity sentiment, since a single
     mixed post can be positive on one theme and negative on another.
  3. Classify each theme as Driver / Barrier / Neutral by net sentiment
     ((Positive - Negative) / Total) against a configurable threshold.
  4. Write a narrative summary per entity synthesizing its drivers/barriers.

Plain AnalysisModule (like Theme Summary) — multi-phase and entity-scoped,
doesn't fit the shared per-row LLMEnrichmentModule coordinator.
"""
import concurrent.futures
import datetime
import random

from core.cost_caps import CostCapExceeded, max_rows_per_run
from core.entity_detection import detect_scored_entities
from core.llm_client import call_json, get_client
from core.llm_cost import PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT, add_usage, new_usage_totals, usage_cost
from core.mentions_io import BadExport, load_sheet_for_enrichment
from core.text_utils import rough_token_estimate
from core.theme_discovery import discover_themes
from .base import AnalysisModule, Estimate, ModuleResult

TAG_TEXT_CHARS = 1500
MAX_WORKERS = 20
MAX_TAGS_PER_MENTION = 4
REAL_SENTIMENTS = ("Positive", "Negative", "Neutral")

DISCOVERY_SYSTEM_PROMPT = (
    "You are analyzing a sample of mentions about {label} from a social-listening dataset. "
    "Identify up to {n_themes} recurring themes/topics in how people talk about {label} — the "
    "kind of buckets an analyst would use to understand what's driving conversation, both "
    "positive and negative. Discover themes independent of sentiment — the same theme can come "
    "up in both praise and criticism; sentiment is analyzed separately afterward. Prefer fewer, "
    "clearer themes over many overlapping ones. Each theme needs a short name (2-5 words) and a "
    "one-sentence description.\n\n"
    "Respond with 'themes': an array of {{name, description}} objects."
)

TAG_SYSTEM_PROMPT_TEMPLATE = (
    "This mention is about {label}. Identify every theme below that genuinely applies to it — "
    "usually one, but tag more than one when the mention clearly touches multiple distinct "
    "themes (e.g. praising one aspect while criticizing another in the same post). For each "
    "theme you tag, also give the sentiment expressed toward THAT SPECIFIC THEME in this "
    "mention (Positive, Negative, or Neutral) — this can differ from theme to theme within the "
    "same mention. Use Neutral for a theme when the mention touches it without a clear "
    "evaluative slant toward that specific aspect — a purely descriptive or passing reference, "
    "not just mild positivity/negativity. Only tag themes that are genuinely present; return an "
    "empty list if none of the themes below fit.\n\nThemes:\n{theme_list}"
)

SUMMARY_SYSTEM_PROMPT_TEMPLATE = (
    "You are a social-listening analyst. Given this table of conversation themes about {label} "
    "— each with its Driver/Barrier/Neutral classification, volume, and net sentiment — write a "
    "concise 3-5 sentence summary of the main drivers and barriers: what's working in {label}'s "
    "favor and what's working against it. Name the actual themes. If there are no clear drivers "
    "or no clear barriers, say so plainly rather than forcing one."
)

REST_OF_FIELD_LABEL = "Rest of Field"

THEME_SHEET_COLS = [
    "Entity", "Theme", "Description", "Classification", "Volume", "Positive", "Negative", "Neutral",
    "Net Sentiment", "Example Positive Quote", "Example Positive Url", "Example Negative Quote",
    "Example Negative Url",
]

MONTHLY_SHEET_COLS = [
    "Entity", "Theme", "Month", "Positive", "Negative", "Neutral", "Volume", "Net Sentiment",
    "Net Sentiment vs Prior Month", "Volume % Change vs Prior Month",
]


def _tag_schema(theme_names):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "driver_tag",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "themes": {
                        "type": "array",
                        "maxItems": MAX_TAGS_PER_MENTION,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "enum": theme_names},
                                "sentiment": {"type": "string", "enum": list(REAL_SENTIMENTS)},
                            },
                            "required": ["name", "sentiment"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["themes"],
                "additionalProperties": False,
            },
        },
    }


def _summary_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "driver_summary",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    }


def _month_key(date_val):
    """"YYYY-MM" for a row's Date value, or None if it can't be determined.
    Mention Filler's own output always writes real datetime cells, but this
    stays lenient about ISO-formatted date strings too (e.g. a file whose
    Date column was touched by another tool, or a CSV round-trip) since
    those are a plausible real input, not just an implementation detail."""
    if isinstance(date_val, (datetime.date, datetime.datetime)):
        return date_val.strftime("%Y-%m")
    if isinstance(date_val, str) and date_val:
        try:
            return datetime.date.fromisoformat(date_val[:10]).strftime("%Y-%m")
        except ValueError:
            return None
    return None


def _filter_relevant(all_rows, entity_name):
    col = f"LLM Sentiment: {entity_name}"
    return [(row_num, row) for row_num, row in all_rows if row.get(col) in REAL_SENTIMENTS]


def _build_rest_of_field_units(all_rows, competitor_names):
    """A row mentioning two competitors contributes one unit per competitor —
    same entity-specific attribution Sentiment Coding already does; a single
    post can be evidence for more than one competitor's analysis."""
    units = []
    for row_num, row in all_rows:
        for name in competitor_names:
            if row.get(f"LLM Sentiment: {name}") in REAL_SENTIMENTS:
                units.append((f"{row_num}::{name}", row))
    return units


def _generate_summary(client, label, theme_rows, usage_totals):
    if not theme_rows:
        return f"No mentions found for {label} — nothing to summarize."
    table_text = "\n".join(
        f"- {r['Theme']} ({r['Classification']}, net {r['Net Sentiment']}, volume {r['Volume']}): {r['Description']}"
        for r in theme_rows
    )
    try:
        result, usage = call_json(client, _summary_schema(), SUMMARY_SYSTEM_PROMPT_TEMPLATE.format(label=label), table_text)
        add_usage(usage_totals, usage)
        return result.get("summary", "")
    except Exception as e:
        return f"(Summary generation failed: {e})"


def _analyze_bucket(client, label, units, n_themes, sample_size, threshold, usage_totals, progress_cb):
    """units: list of (unit_key, row). Returns (theme_rows, monthly_rows,
    summary_text, n_tagged, n_failed, n_no_date)."""
    if not units:
        return [], [], f"No mentions found for {label}.", 0, 0, 0

    if progress_cb:
        progress_cb(0.0, f"{label}: discovering themes from a sample...")
    sample = random.sample(units, min(sample_size, len(units)))
    sample_rows = [row for _, row in sample]
    discovery_system = DISCOVERY_SYSTEM_PROMPT.format(label=label, n_themes=n_themes)
    theme_names, theme_descriptions = discover_themes(client, discovery_system, sample_rows, n_themes, usage_totals)

    theme_list_text = "\n".join(f"- {name}: {theme_descriptions[name]}" for name in theme_names)
    tag_system = TAG_SYSTEM_PROMPT_TEMPLATE.format(label=label, theme_list=theme_list_text)
    tag_schema = _tag_schema(theme_names)

    def tag_one(unit_key, row):
        text = str(row.get("Full Text") or "")[:TAG_TEXT_CHARS]
        title = row.get("Title") or ""
        user_message = f"Post Title: {title}\nFull Text: {text}"
        content, usage = call_json(client, tag_schema, tag_system, user_message)
        return unit_key, content, usage

    results = {}
    n_failed = 0
    done = 0

    first_key, first_row = units[0]
    try:
        _, first_content, first_usage = tag_one(first_key, first_row)
    except Exception as e:
        raise RuntimeError(
            f"The first tagging test call for {label} failed — check your API key/billing "
            f"before running the full batch. Details: {e}"
        ) from e
    results[first_key] = first_content
    add_usage(usage_totals, first_usage)
    done = 1
    if progress_cb:
        progress_cb(done / len(units), f"{label}: tagging {done:,}/{len(units):,}")

    remaining = units[1:]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(tag_one, k, r) for k, r in remaining]
        for fut in concurrent.futures.as_completed(futures):
            try:
                k, content, usage = fut.result()
                results[k] = content
                add_usage(usage_totals, usage)
            except Exception:
                n_failed += 1
            done += 1
            if progress_cb:
                progress_cb(done / len(units), f"{label}: tagging {done:,}/{len(units):,}")

    counts = {name: {"Positive": 0, "Negative": 0, "Neutral": 0} for name in theme_names}
    quotes = {name: {"Positive": None, "Negative": None} for name in theme_names}
    # (theme, "YYYY-MM") -> counts, accumulated from the same tag results
    # above — no extra LLM calls needed for the monthly breakdown.
    monthly_counts = {}
    no_date_unit_keys = set()
    for unit_key, row in units:
        result = results.get(unit_key)
        if not result:
            continue
        tags = result.get("themes", [])
        if not tags:
            continue
        month_key = _month_key(row.get("Date"))
        if month_key is None:
            no_date_unit_keys.add(unit_key)
        for tag in tags:
            name, sentiment = tag.get("name"), tag.get("sentiment")
            if name not in counts or sentiment not in REAL_SENTIMENTS:
                continue
            counts[name][sentiment] += 1
            if sentiment in ("Positive", "Negative") and quotes[name][sentiment] is None:
                quotes[name][sentiment] = (str(row.get("Full Text") or "")[:250], row.get("Url", ""))
            if month_key:
                monthly_key = (name, month_key)
                if monthly_key not in monthly_counts:
                    monthly_counts[monthly_key] = {"Positive": 0, "Negative": 0, "Neutral": 0}
                monthly_counts[monthly_key][sentiment] += 1

    theme_rows = []
    for name in theme_names:
        c = counts[name]
        total = c["Positive"] + c["Negative"] + c["Neutral"]
        net = round((c["Positive"] - c["Negative"]) / total, 3) if total else None
        if net is None:
            classification = "No data"
        elif net > threshold:
            classification = "Driver"
        elif net < -threshold:
            classification = "Barrier"
        else:
            classification = "Neutral"
        pos_quote, pos_url = quotes[name]["Positive"] or ("", "")
        neg_quote, neg_url = quotes[name]["Negative"] or ("", "")
        theme_rows.append({
            "Entity": label, "Theme": name, "Description": theme_descriptions.get(name, ""),
            "Classification": classification, "Volume": total,
            "Positive": c["Positive"], "Negative": c["Negative"], "Neutral": c["Neutral"],
            "Net Sentiment": net if net is not None else "",
            "Example Positive Quote": pos_quote, "Example Positive Url": pos_url,
            "Example Negative Quote": neg_quote, "Example Negative Url": neg_url,
        })
    theme_rows.sort(key=lambda r: -(r["Net Sentiment"] if isinstance(r["Net Sentiment"], (int, float)) else 0))

    monthly_rows = []
    for name in theme_names:
        months_for_theme = sorted(m for (n, m) in monthly_counts if n == name)
        prev_net, prev_volume = None, None
        for month in months_for_theme:
            c = monthly_counts[(name, month)]
            total = c["Positive"] + c["Negative"] + c["Neutral"]
            net = round((c["Positive"] - c["Negative"]) / total, 3) if total else None
            net_change = round(net - prev_net, 3) if (net is not None and prev_net is not None) else ""
            vol_change = round((total - prev_volume) / prev_volume * 100, 1) if prev_volume else ""
            monthly_rows.append({
                "Entity": label, "Theme": name, "Month": month,
                "Positive": c["Positive"], "Negative": c["Negative"], "Neutral": c["Neutral"],
                "Volume": total, "Net Sentiment": net if net is not None else "",
                "Net Sentiment vs Prior Month": net_change,
                "Volume % Change vs Prior Month": vol_change,
            })
            prev_net, prev_volume = net, total

    summary_text = _generate_summary(client, label, theme_rows, usage_totals)
    return theme_rows, monthly_rows, summary_text, len(units) - n_failed, n_failed, len(no_date_unit_keys)


class DriverAnalysisModule(AnalysisModule):
    key = "driver_analysis"
    label = "Driver / Barrier Analysis"
    description = "Thematic analysis of what's driving positive/negative sentiment toward each brand (LLM)."
    uses_paid_api = True

    def render_options(self, st, key_prefix, parsed=None, file_bytes=None, filename=None):
        available = []
        if file_bytes and filename:
            try:
                available = detect_scored_entities(file_bytes, filename)
            except Exception:
                available = []

        if available:
            st.caption(f"Detected entities (from 'LLM Sentiment: X' columns already in this file): "
                       f"{', '.join(available)}")
        else:
            st.info("No entity-sentiment columns detected yet in this file. If you're running "
                     "Sentiment Coding (Multiple entities mode) in this same pass, that's expected — "
                     "entities are picked up automatically once this module actually runs. Otherwise, "
                     "run Sentiment Coding first.")

        mode = st.radio(
            "Analysis mode",
            ["By-brand (drivers/barriers for each)", "Owned vs. competitor"],
            key=f"{key_prefix}_mode",
        )
        params = {"mode": "by_brand" if mode.startswith("By-brand") else "owned_vs_competitor", "owned_entity": ""}

        if params["mode"] == "owned_vs_competitor":
            if available:
                owned = st.selectbox("Owned brand", available, key=f"{key_prefix}_owned")
            else:
                owned = st.text_input(
                    "Owned brand (must match a Sentiment Coding entity name exactly)",
                    key=f"{key_prefix}_owned_text",
                )
            params["owned_entity"] = (owned or "").strip()

        params["n_themes"] = st.slider("Number of themes to identify per brand", 3, 12, 6, key=f"{key_prefix}_n_themes")
        params["sample_size"] = st.slider(
            "Sample size for theme discovery (per brand)", 20, 500, 150, key=f"{key_prefix}_sample_size"
        )
        params["threshold"] = st.slider(
            "Neutrality threshold (+/- %)", 0, 30, 10, key=f"{key_prefix}_threshold",
            help="Themes with net sentiment inside +/- this band are classified Neutral rather "
                 "than Driver/Barrier.",
        ) / 100.0
        return params

    def estimate(self, parsed, params, context) -> Estimate:
        n = len(parsed.urls)
        lines = [
            "Requires Sentiment Coding (Multiple entities mode) to have already run — either "
            "earlier in this same pass or on an already-processed file.",
        ]
        assumed_relevant_frac = 0.3
        assumed_n_buckets = 5 if params["mode"] == "by_brand" else 2
        per_bucket_rows = max(1, int(n * assumed_relevant_frac))
        discovery_in = min(params["sample_size"], per_bucket_rows) * 90 + 100
        discovery_out = params["n_themes"] * 30 + 20
        tag_in_per_row = rough_token_estimate(TAG_SYSTEM_PROMPT_TEMPLATE) + params["n_themes"] * 20 + 250
        tag_out_per_row = 60
        summary_in, summary_out = 300, 150
        per_bucket_cost = (
            (discovery_in + tag_in_per_row * per_bucket_rows + summary_in) / 1_000_000 * PRICE_PER_1M_INPUT
            + (discovery_out + tag_out_per_row * per_bucket_rows + summary_out) / 1_000_000 * PRICE_PER_1M_OUTPUT
        )
        cost = per_bucket_cost * assumed_n_buckets
        lines.append(
            f"Very rough estimate assuming ~{assumed_n_buckets} brand(s)/buckets and ~"
            f"{int(assumed_relevant_frac * 100)}% of rows relevant per brand — actual scope depends on "
            f"how many entities Sentiment Coding found and how many rows mention each; the real number "
            f"isn't known until this module actually runs."
        )
        lines.append(f"Uses OpenAI gpt-4o-mini — very rough estimated cost: ${cost:,.2f}.")
        lines.append("Real cost is computed from actual token usage after the run and shown in the results.")
        if not context.get("text_will_be_filled"):
            lines.insert(0, "⚠️ Full Text doesn't look filled yet — run Mention Filler first.")
        return Estimate(
            headline=f"Driver/Barrier analysis ({params['mode'].replace('_', ' ')})", lines=lines, est_cost_usd=cost
        )

    def run(self, parsed, file_bytes, filename, params, progress_cb=None) -> ModuleResult:
        sheet = load_sheet_for_enrichment(file_bytes)

        if "Full Text" not in sheet.col_index:
            raise BadExport("No 'Full Text' column found — run Mention Filler first, or upload an already-filled export.")

        available = detect_scored_entities(file_bytes, filename)
        if not available:
            raise BadExport(
                "No 'LLM Sentiment: <entity>' columns found — run Sentiment Coding (Multiple entities "
                "mode) first, either in this same pass or on the uploaded file."
            )

        all_rows = list(sheet.iter_rows())

        buckets = []
        if params["mode"] == "by_brand":
            for entity in available:
                buckets.append((entity, _filter_relevant(all_rows, entity)))
        else:
            owned = params.get("owned_entity", "").strip()
            matched = next((e for e in available if e.lower() == owned.lower()), None)
            if not matched:
                raise BadExport(
                    f"Owned brand '{owned}' doesn't match any detected entity ({', '.join(available)}) — "
                    f"check the spelling matches Sentiment Coding exactly."
                )
            competitors = [e for e in available if e != matched]
            buckets.append((matched, _filter_relevant(all_rows, matched)))
            buckets.append((REST_OF_FIELD_LABEL, _build_rest_of_field_units(all_rows, competitors)))

        total_units = sum(len(units) for _, units in buckets)
        cap = max_rows_per_run()
        if total_units > cap:
            raise CostCapExceeded(
                f"This run would send {total_units:,} mentions to OpenAI across all brands — above "
                f"the configured cap of {int(cap):,} (MAX_LLM_ROWS_PER_RUN in Secrets). Narrow the "
                f"entity list, or raise the cap if this run is intentional."
            )

        usage_totals = new_usage_totals()
        client = get_client()

        theme_sheet_rows = []
        monthly_sheet_rows = []
        summary_sheet_rows = []
        bucket_summaries = []
        total_no_date = 0
        n_buckets = len(buckets)
        for i, (label, units) in enumerate(buckets):
            def bucket_progress_cb(frac, message, _i=i):
                if progress_cb:
                    progress_cb((_i + frac) / n_buckets, message)

            theme_rows, monthly_rows, summary_text, n_tagged, n_failed, n_no_date = _analyze_bucket(
                client, label, units, params["n_themes"], params["sample_size"], params["threshold"],
                usage_totals, bucket_progress_cb,
            )
            theme_sheet_rows.extend(theme_rows)
            monthly_sheet_rows.extend(monthly_rows)
            total_no_date += n_no_date
            summary_sheet_rows.append((label, summary_text))
            bucket_summaries.append(
                f"{label}: {len(units):,} mentions, {len(theme_rows)} themes"
                + (f", {n_failed:,} tagging failures" if n_failed else "")
            )

        theme_rows_out = [[r[c] for c in THEME_SHEET_COLS] for r in theme_sheet_rows]
        summary_rows_out = [[entity, summary_text] for entity, summary_text in summary_sheet_rows]
        monthly_rows_out = [[r[c] for c in MONTHLY_SHEET_COLS] for r in monthly_sheet_rows]
        if progress_cb:
            progress_cb(0.99, "Writing output workbook...")
        out_bytes = sheet.to_bytes(extra_sheets=[
            ("Driver Analysis", THEME_SHEET_COLS, theme_rows_out),
            ("Driver Analysis Summary", ["Entity", "Summary"], summary_rows_out),
            ("Driver Analysis Monthly", MONTHLY_SHEET_COLS, monthly_rows_out),
        ])

        cost = usage_cost(usage_totals)
        summary_lines = bucket_summaries + [
            f"Actual cost: ${cost:,.4f} ({usage_totals['input']:,} input + {usage_totals['cached_input']:,} "
            f"cached + {usage_totals['output']:,} output tokens)",
            "See the 'Driver Analysis' sheet for the full theme breakdown, 'Driver Analysis Summary' "
            "for a narrative per brand, and 'Driver Analysis Monthly' for month-over-month trends.",
        ]
        if total_no_date:
            summary_lines.insert(-1, f"{total_no_date:,} tagged mentions lacked a valid Date and were "
                                       f"excluded from the monthly breakdown (still counted in the other sheets).")
        return ModuleResult(output_bytes=out_bytes, output_filename=filename, summary_lines=summary_lines)
