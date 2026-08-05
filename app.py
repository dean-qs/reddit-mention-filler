"""Reddit Mention Filler — Streamlit front end.

Gated behind a @quadstrat.com email check (core/access_gate.py) since this
can spend real OpenAI credits. Upload (or paste) a Bulk Mentions export, pick
which analysis modules to run, see an estimate (row count / time / cost)
before anything executes, then run and download the result. See
modules/registry.py for how to add new modules.

Modules come in two flavors (see modules/base.py): plain AnalysisModules
(Mention Filler, Theme Summary — each manages its own run however it needs
to) and LLMEnrichmentModules (Sentiment Coding, Geolocation) — any number of
those selected together are merged into ONE LLM call per row by
core/llm_enrichment.py, instead of one call per module per row.
"""
import streamlit as st

from core.access_gate import require_quadstrat_email
from core.cost_caps import CostCapExceeded, max_cost_usd_per_run
from core.llm_client import MissingApiKey
from core.llm_enrichment import run_llm_modules
from core.mentions_io import BadExport, parse_export
from modules.base import LLMEnrichmentModule
from modules.registry import MODULES

st.set_page_config(page_title="Reddit Mention Filler", page_icon="🧵", layout="centered")
require_quadstrat_email(st)

st.title("🧵 Reddit Mention Filler")
st.caption("Quadrant Strategies")
st.write(
    "Turn a Bulk Mentions export (Reddit rows with text withheld) into a filled, enriched copy — "
    "no Python, no installs. Upload the export, pick your modules, review the estimate, run."
)

with st.expander("What is this / how does it work?"):
    st.markdown(
        "Our social-listening tool's Bulk Mentions export withholds Reddit text "
        "(*\"Due to licensing restrictions, this mention cannot be downloaded\"*) and leaves "
        "Date empty. **Mention Filler** looks up every URL in the free "
        "[Arctic Shift](https://arctic-shift.photon-reddit.com/) Reddit archive and fills in "
        "Date + Full Text (plus Score, Type, Edited, and linked-URL columns), and adds an "
        "Author Rollup sheet. Rows the archive doesn't have (usually well under 1%) are listed "
        "in a separate `unmatched.csv` rather than live-scraped.\n\n"
        "**Sentiment Coding**, **Geolocation**, and **Theme Summary** are optional LLM-powered "
        "add-ons that run on top of the filled text (OpenAI gpt-4o-mini) — each shows its own "
        "estimated cost before you run it, and the real cost from actual usage afterward. All "
        "three need Full Text already filled in, so run Mention Filler first (in the same pass, "
        "or on an already-filled file). Sentiment Coding can score general tone, sentiment toward "
        "one or more named entities (by alias list or Brandwatch parent category), or a fully "
        "custom prompt.\n\n"
        "This tool can spend real OpenAI credits, so it's gated to Quadrant emails, and a "
        "row-count / cost cap in Secrets backstops accidental huge runs."
    )

st.divider()

# ---------------------------------------------------------------------------
# 1. Input
# ---------------------------------------------------------------------------
st.header("1. Input")
input_mode = st.radio("Provide your export by:", ["Upload a file", "Paste CSV text"], horizontal=True)

file_bytes = None
filename = None

if input_mode == "Upload a file":
    uploaded = st.file_uploader("Bulk Mentions export (.xlsx or .csv)", type=["xlsx", "csv"])
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        filename = uploaded.name
else:
    pasted = st.text_area(
        "Paste the export's raw CSV text (including its header row)",
        height=180,
        placeholder="Query Id,Date,Url,Full Text,Author,Title,...",
    )
    if pasted.strip():
        file_bytes = pasted.encode("utf-8")
        filename = "pasted_export.csv"

parsed = None
if file_bytes is not None:
    try:
        parsed = parse_export(file_bytes, filename)
    except BadExport as e:
        st.error(str(e))

# ---------------------------------------------------------------------------
# 2. Modules
# ---------------------------------------------------------------------------
if parsed is not None:
    st.divider()
    st.header("2. Modules")
    selected = []
    module_params = {}
    for mod in MODULES:
        badge = "💵 paid (OpenAI)" if mod.uses_paid_api else "🟢 free"
        checked = st.checkbox(
            f"**{mod.label}** · _{badge}_ — {mod.description}",
            value=(mod.key == "mention_filler"),
            key=f"enable_{mod.key}",
        )
        if checked:
            with st.container(border=True):
                module_params[mod.key] = mod.render_options(
                    st, key_prefix=mod.key, parsed=parsed, file_bytes=file_bytes, filename=filename
                )
            selected.append(mod)

    run_context = {"text_will_be_filled": any(m.key == "mention_filler" for m in selected)}

    # ---------------------------------------------------------------------
    # 3. Estimate
    # ---------------------------------------------------------------------
    if selected:
        st.divider()
        st.header("3. Estimate")
        total_seconds = 0.0
        total_cost = 0.0
        has_llm = any(isinstance(m, LLMEnrichmentModule) for m in selected)
        for mod in selected:
            est = mod.estimate(parsed, module_params[mod.key], run_context)
            with st.container(border=True):
                st.subheader(mod.label)
                st.write(f"**{est.headline}**")
                for line in est.lines:
                    st.write(f"- {line}")
            total_seconds += est.est_seconds or 0.0
            total_cost += est.est_cost_usd or 0.0

        m1, m2 = st.columns(2)
        m1.metric("Estimated time", f"~{max(1, round(total_seconds / 60))} min")
        m2.metric("Estimated cost", f"${total_cost:,.2f}")
        if has_llm:
            st.caption(
                "LLM modules run as one combined call per row, so actual cost (shown after the run) "
                "is usually a bit lower than this sum."
            )

        cost_cap = max_cost_usd_per_run()
        cap_exceeded = has_llm and total_cost > cost_cap
        if cap_exceeded:
            st.error(
                f"Estimated cost (${total_cost:,.2f}) is above this app's configured cap "
                f"(${cost_cap:,.2f} — MAX_LLM_COST_USD_PER_RUN in Secrets). Reduce scope, or raise "
                f"the cap in Secrets if this run is intentional."
            )

        # -------------------------------------------------------------
        # 4. Run
        # -------------------------------------------------------------
        st.divider()
        st.header("4. Run")
        if st.button("Run", type="primary", disabled=cap_exceeded):
            current_bytes, current_filename = file_bytes, filename
            all_summary = []
            all_extra_files = {}
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                i = 0
                while i < len(selected):
                    mod = selected[i]
                    if isinstance(mod, LLMEnrichmentModule):
                        batch = []
                        while i < len(selected) and isinstance(selected[i], LLMEnrichmentModule):
                            batch.append(selected[i])
                            i += 1

                        def llm_progress_cb(frac, message):
                            progress_bar.progress(min(1.0, max(0.0, frac)))
                            status_text.write(message)

                        out_bytes, out_name, cost_summary = run_llm_modules(
                            batch, module_params, current_bytes, current_filename, progress_cb=llm_progress_cb
                        )
                        current_bytes, current_filename = out_bytes, out_name
                        lines = [f"{cost_summary['n_rows']:,} rows enriched"]
                        if cost_summary["n_skipped"]:
                            lines[0] += f", {cost_summary['n_skipped']:,} skipped (Full Text not filled)"
                        if cost_summary.get("n_no_signal"):
                            lines[0] += f", {cost_summary['n_no_signal']:,} skipped (no configured entity detected — free)"
                        if cost_summary["n_failed"]:
                            lines[0] += f", {cost_summary['n_failed']:,} failed after retries"
                        lines.append(
                            f"Actual cost: ${cost_summary['cost_usd']:,.4f} "
                            f"({cost_summary['input']:,} input + {cost_summary['cached_input']:,} cached + "
                            f"{cost_summary['output']:,} output tokens)"
                        )
                        all_summary.append((" + ".join(m.label for m in batch), lines))
                    else:
                        current_parsed = parse_export(current_bytes, current_filename) if current_bytes is not file_bytes else parsed

                        def module_progress_cb(frac, message, _mod=mod):
                            progress_bar.progress(min(1.0, max(0.0, frac)))
                            status_text.write(f"**{_mod.label}:** {message}")

                        result = mod.run(current_parsed, current_bytes, current_filename, module_params[mod.key], progress_cb=module_progress_cb)
                        current_bytes, current_filename = result.output_bytes, result.output_filename
                        all_summary.append((mod.label, result.summary_lines))
                        all_extra_files.update(result.extra_files)
                        i += 1
            except MissingApiKey as e:
                st.error(str(e))
                st.stop()
            except CostCapExceeded as e:
                st.error(str(e))
                st.stop()
            except BadExport as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"The run failed: {e}")
                st.stop()

            progress_bar.progress(1.0)
            status_text.write("Done.")

            st.success("Finished — see results below.")
            for label, lines in all_summary:
                with st.container(border=True):
                    st.write(f"**{label}**")
                    for line in lines:
                        st.write(f"- {line}")

            st.download_button(
                "⬇️ Download workbook",
                data=current_bytes,
                file_name=current_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            for extra_name, extra_bytes in all_extra_files.items():
                st.download_button(f"⬇️ Download {extra_name}", data=extra_bytes, file_name=extra_name, mime="text/csv")
    else:
        st.info("Pick at least one module above to see an estimate and run it.")
