"""Reddit Mention Filler — Streamlit front end.

Gated behind a @quadstrat.com email check (core/access_gate.py) since this
can spend real OpenAI credits. Upload (or paste) a Bulk Mentions export, pick
which analysis modules to run, see an estimate (row count / time / cost)
before anything executes, then run and download the result. See
modules/registry.py for how to add new modules.

Modules come in two flavors (see modules/base.py): plain AnalysisModules
(Mention Filler, Theme Summary, Driver Analysis — each manages its own run
however it needs to) and LLMEnrichmentModules (Sentiment Coding,
Geolocation) — any number of those selected together are merged into ONE
LLM call per row by core/llm_enrichment.py, instead of one call per module
per row.
"""
import streamlit as st

from core.access_gate import require_quadstrat_email
from core.cost_caps import CostCapExceeded, max_cost_usd_per_run
from core.llm_client import MissingApiKey
from core.llm_enrichment import run_llm_modules
from core.mentions_io import BadExport, merge_exports, parse_export
import core.run_storage as run_storage
from modules.base import LLMEnrichmentModule
from modules.registry import MODULES

st.set_page_config(page_title="QS Reddit Mention Filler", page_icon="🧵", layout="centered")
require_quadstrat_email(st)


@st.cache_data(show_spinner="Reading file...")
def _cached_parse_export(file_bytes, filename):
    # Streamlit reruns this whole script on every widget interaction (a
    # checkbox, a slider drag) — without caching, a large file (tens of
    # thousands of rows) would get fully re-read on every single one of
    # those, not just on upload.
    return parse_export(file_bytes, filename)


@st.cache_data(show_spinner="Merging files...")
def _cached_merge_exports(files):
    return merge_exports(files)

st.title("🧵 QS Reddit Mention Filler")
st.caption("Team Digital")
st.write(
    "Turn a Bulk Mentions export (Reddit rows with text withheld) into a filled, enriched copy "
    "without having to install Python or anything! Upload the BW export, pick how you want to process it, "
    "review the estimate, and then you should be good to run :)"
)

with st.expander("What is this / how does it work?"):
    st.markdown(
        "Our social-listening tool's Bulk Mentions export withholds Reddit text "
        "(*\"Due to licensing restrictions, this mention cannot be downloaded\"*) and leaves "
        "Date and Author empty. This **Mention Filler** looks up every URL in the free "
        "[Arctic Shift](https://arctic-shift.photon-reddit.com/) Reddit archive and fills in "
        "Date, Author, and Full Text (plus Score, whether or not the post was edited, and any sites it links to), and adds an "
        "Author Rollup sheet so we can see the most vocal authors. Rows that Arctic Shift doesn't have (usually well under 1%) are listed "
        "in a separate `unmatched.csv` rather than live-scraped.\n\n"
        "**Sentiment Coding**, **Geolocation**, **Theme Summary**, and **Driver / Barrier Analysis** "
        "are optional LLM-powered add-ons that run on top of the filled text (OpenAI gpt-4o-mini) — "
        "each shows its own estimated cost before you run it, and the real cost from actual usage "
        "afterward. All need Full Text already filled in, so run Mention Filler first (in the same "
        "pass, or on an already-filled file). Sentiment Coding can score general tone, sentiment "
        "toward one or more named entities (by alias list or Brandwatch parent category), or a fully "
        "custom prompt. Driver / Barrier Analysis builds on Sentiment Coding's multi-entity output to "
        "find what themes drive positive vs. negative sentiment toward each brand.\n\n"
        "This tool can spend real OpenAI credits, so it's gated to Quadrant emails, and a "
        "row-count / cost cap in Secrets backstops accidental huge runs."
    )

# Reads straight from disk on this container — not st.session_state — so it
# still shows up after a dropped connection forces a brand-new browser
# session (session_state resets then; this doesn't). Doesn't survive an app
# reboot/redeploy though, so still download promptly — this is a rescue
# path, not storage. See core/run_storage.py.
recent_runs = run_storage.list_runs()
if recent_runs:
    with st.expander(
        f"🗂️ Recent runs ({len(recent_runs)}) — backup downloads in case a download failed or "
        f"the connection dropped. Not permanent — grab what you need soon."
    ):
        for run in recent_runs:
            with st.container(border=True):
                st.write(f"**{run['filename']}** — {run['saved_at']}")
                for line in run.get("summary_lines", []):
                    st.caption(f"- {line}")
                main_bytes = run_storage.load_file(run["dir"], run["filename"])
                if main_bytes:
                    st.download_button(
                        f"⬇️ Download {run['filename']}", data=main_bytes, file_name=run["filename"],
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"recent_{run['dir'].name}_main",
                    )
                for extra_name in run.get("extra_files", []):
                    extra_bytes = run_storage.load_file(run["dir"], extra_name)
                    if extra_bytes:
                        st.download_button(
                            f"⬇️ Download {extra_name}", data=extra_bytes, file_name=extra_name,
                            mime="text/csv", key=f"recent_{run['dir'].name}_{extra_name}",
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
    uploaded = st.file_uploader(
        "Bulk Mentions export (.xlsx or .csv) — upload more than one to concatenate + dedupe by Url",
        type=["xlsx", "csv"], accept_multiple_files=True,
    )
    if uploaded:
        if len(uploaded) == 1:
            file_bytes = uploaded[0].getvalue()
            filename = uploaded[0].name
        else:
            files = [(f.getvalue(), f.name) for f in uploaded]
            try:
                file_bytes, merge_stats = _cached_merge_exports(files)
                filename = f"{merge_stats['n_files']} files merged.xlsx"
                per_file = ", ".join(f"{name}: {n:,} rows" for name, n in merge_stats["per_file_counts"])
                st.info(
                    f"Merged {merge_stats['n_files']} files ({per_file}) into "
                    f"{merge_stats['n_merged_rows']:,} rows — {merge_stats['n_duplicates']:,} duplicate "
                    f"Urls dropped (first occurrence kept)."
                )
            except BadExport as e:
                st.error(str(e))
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
        parsed = _cached_parse_export(file_bytes, filename)
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

        # The whole results section (below) used to live inside this `if
        # st.button("Run"):` block, which is only True on the exact script
        # run where Run was clicked — any later rerun (including the
        # download button's OWN click) re-runs with that False again, so
        # the results and the download button itself would vanish. Stash
        # the result in session_state and render it from there instead, so
        # it survives reruns. source_key clears it out when the upload
        # changes, so a stale download doesn't linger next to a new file.
        current_source_key = (filename, len(file_bytes) if file_bytes else None)
        if st.session_state.get("run_result", {}).get("source_key") != current_source_key:
            st.session_state.pop("run_result", None)

        if st.button("Run", type="primary", disabled=cap_exceeded):
            current_bytes, current_filename = file_bytes, filename
            all_summary = []
            all_extra_files = {}
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            # One id for this whole Run click — every checkpoint below writes
            # into the SAME on-disk run directory (overwriting the previous
            # module's snapshot), so if the connection drops mid-pipeline,
            # whatever finished before that point is still recoverable from
            # the "Recent runs" section on a fresh page load, not just a
            # fully-successful run. Best-effort: a storage hiccup must never
            # break the actual run, so every call is wrapped and swallowed.
            run_id = run_storage.new_run_id()

            def checkpoint():
                try:
                    run_storage.save_run(
                        run_id, current_filename, current_bytes, all_extra_files,
                        [line for _, lines in all_summary for line in lines],
                    )
                except Exception:
                    pass

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
                        checkpoint()
                    else:
                        current_parsed = _cached_parse_export(current_bytes, current_filename) if current_bytes is not file_bytes else parsed

                        def module_progress_cb(frac, message, _mod=mod):
                            progress_bar.progress(min(1.0, max(0.0, frac)))
                            status_text.write(f"**{_mod.label}:** {message}")

                        result = mod.run(current_parsed, current_bytes, current_filename, module_params[mod.key], progress_cb=module_progress_cb)
                        current_bytes, current_filename = result.output_bytes, result.output_filename
                        all_summary.append((mod.label, result.summary_lines))
                        all_extra_files.update(result.extra_files)
                        checkpoint()
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
                if all_summary:
                    st.caption(
                        "Whatever finished before this failure was checkpointed — reload the page and "
                        "check '🗂️ Recent runs' near the top to recover it."
                    )
                st.stop()

            progress_bar.progress(1.0)
            status_text.write("Done.")

            st.session_state["run_result"] = {
                "bytes": current_bytes,
                "filename": current_filename,
                "summary": all_summary,
                "extra_files": all_extra_files,
                "source_key": current_source_key,
            }

        run_result = st.session_state.get("run_result")
        if run_result:
            st.success("Finished — see results below.")
            for label, lines in run_result["summary"]:
                with st.container(border=True):
                    st.write(f"**{label}**")
                    for line in lines:
                        st.write(f"- {line}")

            st.download_button(
                "⬇️ Download workbook",
                data=run_result["bytes"],
                file_name=run_result["filename"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_main",
            )
            for extra_name, extra_bytes in run_result["extra_files"].items():
                st.download_button(
                    f"⬇️ Download {extra_name}", data=extra_bytes, file_name=extra_name, mime="text/csv",
                    key=f"download_{extra_name}",
                )
    else:
        st.info("Pick at least one module above to see an estimate and run it.")
