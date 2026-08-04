"""Reddit Mention Filler — Streamlit front end.

Upload (or paste) a Bulk Mentions export, pick which analysis modules to run,
see an estimate (row count / time / cost) before anything executes, then run
and download the result. See modules/registry.py for how to add new modules.
"""
import io

import streamlit as st

from core.mentions_io import BadExport, parse_export
from modules.registry import MODULES

st.set_page_config(page_title="Reddit Mention Filler", page_icon="🧵", layout="centered")

st.title("🧵 Reddit Mention Filler")
st.caption(
    "Turn a Bulk Mentions export (Reddit rows with text withheld) into a filled copy — "
    "no Python, no installs. Upload the export, pick your modules, review the estimate, run."
)

with st.expander("What is this / how does it work?"):
    st.markdown(
        "Our social-listening tool's Bulk Mentions export withholds Reddit text "
        "(*\"Due to licensing restrictions, this mention cannot be downloaded\"*) and leaves "
        "Date empty. This tool looks up every URL in the free "
        "[Arctic Shift](https://arctic-shift.photon-reddit.com/) Reddit archive and fills in "
        "Date + Full Text (plus Score, Type, Edited, and linked-URL columns), and adds an "
        "Author Rollup sheet. Rows the archive doesn't have (usually well under 1%) are listed "
        "in a separate `unmatched.csv` rather than live-scraped."
    )

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
    st.header("2. Modules")
    selected = []
    module_params = {}
    for mod in MODULES:
        checked = st.checkbox(f"**{mod.label}** — {mod.description}", value=True, key=f"enable_{mod.key}")
        if checked:
            with st.container(border=True):
                module_params[mod.key] = mod.render_options(st, key_prefix=mod.key)
            selected.append(mod)

    # ---------------------------------------------------------------------
    # 3. Estimate
    # ---------------------------------------------------------------------
    if selected:
        st.header("3. Estimate")
        total_seconds = 0.0
        total_cost = 0.0
        for mod in selected:
            est = mod.estimate(parsed, module_params[mod.key])
            st.subheader(mod.label)
            st.write(f"**{est.headline}**")
            for line in est.lines:
                st.write(f"- {line}")
            total_seconds += est.est_seconds or 0.0
            total_cost += est.est_cost_usd or 0.0

        st.info(
            f"**Total estimate:** ~{max(1, round(total_seconds / 60))} min · "
            f"**${total_cost:,.2f}** estimated cost"
        )

        # -------------------------------------------------------------
        # 4. Run
        # -------------------------------------------------------------
        st.header("4. Run")
        if st.button("Run", type="primary"):
            current_bytes, current_filename = file_bytes, filename
            all_summary = []
            all_extra_files = {}
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for mod in selected:
                current_parsed = parse_export(current_bytes, current_filename) if current_bytes is not file_bytes else parsed

                def progress_cb(frac, message, _mod=mod):
                    progress_bar.progress(min(1.0, max(0.0, frac)))
                    status_text.write(f"**{_mod.label}:** {message}")

                result = mod.run(current_parsed, current_bytes, current_filename, module_params[mod.key], progress_cb=progress_cb)
                current_bytes, current_filename = result.output_bytes, result.output_filename
                all_summary.append((mod.label, result.summary_lines))
                all_extra_files.update(result.extra_files)

            progress_bar.progress(1.0)
            status_text.write("Done.")

            st.success("Finished — see results below.")
            for label, lines in all_summary:
                st.write(f"**{label}**")
                for line in lines:
                    st.write(f"- {line}")

            st.download_button(
                "⬇️ Download filled workbook",
                data=current_bytes,
                file_name=current_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            for extra_name, extra_bytes in all_extra_files.items():
                st.download_button(f"⬇️ Download {extra_name}", data=extra_bytes, file_name=extra_name, mime="text/csv")
    else:
        st.info("Pick at least one module above to see an estimate and run it.")
