"""Common interfaces every analysis module implements.

A module is deliberately narrow: given the parsed export (and the original
file bytes, needed for the xlsx in-place-edit path), draw its own options,
report what it's about to do (count + time + $ cost) before the run, and
produce an output workbook. The Streamlit shell (app.py) only ever talks to
modules through these interfaces, so adding another module later is a new
file + one line in MODULES, not a rewrite of app.py.

Two flavors:
  - AnalysisModule: the general case (e.g. Mention Filler).
  - LLMEnrichmentModule: a module whose per-row work is "ask an LLM for some
    JSON fields". Any number of these selected in the same run are merged by
    core/llm_enrichment.py into ONE chat completion call per row instead of
    one call per module per row — see that file for how the merge works.
"""
from dataclasses import dataclass, field


@dataclass
class Estimate:
    headline: str
    lines: list
    est_seconds: float = None
    est_cost_usd: float = None


@dataclass
class ModuleResult:
    output_bytes: bytes
    output_filename: str
    summary_lines: list
    extra_files: dict = field(default_factory=dict)  # filename -> bytes


class AnalysisModule:
    key = "base"
    label = "Base module"
    description = ""
    uses_paid_api = False  # override to True on any module that spends real $ (for app.py's badge)

    def render_options(self, st, key_prefix, parsed=None, file_bytes=None, filename=None):
        """Draw this module's Streamlit widgets and return a params dict.
        parsed/file_bytes/filename describe the currently-uploaded export, for
        modules that need to preview/scan actual data while rendering options
        (e.g. discovering entities under a Brandwatch parent category)."""
        return {}

    def estimate(self, parsed, params, context) -> Estimate:
        """context carries cross-module info, e.g. context['text_will_be_filled']
        is True when Mention Filler is also selected (and runs first) this session,
        so a text-length-dependent estimate isn't misled by still-placeholder text."""
        raise NotImplementedError

    def run(self, parsed, file_bytes, filename, params, progress_cb=None) -> ModuleResult:
        """progress_cb(fraction: float, message: str), if given, is called periodically."""
        raise NotImplementedError


class LLMEnrichmentModule(AnalysisModule):
    """A module whose run() is entirely "send some row context to an LLM, get
    JSON fields back". Implement these instead of run() directly — app.py
    detects LLMEnrichmentModule instances and routes them through
    core/llm_enrichment.py's combined-call orchestrator.
    """
    uses_paid_api = True

    def output_columns(self, params) -> list:
        """New column names (in order) this module writes."""
        raise NotImplementedError

    def system_prompt_fragment(self, params) -> str:
        """Static instructions for this module, merged into one shared system prompt."""
        raise NotImplementedError

    def json_schema_fragment(self, row: dict, params) -> dict:
        """{'properties': {...}, 'required': [...]} merged into that row's JSON
        schema. The schema is rebuilt per row, not once for the whole run, so a
        module can ask for different (or zero) fields depending on the row —
        e.g. multi-entity Sentiment Coding only requests entities its regex
        prefilter found in THIS row's text. Return {} / no properties to opt
        this module out of this row entirely (costs nothing for it that row).
        Modules that don't need row-dependent schemas just ignore `row`."""
        raise NotImplementedError

    def row_context(self, row: dict, params) -> dict:
        """row is a dict of {column_name: value} for one data row (every column in
        the sheet, not just the ones this module cares about). Return the subset
        (as plain strings) this module wants in the per-row user message."""
        raise NotImplementedError

    def columns_from_result(self, row: dict, result: dict, params) -> dict:
        """result is this row's parsed JSON (all modules' fields together — pick
        out your own), or None if the row was never sent to the API (every
        module opted out via an empty json_schema_fragment that row). Return
        {column_name: value} to write — including sensible defaults for the
        result-is-None / my-fields-weren't-requested-this-row cases."""
        raise NotImplementedError

    def estimate_tokens_per_row(self, parsed, params, context) -> tuple:
        """Rough (input_tokens, output_tokens) per row, for the pre-run $ estimate."""
        raise NotImplementedError

    def build_extra_sheets(self, wb, params, row_values) -> None:
        """Optional: called once after the main per-row writing loop, with
        row_values = [(row_num, row, {this module's output_columns: value}), ...]
        in original row order (`row` is the original column_name->value dict,
        e.g. for pulling Url as a row identifier). Append any aggregate/summary
        sheets straight to `wb` (e.g. a breakdown of counts per entity) —
        default does nothing."""
        pass

    # LLMEnrichmentModule never implements run() itself — core/llm_enrichment.py
    # calls the methods above directly instead.
