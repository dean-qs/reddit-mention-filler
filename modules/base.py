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

    def render_options(self, st, key_prefix):
        """Draw this module's Streamlit widgets and return a params dict."""
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

    def output_columns(self, params) -> list:
        """New column names (in order) this module writes."""
        raise NotImplementedError

    def system_prompt_fragment(self, params) -> str:
        """Static instructions for this module, merged into one shared system prompt."""
        raise NotImplementedError

    def json_schema_fragment(self, params) -> dict:
        """{'properties': {...}, 'required': [...]} merged into one shared JSON schema."""
        raise NotImplementedError

    def row_context(self, row: dict, params) -> dict:
        """row is a dict of {column_name: value} for one data row (every column in
        the sheet, not just the ones this module cares about). Return the subset
        (as plain strings) this module wants in the per-row user message."""
        raise NotImplementedError

    def columns_from_result(self, result: dict, params) -> dict:
        """result is this row's parsed JSON (all modules' fields together — pick
        out your own). Return {column_name: value} to write."""
        raise NotImplementedError

    def estimate_tokens_per_row(self, parsed, params, context) -> tuple:
        """Rough (input_tokens, output_tokens) per row, for the pre-run $ estimate."""
        raise NotImplementedError

    # LLMEnrichmentModule never implements run() itself — core/llm_enrichment.py
    # calls the methods above directly instead.
