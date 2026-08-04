"""Common interface every analysis module implements.

A module is deliberately narrow: given the parsed export (and the original
file bytes, needed for the xlsx in-place-edit path), draw its own options,
report what it's about to do (count + time + $ cost) before the run, and
produce an output workbook. The Streamlit shell (app.py) only ever talks to
modules through this interface, so adding sentiment recoding or geolocation
later is a new file + one line in MODULES, not a rewrite of app.py.
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

    def estimate(self, parsed, params) -> Estimate:
        raise NotImplementedError

    def run(self, parsed, file_bytes, filename, params, progress_cb=None) -> ModuleResult:
        """progress_cb(fraction: float, message: str), if given, is called periodically."""
        raise NotImplementedError
