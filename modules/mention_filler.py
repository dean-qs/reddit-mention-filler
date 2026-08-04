"""Mention Filler — the v0 module. Fetches Date + Full Text (+ metadata) for
every URL in a Bulk Mentions export from the free Arctic Shift archive.

No live-scrape fallback in v0 (see README for why): URLs the archive doesn't
have are listed in a downloadable unmatched.csv instead of being auto-scraped.
"""
from pathlib import Path

from core.mentions_io import fill_export, unmatched_csv_bytes
from core.reddit_fetch import fetch_archive
from .base import AnalysisModule, Estimate, ModuleResult

ARCHIVE_RATE_PER_MIN = 1400  # matches the original pipeline's measured Arctic Shift throughput


def _fmt_duration(seconds):
    if seconds < 60:
        return "under a minute"
    minutes = seconds / 60
    if minutes < 1.5:
        return "~1 minute"
    return f"~{round(minutes)} minutes"


class MentionFillerModule(AnalysisModule):
    key = "mention_filler"
    label = "Mention Filler"
    description = "Fill in Date + Full Text for every Reddit URL in a Bulk Mentions export."

    def render_options(self, st, key_prefix):
        return {}

    def estimate(self, parsed, params) -> Estimate:
        n = len(parsed.urls)
        est_seconds = (n / ARCHIVE_RATE_PER_MIN) * 60
        return Estimate(
            headline=f"{n:,} mention URLs found",
            lines=[
                f"Estimated fetch time: {_fmt_duration(est_seconds)} "
                f"(Arctic Shift archive — free, no API key, ~{ARCHIVE_RATE_PER_MIN:,} URLs/min)",
                "No paid APIs are used by this module — estimated cost: $0.",
                "URLs the archive doesn't have (typically well under 1%) will be listed in a "
                "downloadable 'unmatched.csv' rather than live-scraped — see the README for why.",
            ],
            est_seconds=est_seconds,
            est_cost_usd=0.0,
        )

    def run(self, parsed, file_bytes, filename, params, progress_cb=None) -> ModuleResult:
        def on_progress(label, done, total, found):
            if progress_cb and total:
                progress_cb(done / total, f"Fetching {label} from archive: {done:,}/{total:,} ({found:,} found)")

        results = fetch_archive(parsed.urls, on_progress=on_progress)
        out_buf, stats = fill_export(parsed, results, file_bytes=file_bytes)

        stem = Path(filename).stem
        out_name = f"{stem} - filled.xlsx"

        summary_lines = [
            f"Filled {stats['filled']:,} of {stats['total']:,} rows",
            f"{stats['author_count']:,} unique authors — see the 'Author Rollup' sheet",
        ]
        if stats["truncated"]:
            summary_lines.append(
                f"{stats['truncated']:,} texts exceeded Excel's 32,767-char cell limit and were truncated"
            )
        extra_files = {}
        if stats["unmatched"]:
            reasons = {}
            for _, _, reason in stats["unmatched"]:
                reasons[reason] = reasons.get(reason, 0) + 1
            reason_str = ", ".join(f"{n_} {r}" for r, n_ in sorted(reasons.items(), key=lambda kv: -kv[1]))
            summary_lines.append(f"{len(stats['unmatched']):,} rows could not be filled ({reason_str})")
            extra_files["unmatched.csv"] = unmatched_csv_bytes(stats["unmatched"])

        return ModuleResult(
            output_bytes=out_buf.getvalue(),
            output_filename=out_name,
            summary_lines=summary_lines,
            extra_files=extra_files,
        )
