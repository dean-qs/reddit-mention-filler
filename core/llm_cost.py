"""OpenAI usage/cost accounting shared by every module that calls the API
directly: core/llm_enrichment.py's combined-call coordinator, Theme Summary,
and Driver Analysis.
"""

# gpt-4o-mini, as of Aug 2026 — https://devtk.ai/en/models/gpt-4o-mini/
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_CACHED_INPUT = 0.075
PRICE_PER_1M_OUTPUT = 0.60


def new_usage_totals():
    return {"input": 0, "cached_input": 0, "output": 0}


def add_usage(totals, usage):
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    totals["cached_input"] += cached
    totals["input"] += max(0, (usage.prompt_tokens or 0) - cached)
    totals["output"] += usage.completion_tokens or 0


def usage_cost(totals):
    return (
        totals["input"] / 1_000_000 * PRICE_PER_1M_INPUT
        + totals["cached_input"] / 1_000_000 * PRICE_PER_1M_CACHED_INPUT
        + totals["output"] / 1_000_000 * PRICE_PER_1M_OUTPUT
    )
