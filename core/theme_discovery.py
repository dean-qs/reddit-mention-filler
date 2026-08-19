"""Shared theme-discovery step for Theme Summary and Driver Analysis: given a
sample of mentions, ask the LLM once for up to N recurring themes (name +
description). Tagging — how each mention relates to the discovered themes —
differs enough between the two modules (single-label for Theme Summary,
multi-label with per-theme sentiment for Driver Analysis) that it isn't
shared here.
"""
from core.llm_client import call_json
from core.llm_cost import add_usage

SAMPLE_TEXT_CHARS = 300  # per-row truncation for the discovery prompt


def discovery_schema(n_themes):
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


def discover_themes(client, system_prompt, sample_rows, n_themes, usage_totals):
    """sample_rows: list of {column_name: value} dicts (must have 'Full Text').
    Adds token usage to usage_totals and returns (theme_names, theme_descriptions).
    Raises RuntimeError if the model returns no themes at all.
    """
    sample_lines = [
        f"{i + 1}. {str(row.get('Full Text') or '')[:SAMPLE_TEXT_CHARS]}" for i, row in enumerate(sample_rows)
    ]
    user_message = "\n".join(sample_lines)
    result, usage = call_json(client, discovery_schema(n_themes), system_prompt, user_message)
    add_usage(usage_totals, usage)
    themes = result.get("themes") or []
    if not themes:
        raise RuntimeError("Theme discovery returned no themes — try a larger sample or fewer/more themes.")
    theme_names = [t["name"] for t in themes]
    theme_descriptions = {t["name"]: t.get("description", "") for t in themes}
    return theme_names, theme_descriptions
