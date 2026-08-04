"""Sentiment Coding — an LLM enrichment module.

Three modes, similar in spirit to the team's Brandwatch sentiment-coder
workflow: general overall tone, sentiment toward a specific entity (or a
few, comma-separated) typed in inline, or a fully custom prompt for
anything else. Runs combined with Geolocation (if also selected) as one
call per row — see core/llm_enrichment.py.
"""
from core.text_utils import rough_token_estimate
from .base import Estimate, LLMEnrichmentModule

MAX_CHARS = 4000  # bounds worst-case per-row cost on unusually long comments

GENERAL_PROMPT = (
    "Classify the overall sentiment expressed in the mention below as Positive, "
    "Neutral, or Negative. Judge the tone of the mention itself, not the "
    "subject matter it discusses."
)

ENTITY_PROMPT_TEMPLATE = (
    "Classify the sentiment expressed toward {entity} specifically — not the "
    "mention's overall tone. If {entity} is not actually mentioned or the "
    "mention is neutral/off-topic with respect to it, use Neutral."
)

OUTPUT_INSTRUCTIONS = (
    "\n\nRespond with 'Sentiment' as exactly one of: Positive, Neutral, Negative. "
    "Also give 'Sentiment Rationale': a short clause (under 15 words) explaining why."
)


class SentimentModule(LLMEnrichmentModule):
    key = "sentiment"
    label = "Sentiment Coding"
    description = "Code each mention's sentiment (Positive/Neutral/Negative) with an LLM."

    def render_options(self, st, key_prefix):
        mode = st.radio(
            "What should sentiment be scored against?",
            ["General (overall tone)", "Toward a specific entity", "Custom prompt"],
            key=f"{key_prefix}_mode",
        )
        entity = ""
        custom_prompt = ""
        if mode == "Toward a specific entity":
            entity = st.text_input(
                "Entity (or comma-separated entities)",
                placeholder="e.g. Google, or Google, the FTC",
                key=f"{key_prefix}_entity",
            )
        default_preview = GENERAL_PROMPT if mode != "Toward a specific entity" else ENTITY_PROMPT_TEMPLATE.format(entity=entity or "[entity]")
        if mode == "Custom prompt":
            custom_prompt = st.text_area(
                "Custom sentiment instructions",
                value=default_preview,
                height=100,
                key=f"{key_prefix}_custom",
                help="Replace this with whatever sentiment instructions you need. The "
                     "Positive/Neutral/Negative + rationale output format is still enforced.",
            )
        else:
            with st.expander("Preview the instructions the LLM will get"):
                st.code(default_preview, language=None)
        return {"mode": mode, "entity": entity.strip(), "custom_prompt": custom_prompt.strip()}

    def _instructions(self, params):
        if params["mode"] == "Custom prompt" and params["custom_prompt"]:
            return params["custom_prompt"]
        if params["mode"] == "Toward a specific entity" and params["entity"]:
            return ENTITY_PROMPT_TEMPLATE.format(entity=params["entity"])
        return GENERAL_PROMPT

    def output_columns(self, params):
        return ["Sentiment", "Sentiment Rationale"]

    def system_prompt_fragment(self, params):
        return self._instructions(params) + OUTPUT_INSTRUCTIONS

    def json_schema_fragment(self, params):
        return {
            "properties": {
                "Sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
                "Sentiment Rationale": {"type": "string"},
            },
            "required": ["Sentiment", "Sentiment Rationale"],
        }

    def row_context(self, row, params):
        text = str(row.get("Full Text") or "")[:MAX_CHARS]
        return {"Post Title": row.get("Title") or "", "Full Text": text}

    def columns_from_result(self, result, params):
        return {
            "Sentiment": result.get("Sentiment", ""),
            "Sentiment Rationale": result.get("Sentiment Rationale", ""),
        }

    def estimate_tokens_per_row(self, parsed, params, context):
        system_tokens = rough_token_estimate(self.system_prompt_fragment(params)) + 40
        assumed_text_tokens = 220  # rough average Reddit mention length
        return system_tokens + assumed_text_tokens, 30

    def estimate(self, parsed, params, context) -> Estimate:
        from core.llm_enrichment import PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT

        in_tok, out_tok = self.estimate_tokens_per_row(parsed, params, context)
        n = len(parsed.urls)
        cost = (in_tok * n / 1_000_000) * PRICE_PER_1M_INPUT + (out_tok * n / 1_000_000) * PRICE_PER_1M_OUTPUT
        lines = [
            f"Uses OpenAI gpt-4o-mini — estimated cost: ${cost:,.2f} for {n:,} rows "
            f"(~{in_tok:,} input + {out_tok:,} output tokens/row, rough).",
            "Real cost is computed from actual token usage after the run and shown in the results.",
        ]
        if not context.get("text_will_be_filled"):
            lines.insert(0, "⚠️ Full Text doesn't look filled yet in this file — run Mention Filler "
                             "first (in this same run, or upload an already-filled export) or this "
                             "module will skip every row.")
        return Estimate(
            headline=f"Sentiment coding for {n:,} rows ({params.get('mode', 'General')})",
            lines=lines,
            est_cost_usd=cost,
        )
