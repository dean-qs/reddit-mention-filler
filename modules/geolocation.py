"""Geolocation (beta) — an LLM enrichment module.

Estimates the likely country (and US state/region when there's real
evidence) of the mention's author, from the subreddit, post title, and full
text. A handful of regex-based signals (British-vs-American spelling tells,
a small subreddit -> country hint table — see core/geo_signals.py) are
computed first and handed to the LLM as *evidence*, not a verdict, so one
combined model call can weigh text content, subreddit, and these hints
together rather than bolting on a separate statistical adjustment pass.
"""
from core.geo_signals import detect_signals
from core.text_utils import rough_token_estimate, subreddit_from_url
from .base import Estimate, LLMEnrichmentModule

MAX_CHARS = 4000

DEFAULT_INSTRUCTIONS = (
    "Estimate the likely country the mention's author is posting from, using "
    "the subreddit, post title, full text, and the detected signals below as "
    "evidence. Prefer the actual content of the text over the signals when "
    "they conflict — the signals are hints, not proof.\n\n"
    "Give a specific US state or region in 'Geo - Region' ONLY when the "
    "country is the United States AND there's real textual evidence for a "
    "specific state/region (e.g. a named city, sports team, local reference). "
    "Otherwise leave 'Geo - Region' blank. Use 'Unknown' for 'Geo - Country' "
    "if there truly isn't enough evidence to guess — don't force a country "
    "when the mention gives nothing to go on."
)

OUTPUT_INSTRUCTIONS = (
    "\n\nRespond with: 'Geo - Country' (a country name, or 'Unknown'), "
    "'Geo - Region' (a US state/region name, or '' if not applicable/not confident), "
    "'Geo - Confidence' (High, Medium, or Low — be honest; most single Reddit "
    "mentions warrant Medium or Low), and 'Geo - Rationale' (a short clause, "
    "under 15 words, naming the key evidence used)."
)


class GeolocationModule(LLMEnrichmentModule):
    key = "geolocation"
    label = "Geolocation (beta)"
    description = "Estimate the author's likely country / US region from subreddit, title, and text."

    def render_options(self, st, key_prefix):
        custom = ""
        with st.expander("Instructions the LLM will get (editable)"):
            custom = st.text_area(
                "Geolocation instructions",
                value=DEFAULT_INSTRUCTIONS,
                height=160,
                key=f"{key_prefix}_instructions",
                help="The Country/Region/Confidence/Rationale output format below this is always enforced.",
            )
        return {"instructions": custom.strip() or DEFAULT_INSTRUCTIONS}

    def output_columns(self, params):
        return ["Geo - Country", "Geo - Region", "Geo - Confidence", "Geo - Rationale"]

    def system_prompt_fragment(self, params):
        return params["instructions"] + OUTPUT_INSTRUCTIONS

    def json_schema_fragment(self, params):
        return {
            "properties": {
                "Geo - Country": {"type": "string"},
                "Geo - Region": {"type": "string"},
                "Geo - Confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
                "Geo - Rationale": {"type": "string"},
            },
            "required": ["Geo - Country", "Geo - Region", "Geo - Confidence", "Geo - Rationale"],
        }

    def row_context(self, row, params):
        subreddit = subreddit_from_url(row.get("Url"))
        title = row.get("Title") or ""
        text = str(row.get("Full Text") or "")[:MAX_CHARS]
        signals = detect_signals(subreddit, title, text)
        return {
            "Subreddit": f"r/{subreddit}" if subreddit else "(unknown)",
            "Post Title": title,
            "Full Text": text,
            "Detected signals (evidence only, not ground truth)": "; ".join(signals) if signals else "none detected",
        }

    def columns_from_result(self, result, params):
        return {
            "Geo - Country": result.get("Geo - Country", ""),
            "Geo - Region": result.get("Geo - Region", ""),
            "Geo - Confidence": result.get("Geo - Confidence", ""),
            "Geo - Rationale": result.get("Geo - Rationale", ""),
        }

    def estimate_tokens_per_row(self, parsed, params, context):
        system_tokens = rough_token_estimate(self.system_prompt_fragment(params)) + 40
        assumed_text_tokens = 260  # text + title + subreddit + signal list
        return system_tokens + assumed_text_tokens, 40

    def estimate(self, parsed, params, context) -> Estimate:
        from core.llm_enrichment import PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT

        in_tok, out_tok = self.estimate_tokens_per_row(parsed, params, context)
        n = len(parsed.urls)
        cost = (in_tok * n / 1_000_000) * PRICE_PER_1M_INPUT + (out_tok * n / 1_000_000) * PRICE_PER_1M_OUTPUT
        lines = [
            f"Uses OpenAI gpt-4o-mini — estimated cost: ${cost:,.2f} for {n:,} rows "
            f"(~{in_tok:,} input + {out_tok:,} output tokens/row, rough).",
            "Beta: single-mention geolocation is inherently uncertain — expect a lot of "
            "Medium/Low confidence and 'Unknown' results, by design rather than a bug.",
            "Real cost is computed from actual token usage after the run and shown in the results.",
        ]
        if not context.get("text_will_be_filled"):
            lines.insert(0, "⚠️ Full Text doesn't look filled yet in this file — run Mention Filler "
                             "first (in this same run, or upload an already-filled export) or this "
                             "module will skip every row.")
        return Estimate(
            headline=f"Geolocation for {n:,} rows",
            lines=lines,
            est_cost_usd=cost,
        )
