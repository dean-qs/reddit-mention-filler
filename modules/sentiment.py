"""Sentiment Coding — an LLM enrichment module.

Four modes:
  - General (overall tone)
  - Toward a specific entity (typed inline)
  - Multiple entities — recall-biased regex aliases OR a Brandwatch parent
    category, mirroring children_safety_classifier_v2.py / stage1_parse.py.
    Each row only asks the LLM about entities its regex/category prefilter
    actually found in that row (a per-row-dynamic JSON schema — see
    core/llm_enrichment.py); entities not detected are written "Not Mentioned"
    at zero extra LLM cost rather than asked-and-answered-empty.
  - Custom prompt

Runs combined with Geolocation (if also selected) as one call per row.
"""
from core.entity_detection import (
    compile_alias_patterns,
    detect_candidate_entities,
    discover_category_entities,
    entities_from_category_details,
    parse_entity_aliases,
)
from core.mentions_io import iter_column_values
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

SINGLE_OUTPUT_INSTRUCTIONS = (
    "\n\nRespond with 'Sentiment' as exactly one of: Positive, Neutral, Negative. "
    "Also give 'Sentiment Rationale': a short clause (under 15 words) explaining why."
)

MULTI_ENTITY_INSTRUCTIONS = (
    "This mention may reference more than one entity. Each row's input lists which "
    "entities to assess THIS mention under 'Entities to assess' (found via a "
    "recall-biased keyword/category search — it over-matches on purpose, so verify "
    "against the actual text). For each one, classify sentiment expressed TOWARD "
    "THAT ENTITY specifically, not the mention's overall tone: Positive, Neutral, or "
    "Negative. A mention can be Positive toward one entity and Negative toward "
    "another. If an entity turns out not to genuinely be referenced (e.g. the "
    "keyword search matched \"discord\" meaning disagreement, not the platform), "
    "classify it Neutral rather than force Positive/Negative — the field is still "
    "required for every entity you're asked about.\n\n"
    "For each entity you're asked about (named EXACTLY as given), respond with two "
    "fields: 'Sentiment: <entity name>' (Positive/Neutral/Negative) and "
    "'Sentiment Rationale: <entity name>' (a short clause, under 15 words)."
)


class SentimentModule(LLMEnrichmentModule):
    key = "sentiment"
    label = "Sentiment Coding"
    description = "Code each mention's sentiment (Positive/Neutral/Negative) with an LLM."

    # ---------------------------------------------------------------- UI ---
    def render_options(self, st, key_prefix, parsed=None, file_bytes=None, filename=None):
        mode = st.radio(
            "What should sentiment be scored against?",
            ["General (overall tone)", "Toward a specific entity", "Multiple entities", "Custom prompt"],
            key=f"{key_prefix}_mode",
        )
        params = {"mode": mode, "entity": "", "custom_prompt": "",
                  "multi_source": "aliases", "entity_names": [], "category_name": "", "category_roster": {}}

        if mode == "Toward a specific entity":
            entity = st.text_input(
                "Entity (or comma-separated entities)",
                placeholder="e.g. Google, or Google, the FTC",
                key=f"{key_prefix}_entity",
            )
            params["entity"] = entity.strip()

        elif mode == "Multiple entities":
            source = st.radio(
                "Define entities by:",
                ["Manual aliases", "Brandwatch parent category"],
                key=f"{key_prefix}_multi_source",
                horizontal=True,
            )
            params["multi_source"] = "aliases" if source == "Manual aliases" else "category"

            if params["multi_source"] == "aliases":
                aliases_text = st.text_area(
                    "One entity per line: EntityName: alias1, alias2, ...",
                    height=140,
                    key=f"{key_prefix}_aliases",
                    placeholder="YouTube: youtube, yt\nTikTok: tiktok, tik tok\nRoblox",
                    help="Bias toward recall over precision — list every common way people refer "
                         "to it. A row only gets asked about entities its aliases actually match; "
                         "the LLM resolves false positives (e.g. a name that's also a normal word) "
                         "to Neutral rather than force a verdict.",
                )
                entities = parse_entity_aliases(aliases_text)
                params["entity_names"] = list(entities.keys())
                params["compiled_patterns"] = compile_alias_patterns(entities)
                if entities:
                    st.caption(f"{len(entities)} entities configured: {', '.join(entities.keys())}")
            else:
                category_name = st.text_input(
                    "Brandwatch parent category name",
                    placeholder="e.g. Tech Companies",
                    key=f"{key_prefix}_category",
                    help="Matches the 'Category Details' column Brandwatch populates when a query "
                         "is set up with categorized entities — same idea as backfill_openai.py's "
                         "--parent-category. Requires that column to be present in this export.",
                )
                params["category_name"] = category_name.strip()
                roster_key = f"{key_prefix}_category_roster"
                if st.button("Scan this file for entities under that category", key=f"{key_prefix}_scan_btn"):
                    if not file_bytes or not filename:
                        st.warning("Upload a file first (Section 1).")
                    elif not params["category_name"]:
                        st.warning("Enter a parent category name first.")
                    else:
                        values = iter_column_values(file_bytes, filename, "Category Details")
                        if not values:
                            st.error("No 'Category Details' column found in this export — the "
                                      "Brandwatch category path needs it. Use Manual aliases instead.")
                            st.session_state[roster_key] = {}
                        else:
                            roster = discover_category_entities(values, params["category_name"])
                            st.session_state[roster_key] = roster
                            if not roster:
                                st.warning(f"Found the 'Category Details' column, but no entities "
                                            f"tagged under '{params['category_name']}'. Check the "
                                            f"category name matches what's in Brandwatch exactly.")
                roster = st.session_state.get(roster_key, {})
                params["category_roster"] = roster
                if roster:
                    preview = ", ".join(f"{name} ({n:,})" for name, n in sorted(roster.items(), key=lambda kv: -kv[1]))
                    st.caption(f"{len(roster)} entities found: {preview}")

        default_preview = GENERAL_PROMPT
        if mode == "Toward a specific entity":
            default_preview = ENTITY_PROMPT_TEMPLATE.format(entity=params["entity"] or "[entity]")
        elif mode == "Multiple entities":
            default_preview = MULTI_ENTITY_INSTRUCTIONS

        if mode == "Custom prompt":
            custom_prompt = st.text_area(
                "Custom sentiment instructions",
                value=GENERAL_PROMPT,
                height=100,
                key=f"{key_prefix}_custom",
                help="Replace this with whatever sentiment instructions you need. The "
                     "Positive/Neutral/Negative + rationale output format is still enforced.",
            )
            params["custom_prompt"] = custom_prompt.strip()
        else:
            with st.expander("Preview the instructions the LLM will get"):
                st.code(default_preview, language=None)

        return params

    # ------------------------------------------------------- shared helpers ---
    def _is_multi(self, params):
        return params.get("mode") == "Multiple entities"

    def _single_instructions(self, params):
        if params["mode"] == "Custom prompt" and params["custom_prompt"]:
            return params["custom_prompt"]
        if params["mode"] == "Toward a specific entity" and params["entity"]:
            return ENTITY_PROMPT_TEMPLATE.format(entity=params["entity"])
        return GENERAL_PROMPT

    def _all_entity_names(self, params):
        if params.get("multi_source") == "category":
            return list((params.get("category_roster") or {}).keys())
        return list(params.get("entity_names") or [])

    def _entity_columns(self, name):
        return f"Sentiment: {name}", f"Sentiment Rationale: {name}"

    def _candidate_entities(self, row, params):
        if params.get("multi_source") == "category":
            return entities_from_category_details(row.get("Category Details"), params.get("category_name", ""))
        text = f"{row.get('Full Text') or ''} {row.get('Title') or ''}"
        return detect_candidate_entities(text, params.get("compiled_patterns") or {})

    # --------------------------------------------------- LLMEnrichmentModule ---
    def output_columns(self, params):
        if not self._is_multi(params):
            return ["Sentiment", "Sentiment Rationale"]
        cols = []
        for name in self._all_entity_names(params):
            cols.extend(self._entity_columns(name))
        return cols

    def system_prompt_fragment(self, params):
        if self._is_multi(params):
            return MULTI_ENTITY_INSTRUCTIONS
        return self._single_instructions(params) + SINGLE_OUTPUT_INSTRUCTIONS

    def json_schema_fragment(self, row, params):
        if not self._is_multi(params):
            return {
                "properties": {
                    "Sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
                    "Sentiment Rationale": {"type": "string"},
                },
                "required": ["Sentiment", "Sentiment Rationale"],
            }
        candidates = self._candidate_entities(row, params)
        if not candidates:
            return {}
        properties, required = {}, []
        for name in candidates:
            s_col, r_col = self._entity_columns(name)
            properties[s_col] = {"type": "string", "enum": ["Positive", "Neutral", "Negative"]}
            properties[r_col] = {"type": "string"}
            required += [s_col, r_col]
        return {"properties": properties, "required": required}

    def row_context(self, row, params):
        text = str(row.get("Full Text") or "")[:MAX_CHARS]
        ctx = {"Post Title": row.get("Title") or "", "Full Text": text}
        if self._is_multi(params):
            candidates = self._candidate_entities(row, params)
            ctx["Entities to assess"] = ", ".join(candidates) if candidates else "(none detected this row)"
        return ctx

    def columns_from_result(self, row, result, params):
        result = result or {}
        if not self._is_multi(params):
            return {
                "Sentiment": result.get("Sentiment", ""),
                "Sentiment Rationale": result.get("Sentiment Rationale", ""),
            }
        candidates = set(self._candidate_entities(row, params))
        out = {}
        for name in self._all_entity_names(params):
            s_col, r_col = self._entity_columns(name)
            if name in candidates:
                out[s_col] = result.get(s_col, "")
                out[r_col] = result.get(r_col, "")
            else:
                out[s_col] = "Not Mentioned"
                out[r_col] = ""
        return out

    def estimate_tokens_per_row(self, parsed, params, context):
        system_tokens = rough_token_estimate(self.system_prompt_fragment(params)) + 40
        assumed_text_tokens = 220  # rough average Reddit mention length
        if self._is_multi(params):
            n_entities = max(1, len(self._all_entity_names(params)))
            # Recall-biased regex rarely matches every configured entity in every row —
            # assume roughly a third are plausible candidates per row, floor of 1.
            assumed_candidates = max(1, round(n_entities / 3))
            return system_tokens + assumed_text_tokens + assumed_candidates * 15, assumed_candidates * 30
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
        if self._is_multi(params):
            n_entities = len(self._all_entity_names(params))
            if n_entities == 0:
                lines.insert(0, "⚠️ No entities configured yet — add aliases or scan a Brandwatch "
                                  "category above, or every row will be skipped.")
            else:
                lines.insert(0, f"{n_entities} entities configured. Rows where none of them are "
                                  f"detected skip the LLM call entirely (free) — the estimate above "
                                  f"assumes roughly a third of entities match per row, on average; "
                                  f"actual cost (shown after the run) will vary with your data.")
        if not context.get("text_will_be_filled"):
            lines.insert(0, "⚠️ Full Text doesn't look filled yet in this file — run Mention Filler "
                             "first (in this same run, or upload an already-filled export) or this "
                             "module will skip every row.")
        return Estimate(
            headline=f"Sentiment coding for {n:,} rows ({params.get('mode', 'General')})",
            lines=lines,
            est_cost_usd=cost,
        )
