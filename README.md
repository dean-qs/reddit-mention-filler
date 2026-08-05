# Reddit Mention Filler

A Streamlit app for turning a Bulk Mentions export into a filled, enriched
copy. Our social-listening tool delivers Reddit rows with Full Text
replaced by *"Due to licensing restrictions, this mention cannot be
downloaded"* and Date empty — **Mention Filler** looks up every URL in the
free [Arctic Shift](https://arctic-shift.photon-reddit.com/) Reddit archive
and fills both in, plus a few extra columns (Score, Type, Edited, linked
URLs) and an Author Rollup sheet. **Sentiment Coding**, **Geolocation**, and
**Theme Summary** are optional LLM add-ons (OpenAI gpt-4o-mini) that run on
top of the filled text.

No Python/Node install needed to use it — open the hosted app, sign in with
a Quadrant email, upload or paste your export, pick modules, review the
estimated time/cost, run, download.

## Architecture

```
app.py                   Streamlit UI shell — input, module picker, estimate, run, download
core/
  text_utils.py           markdown cleanup, link extraction, per-row status helpers
  reddit_fetch.py          Arctic Shift bulk fetch (pure Python/requests, no Node)
  mentions_io.py           parse a Bulk Mentions export, fill it, write the output workbook
  llm_client.py            OpenAI client wrapper (reads OPENAI_API_KEY from st.secrets) +
                           call_json(), the shared retry-wrapped structured-output call
  llm_enrichment.py        merges every selected LLMEnrichmentModule into ONE call/row
                           (schema rebuilt PER ROW — a module can ask for different, or
                           zero, fields depending on the row), parallel + retry + a
                           fail-fast first-row check, real token cost from the API responses
  geo_signals.py           regex/lookup evidence (BrE/AmE spelling, EU-vs-US number
                           formatting, Iberian-vs-LatAm Spanish, subreddit hints) fed
                           into the geolocation prompt
  entity_detection.py      recall-biased alias regexes + Brandwatch "Category Details"
                           parsing, for multi-entity Sentiment Coding
  access_gate.py           the @quadstrat.com email gate (honor-system, see below)
  cost_caps.py             MAX_LLM_ROWS_PER_RUN / MAX_LLM_COST_USD_PER_RUN backstops
modules/
  base.py                  AnalysisModule + LLMEnrichmentModule interfaces
  mention_filler.py        fetch + fill (plain AnalysisModule, no LLM)
  sentiment.py              general / toward-an-entity / multiple-entities / custom-prompt;
                           writes LLM-prefixed columns + Sentiment Breakdown/Long sheets
  geolocation.py            country (+ US region when confident) from subreddit/title/text
  theme_summary.py          discover themes from a sample (or use a predefined list), tag
                           every row (plain AnalysisModule — two-phase, doesn't fit the
                           per-row coordinator)
  registry.py               MODULES list app.py reads — add a module here, nothing else
```

### Adding a new module

**Plain module** (like Mention Filler or Theme Summary — manages its own
run start to finish): write `modules/your_module.py` implementing
`AnalysisModule` (see `modules/base.py`): `render_options()` draws its
toggles, `estimate()` reports a count/time/$ preview, `run()` does the work
and returns a `ModuleResult`. Add an instance to `MODULES` in
`modules/registry.py`.

**LLM module** (like Sentiment/Geolocation — one JSON verdict per row):
implement `LLMEnrichmentModule` instead — `output_columns()`,
`system_prompt_fragment()`, `json_schema_fragment(row, params)`,
`row_context()`, `columns_from_result(row, result, params)`, and
`estimate_tokens_per_row()`. Don't implement `run()` yourself;
`core/llm_enrichment.py` detects `LLMEnrichmentModule` instances and merges
any selected together into one combined chat completion per row — adding a
third LLM module is just another fragment merged in, no prompt-plumbing to
write. `json_schema_fragment`/`columns_from_result` receive the row, so a
module can opt out of a given row entirely (return no properties) when it
has nothing to ask — e.g. multi-entity Sentiment Coding only asks about
entities its regex/category prefilter actually found in that row's text; if
every module opts out, the row skips the API call altogether. Optionally
implement `build_extra_sheets(wb, params, row_values)` to append aggregate/
summary sheets after the main per-row writing pass (see Sentiment Coding's
Breakdown/Long sheets).

Either way, `app.py` doesn't change. Modules run in sequence in one
session; each module's output workbook becomes the next module's input
(new columns are always appended, never inserted, so the original header
stays put between steps) — so Sentiment/Geolocation/Theme Summary can run
right after Mention Filler in the same pass, or later on a file that's
already filled.

## Multi-entity Sentiment Coding

Two ways to define entities (Sentiment Coding → "Multiple entities"):

- **Manual aliases** — one entity per line, `EntityName: alias1, alias2,
  ...`. Compiled into a recall-biased regex per entity (left word boundary,
  open right boundary, so derivative forms like "TikTokkers" still match) —
  same approach as `children_safety_classifier_v2.py`. Bias toward listing
  too many aliases; the LLM resolves false positives (e.g. a brand name
  that's also an ordinary word) to "Not Mentioned"/Neutral rather than the
  regex trying to be precise.
- **Brandwatch parent category** — type the category name (e.g. "Tech
  Companies") and click "Scan this file for entities under that category".
  Parses the `Category Details` column Brandwatch itself populates when a
  query is set up with categorized entities (same technique as
  `stage1_parse.py`'s `extract_entities`) — no aliases to write, BW already
  tagged it. Requires that column to be present in the export.

Either way, each row only gets asked (and costs tokens) for entities it
plausibly mentions; the rest are written "Not Mentioned" for free.

Output columns are named `LLM Sentiment` / `LLM Sentiment Rationale` (or
`LLM Sentiment: <entity>` in multi-entity mode) rather than plain
`Sentiment` — Brandwatch exports already carry their own native `Sentiment`
column, and writing to `Sentiment` would silently overwrite Brandwatch's
own values. **Any new module should follow the same `LLM `-prefix
convention** for columns that might collide with a source column BW already
populates (`Sentiment`, `Country`, `Language`, etc.).

Two extra sheets get added alongside the main output:
- **Sentiment Breakdown** — Positive/Negative/Neutral counts + Net Sentiment
  (`(Positive - Negative) / Total Assessed`), one row per entity in multi-
  entity mode, or a single overall row otherwise.
- **Sentiment Long** (multi-entity mode only) — one row per (mention,
  entity) pair actually assessed (Url, Entity, Sentiment, Rationale),
  skipping "Not Mentioned" pairs — a long/pivot-friendly view alongside the
  wide per-entity columns on the main sheet, without restructuring the main
  sheet's rows (which every other module relies on staying stable).

## Theme Summary: predefined vs. discovered themes

Theme Summary defaults to auto-discovering themes from a data sample, but
can also take a predefined list (`ThemeName: description` per line) that
skips the discovery call entirely — useful when you already know the
buckets you want (e.g. a recurring monthly report with a fixed taxonomy).

## Mixed-source exports (Reddit + other platforms)

Mention Filler only touches Reddit URLs — every other URL (X/Twitter, news,
forums, whatever else Brandwatch already filled in) passes through
completely untouched: its existing Full Text/Date/Author stay exactly as
Brandwatch gave them, only the Fetch Status column gets tagged
`UNPARSEABLE_URL`. That means a mixed-source Brandwatch export just works
in one pass: run Mention Filler once, only the Reddit rows get fetched, and
every downstream module (Sentiment/Geolocation/Theme Summary) sees real
text for every row either way — no separate handling needed.

## Access control

This app can spend real OpenAI credits, so `core/access_gate.py` gates the
whole app behind a `@quadstrat.com` email text box before anything else
renders. **This is a deterrent, not security** — the repo is public, so
anyone who reads the source can bypass it trivially. Treat it as a speed
bump against casual/accidental public use, not a real access boundary.
Streamlit Community Cloud also offers real viewer-restriction (Google-auth-
backed, under the app's sharing settings) as a stronger option worth
layering on if that becomes a real concern.

As a technical backstop independent of the honor-system gate,
`core/cost_caps.py` enforces `MAX_LLM_ROWS_PER_RUN` (hard cap on rows
actually sent to OpenAI in one run, default 20,000) and
`MAX_LLM_COST_USD_PER_RUN` (soft cap on the pre-run cost *estimate*,
default $25) — both configurable via Secrets.

## Ideas for future modules

A few directions that would fit the same pattern:

- **Quotable-sentence extraction** — pull the single most report-ready
  sentence per mention (LLM, cheap, high value for analysts writing up
  findings).
- **Author influence tier** — no LLM needed; derivable straight from the
  Author Rollup sheet's existing count/score aggregates.
- **Toxicity/risk flag** — useful for crisis-monitoring workflows.
- **Duplicate/near-duplicate detection** — flag cross-posted comments
  before they inflate a theme count.
- **BERTopic-based topic modeling** — a more rigorous alternative to Theme
  Summary's LLM-based approach, but pulls in sentence-transformers/torch/
  umap/hdbscan (likely 1-2GB of deps) — real risk of not fitting Streamlit
  Community Cloud's free-tier resource limits; would need testing to
  confirm it deploys at all before committing to it.

## Cost model

Sentiment Coding, Geolocation, and Theme Summary call OpenAI gpt-4o-mini.
Before running, each selected module reports its own rough token estimate
and $ figure (`core/llm_enrichment.py` holds the pricing table — update it
if OpenAI's pricing changes). After running, the actual cost is computed
from real token usage in the API responses and shown alongside the results
— treat the pre-run number as a ballpark, the post-run number as the real
figure. Rows whose Full Text isn't filled in yet (still the licensing
placeholder, or blank) are skipped rather than sent to the API, as are
rows where every selected LLM module has nothing to ask about (e.g.
multi-entity sentiment with zero configured entities detected that row).

## Why there's no live-scrape fallback for archive misses

The original local script had a step 3: for the rare URL Arctic Shift
doesn't have (typically well under 1%), it used a real headless Chrome
(Puppeteer) to scrape the page directly, because Reddit blocks plain
HTTP requests outright — confirmed while building this: even a `curl` with
a real browser User-Agent gets an HTTP 403 from old.reddit.com, while the
Arctic Shift API (a plain HTTP JSON API) works fine. A real browser is
needed to get past that, which doesn't fit a lightweight Streamlit Cloud
container. This app ships without it: archive-miss URLs are listed in a
downloadable `unmatched.csv` instead of being auto-scraped. A Playwright
(Python, headless-Chromium) version of that fallback is a reasonable
future addition if the trickle of misses turns out to matter in practice.

## Running locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml  # fill in OPENAI_API_KEY
streamlit run app.py
```

`OPENAI_API_KEY` is only needed for Sentiment Coding / Geolocation / Theme
Summary — Mention Filler alone works with no secrets at all.

## Deploying

See the deploy instructions shared alongside this repo for connecting it to
Streamlit Community Cloud, including where to paste `OPENAI_API_KEY` (and
the optional cost-cap overrides) into the Cloud app's Secrets panel.
