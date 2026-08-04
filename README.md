# Reddit Mention Filler

A Streamlit app for turning a Bulk Mentions export into a filled, enriched
copy. Our social-listening tool delivers Reddit rows with Full Text
replaced by *"Due to licensing restrictions, this mention cannot be
downloaded"* and Date empty — **Mention Filler** looks up every URL in the
free [Arctic Shift](https://arctic-shift.photon-reddit.com/) Reddit archive
and fills both in, plus a few extra columns (Score, Type, Edited, linked
URLs) and an Author Rollup sheet. **Sentiment Coding** and **Geolocation**
are optional LLM add-ons (OpenAI gpt-4o-mini) that run on top of the filled
text.

No Python/Node install needed to use it — open the hosted app, upload or
paste your export, pick modules, review the estimated time/cost, run,
download.

## Architecture

```
app.py                   Streamlit UI shell — input, module picker, estimate, run, download
core/
  text_utils.py           markdown cleanup, link extraction, per-row status helpers
  reddit_fetch.py          Arctic Shift bulk fetch (pure Python/requests, no Node)
  mentions_io.py           parse a Bulk Mentions export, fill it, write the output workbook
  llm_client.py            OpenAI client wrapper, reads OPENAI_API_KEY from st.secrets
  llm_enrichment.py        merges every selected LLM module into ONE call/row, runs them
                           in parallel with retry + a fail-fast first-row check, tallies
                           real token cost from the API responses
  geo_signals.py           regex/lookup evidence (BrE-vs-AmE spelling, subreddit hints)
                           fed into the geolocation prompt
modules/
  base.py                  AnalysisModule + LLMEnrichmentModule interfaces
  mention_filler.py        fetch + fill — the only non-LLM module
  sentiment.py              general / toward-an-entity / custom-prompt sentiment coding
  geolocation.py            country (+ US region when confident) from subreddit/title/text
  registry.py               MODULES list app.py reads — add a module here, nothing else
```

### Adding a new module

**Plain module** (like Mention Filler — no LLM call): write
`modules/your_module.py` implementing `AnalysisModule` (see
`modules/base.py`): `render_options()` draws its toggles, `estimate()`
reports a count/time/$ preview, `run()` does the work and returns a
`ModuleResult`. Add an instance to `MODULES` in `modules/registry.py`.

**LLM module** (like Sentiment/Geolocation): implement
`LLMEnrichmentModule` instead — `output_columns()`, `system_prompt_fragment()`,
`json_schema_fragment()`, `row_context()`, `columns_from_result()`, and
`estimate_tokens_per_row()`. Don't implement `run()` yourself;
`core/llm_enrichment.py` detects `LLMEnrichmentModule` instances and merges
any selected together into one combined chat completion per row — adding a
third LLM module is just another fragment merged in, no prompt-plumbing to
write.

Either way, `app.py` doesn't change. Modules run in sequence in one
session; each module's output workbook becomes the next module's input
(new columns are always appended, never inserted, so the original header
stays put between steps) — so Sentiment/Geolocation can run right after
Mention Filler in the same pass, or later on a file that's already filled.

## Ideas for future modules

A few directions that would fit the same pattern, roughly in order of how
easy they'd be to bolt on:

- **Quotable-sentence extraction** — pull the single most report-ready
  sentence per mention (LLM, cheap, high value for analysts writing up
  findings).
- **Topic/theme tagging** — multi-label classification into a
  client-defined taxonomy, same LLM-module shape as sentiment.
- **Author influence tier** — no LLM needed; derivable straight from the
  Author Rollup sheet's existing count/score aggregates.
- **Toxicity/risk flag** — useful for crisis-monitoring workflows.
- **Duplicate/near-duplicate detection** — flag cross-posted comments
  before they inflate a theme count.

## Cost model

Sentiment Coding and Geolocation call OpenAI gpt-4o-mini. Before running,
each selected module reports its own rough per-row token estimate and $
figure (`core/llm_enrichment.py` holds the pricing table — update it if
OpenAI's pricing changes). After running, the actual cost is computed from
real token usage in the API responses and shown alongside the results —
treat the pre-run number as a ballpark, the post-run number as the real
figure. Rows whose Full Text isn't filled in yet (still the licensing
placeholder, or blank) are skipped rather than sent to the API.

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

`OPENAI_API_KEY` is only needed for Sentiment Coding / Geolocation —
Mention Filler alone works with no secrets at all.

## Deploying

See the deploy instructions shared alongside this repo for connecting it to
Streamlit Community Cloud, including where to paste `OPENAI_API_KEY` into
the Cloud app's Secrets panel.
