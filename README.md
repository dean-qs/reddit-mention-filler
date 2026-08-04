# Reddit Mention Filler

A Streamlit app that fills in the withheld Reddit text on a Bulk Mentions
export. Our social-listening tool delivers Reddit rows with Full Text
replaced by *"Due to licensing restrictions, this mention cannot be
downloaded"* and Date empty — this looks up every URL in the free
[Arctic Shift](https://arctic-shift.photon-reddit.com/) Reddit archive and
fills both in, plus a few extra columns (Score, Type, Edited, linked URLs)
and an Author Rollup sheet.

No Python/Node install needed to use it — open the hosted app, upload or
paste your export, pick modules, run, download.

## Architecture

```
app.py                   Streamlit UI shell — input, module picker, estimate, run, download
core/
  text_utils.py           markdown cleanup, link extraction, per-row status helpers
  reddit_fetch.py          Arctic Shift bulk fetch (pure Python/requests, no Node)
  mentions_io.py           parse a Bulk Mentions export, fill it, write the output workbook
modules/
  base.py                  AnalysisModule interface every module implements
  mention_filler.py        v0's only module — wraps core/ into estimate() + run()
  registry.py              MODULES list app.py reads — add a module here, nothing else
```

### Adding a new module (sentiment recoding, geolocation, ...)

1. Write `modules/your_module.py` with a class implementing `AnalysisModule`
   (see `modules/base.py`): `render_options()` draws its toggles,
   `estimate()` reports a count/time/$ preview before anything runs, `run()`
   does the work and returns a `ModuleResult`.
2. Add an instance to `MODULES` in `modules/registry.py`.

That's it — `app.py` doesn't change. Modules run in sequence in one session;
each module's output workbook becomes the next module's input (new columns
are always appended, never inserted, so the original header stays put
between steps).

Any module that calls a paid API (an LLM for sentiment, a geocoding API,
etc.) should read its key via `st.secrets["YOUR_KEY"]` — see
`.streamlit/secrets.toml.example` for the pattern — and should report a
real `$` estimate in `estimate()`, not the `$0` the archive-only Mention
Filler reports.

## Why there's no live-scrape fallback (yet)

The original local script had a step 3: for the rare URL Arctic Shift
doesn't have (typically well under 1%), it used a real headless Chrome
(Puppeteer) to scrape the page directly, because Reddit blocks plain
HTTP requests outright — confirmed while building this: even a `curl` with
a real browser User-Agent gets an HTTP 403 from old.reddit.com, while the
Arctic Shift API (a plain HTTP JSON API) works fine. A real browser is
needed to get past that, which doesn't fit a lightweight Streamlit Cloud
container. v0 ships without it: archive-miss URLs are listed in a
downloadable `unmatched.csv` instead of being auto-scraped. A Playwright
(Python, headless-Chromium) version of that fallback is a reasonable v1
addition if the trickle of misses turns out to matter in practice.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying

See the deploy instructions shared alongside this repo for connecting it to
Streamlit Community Cloud.
