"""Thin OpenAI client wrapper for the enrichment modules.

Reads the key from Streamlit secrets so it never lands in the repo — see
.streamlit/secrets.toml.example. Raised errors are meant to be caught by
app.py and shown as a friendly message, not a stack trace.
"""
import json
import time

import streamlit as st
from openai import OpenAI

MODEL = "gpt-4o-mini"
MAX_RETRIES = 3


class MissingApiKey(Exception):
    pass


_client = None


def get_client():
    global _client
    if _client is not None:
        return _client
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        raise MissingApiKey(
            "No OPENAI_API_KEY found. Add it in the app's Secrets settings "
            "(Streamlit Community Cloud: app settings -> Secrets) or, for a local "
            "run, in .streamlit/secrets.toml — see .streamlit/secrets.toml.example."
        )
    _client = OpenAI(api_key=key)
    return _client


def call_json(client, schema, system_prompt, user_message, attempt=1):
    """One structured-output chat completion call, with exponential-backoff
    retry. Returns (parsed_json_dict, usage). Shared by every module that
    talks to OpenAI, so retry/backoff behavior stays consistent."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=schema,
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content), resp.usage
    except Exception:
        if attempt >= MAX_RETRIES:
            raise
        time.sleep(2 ** attempt)
        return call_json(client, schema, system_prompt, user_message, attempt + 1)
