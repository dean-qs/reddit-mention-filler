"""Thin OpenAI client wrapper for the enrichment modules.

Reads the key from Streamlit secrets so it never lands in the repo — see
.streamlit/secrets.toml.example. Raised errors are meant to be caught by
app.py and shown as a friendly message, not a stack trace.
"""
import streamlit as st
from openai import OpenAI

MODEL = "gpt-4o-mini"


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
