"""Configurable spend/volume caps for LLM modules — a technical backstop
independent of the email-gate honor system (core/access_gate.py). Override
via st.secrets; sane defaults apply otherwise. See .streamlit/secrets.toml.example.
"""
import streamlit as st

DEFAULT_MAX_ROWS_PER_RUN = 20_000
DEFAULT_MAX_COST_USD_PER_RUN = 25.0


class CostCapExceeded(Exception):
    pass


def _get_cap(name, default):
    try:
        val = st.secrets.get(name)
        return float(val) if val is not None else default
    except Exception:
        return default


def max_rows_per_run():
    return _get_cap("MAX_LLM_ROWS_PER_RUN", DEFAULT_MAX_ROWS_PER_RUN)


def max_cost_usd_per_run():
    return _get_cap("MAX_LLM_COST_USD_PER_RUN", DEFAULT_MAX_COST_USD_PER_RUN)
