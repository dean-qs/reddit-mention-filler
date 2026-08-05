"""Lightweight, low-tech access gate: type your @quadstrat.com email or the
app won't proceed. This is a deterrent for casual/accidental public use of a
tool that can spend real OpenAI credits — NOT real access control. The repo
is public, so anyone who reads the source can bypass it trivially. Treat it
as a speed bump, not a security boundary. Streamlit Community Cloud also
offers real viewer-restriction (Google-auth-backed) as a stronger option —
see the README.
"""
import re

ALLOWED_DOMAIN = "quadstrat.com"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_allowed(email):
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        return False
    domain = email.rpartition("@")[2]
    return domain == ALLOWED_DOMAIN or domain.endswith(f".{ALLOWED_DOMAIN}")


def require_quadstrat_email(st):
    """Blocks the rest of the app (via st.stop()) until a valid-looking
    @quadstrat.com email has been entered this session."""
    if st.session_state.get("_gate_email_ok"):
        return

    st.title("🧵 Reddit Mention Filler")
    st.caption("Quadrant Strategies")
    st.info("This tool can spend real OpenAI credits. Enter your Quadrant email to continue.")
    email = st.text_input("Work email", placeholder="you@quadstrat.com")
    if st.button("Continue"):
        if _is_allowed(email):
            st.session_state["_gate_email_ok"] = True
            st.rerun()
        else:
            st.error(f"That doesn't look like a @{ALLOWED_DOMAIN} email — access is limited to the Quadrant team.")
    st.stop()
