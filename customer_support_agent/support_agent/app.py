import json
from pathlib import Path

import streamlit as st

from agent import get_response

BASE_DIR = Path(__file__).parent
ESCALATIONS_PATH = BASE_DIR / "escalations.json"

st.set_page_config(page_title="Customer Support Assistant", page_icon="🎧", layout="wide")

# ----------------------------------------------------------------------------------
# API key — read from Streamlit secrets only (never hardcode, never ask the user)
# ----------------------------------------------------------------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "GEMINI_API_KEY is not set.\n\n"
        "On Streamlit Cloud: open your app → Settings → Secrets, and add:\n\n"
        'GEMINI_API_KEY = "your-key-here"'
    )
    st.stop()

# ----------------------------------------------------------------------------------
# Session state (per-browser-session chat memory)
# ----------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}, ...]


def format_history() -> str:
    lines = []
    for m in st.session_state.messages:
        speaker = "Customer" if m["role"] == "user" else "Support Agent"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def load_escalations() -> list[dict]:
    if ESCALATIONS_PATH.exists():
        try:
            with open(ESCALATIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


# ----------------------------------------------------------------------------------
# Sidebar — pending human-escalation queue
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.header("🧑‍💼 Human Escalation Queue")
    escalations = load_escalations()
    pending = [e for e in escalations if e.get("status") == "pending"]

    if not pending:
        st.caption("No pending escalations.")
    else:
        st.caption(f"{len(pending)} pending")
        for ticket in reversed(pending):
            label = f"{ticket.get('ticket_id', '?')} — {ticket.get('customer_name', 'Unknown')}"
            with st.expander(label):
                st.write(f"**Order ID:** {ticket.get('order_id', 'N/A')}")
                st.write(f"**Summary:** {ticket.get('summary', '')}")
                st.caption(ticket.get("timestamp", ""))

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------------------
# Main chat UI
# ----------------------------------------------------------------------------------
st.title("🎧 Customer Support Assistant")
st.caption("Ask about your order, our products, or our policies. A human will step in if I can't help.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    conversation_so_far = format_history()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = get_response(
                    api_key=GEMINI_API_KEY,
                    conversation_history=conversation_so_far,
                    user_message=user_input,
                )
            except Exception as e:
                reply = f"Sorry, something went wrong while handling your request. ({e})"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
