# Customer Support AI Agent (CrewAI + Gemini + Streamlit)

A single-agent customer support chatbot built with CrewAI. It can:

- Answer company/product/policy questions using your **prebuilt FAISS knowledge base**
  (`data/faiss.index` + `data/chunks.json`)
- Look up order status/details from an **Excel "database"** (`data/pakistani_dummy_orders.xlsx`)
- **Escalate to a human** when it can't resolve something, or when the customer asks for
  one — creating a pending ticket visible in the sidebar
- Remember the conversation for the whole chat session

## Project structure

```
support_agent/
├── app.py                  # Streamlit chat UI (entry point)
├── agent.py                # CrewAI Agent/Task/Crew setup (Gemini LLM)
├── tools.py                # The 3 tools: knowledge search, order lookup, escalate
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
└── data/
    ├── faiss.index                    # <- put your prebuilt FAISS index here
    ├── chunks.json                    # <- put your prebuilt chunks here
    └── pakistani_dummy_orders.xlsx    # dummy order "database"
```

`escalations.json` is created automatically at runtime in the project root the first
time an issue is escalated — you don't need to create it yourself.

## 1. Add your data files

Copy your own `faiss.index` and `chunks.json` into `data/` (replacing the placeholders
if any). `tools.py` expects `chunks.json` entries shaped roughly like:

```json
{
  "id": 0,
  "text": "...",
  "metadata": {
    "chunk_id": "...",
    "source_filename": "...",
    "page_number": 1,
    "doc_type": "..."
  }
}
```

If your real file uses different key names, open `tools.py` → `_get_chunks()` and adjust
the `.get(...)` calls — it's one small function.

The embedding model used for search **must match** the one your index was built with:
`all-MiniLM-L6-v2` (384 dimensions), already set in `tools.py`.

## 2. Push to GitHub

Create a new GitHub repo and push this whole `support_agent/` folder (including the
`data/` folder with your real files) as the repo contents. Do **not** commit a real
`.streamlit/secrets.toml` — it's already in `.gitignore`.

## 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Pick your GitHub repo and branch
3. Set **Main file path** to `app.py`
4. Before (or after) deploying, open **Settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-actual-gemini-api-key"
   ```
5. Click **Deploy**

That's it — the app reads the key via `st.secrets["GEMINI_API_KEY"]`, so nothing else
needs configuring.

## How it works, briefly

- **LLM**: `crewai.LLM(model="gemini/gemini-3.5-flash-lite", api_key=..., temperature=0.3)`
  — CrewAI has native Gemini support, so no extra Google SDK setup is needed.
- **Memory**: `app.py` keeps the whole chat in `st.session_state.messages` and passes
  the formatted transcript into the CrewAI `Task` description on every turn, so the
  agent always has full context of the conversation. This keeps things simple and
  beginner-friendly — no separate memory database required.
- **Escalation**: the `escalate_to_human` tool appends a ticket (`ticket_id`, timestamp,
  customer name, order ID, summary, `status: "pending"`) to `escalations.json`. The
  sidebar reads this file and lists all `pending` tickets. This file lives on
  Streamlit Cloud's local (ephemeral) disk — fine for a demo, but it will reset if the
  app restarts or redeploys. For a real production system you'd swap this for a proper
  database (e.g. Google Sheets, Airtable, or Postgres).

## Notes on the Gemini model

`gemini-3.5-flash-lite` is a current, low-latency, cost-effective GA model — a good fit
for a support chatbot. If Google renames or retires it later, just change
`GEMINI_MODEL` at the top of `agent.py`.
