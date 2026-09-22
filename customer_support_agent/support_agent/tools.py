"""
Tools used by the customer support agent.

1. KnowledgeSearchTool  -> semantic search over data/faiss.index + data/chunks.json
2. OrderLookupTool      -> looks up rows in data/pakistani_dummy_orders.xlsx (our "database")
3. EscalateToHumanTool  -> writes a pending ticket to escalations.json for a human to review

All resources (embedding model, faiss index, chunks, orders spreadsheet) are loaded once
and cached at module level so repeated tool calls within a session are fast.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ESCALATIONS_PATH = BASE_DIR / "escalations.json"

# Must match the model the embeddings were originally built with.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------------------
# Lazy-loaded shared resources
# --------------------------------------------------------------------------------------

_embedding_model = None
_faiss_index = None
_chunks_by_id = None
_orders_df = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        import faiss
        index_path = DATA_DIR / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Could not find {index_path}. Make sure faiss.index is committed under data/."
            )
        _faiss_index = faiss.read_index(str(index_path))
    return _faiss_index


def _get_chunks():
    """
    Returns a dict: {id: {"text": ..., "metadata": {...}}}

    Expected chunks.json shape (adjust the parsing below if your file differs):
    [
      {
        "id": 0,
        "text": "...",
        "metadata": {
            "chunk_id": "...",
            "source_filename": "...",
            "source_path": "...",
            "page_number": 1,
            "chunk_number": 1,
            "doc_type": "..."
        }
      },
      ...
    ]
    Flat records (no nested "metadata" key, fields at top level) are also supported.
    """
    global _chunks_by_id
    if _chunks_by_id is None:
        chunks_path = DATA_DIR / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Could not find {chunks_path}. Make sure chunks.json is committed under data/."
            )
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        by_id = {}
        for i, rec in enumerate(raw):
            chunk_id = rec.get("id", i)
            text = rec.get("text") or rec.get("content") or ""
            meta = rec.get("metadata", rec)  # fall back to flat record
            by_id[int(chunk_id)] = {
                "text": text,
                "source_filename": meta.get("source_filename") or meta.get("source") or "unknown",
                "page_number": meta.get("page_number", meta.get("page", "")),
                "doc_type": meta.get("doc_type", meta.get("document_type", "")),
            }
        _chunks_by_id = by_id
    return _chunks_by_id


def _get_orders_df():
    global _orders_df
    if _orders_df is None:
        orders_path = DATA_DIR / "pakistani_dummy_orders.xlsx"
        if not orders_path.exists():
            raise FileNotFoundError(
                f"Could not find {orders_path}. Make sure the orders spreadsheet is committed under data/."
            )
        df = pd.read_excel(orders_path)
        df.columns = [str(c).strip() for c in df.columns]
        _orders_df = df
    return _orders_df


def _resolve_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Find the first matching column name (case-insensitive) from a list of candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# --------------------------------------------------------------------------------------
# Tool 1: Knowledge base search (FAISS + chunks.json)
# --------------------------------------------------------------------------------------

class KnowledgeSearchInput(BaseModel):
    query: str = Field(..., description="The customer's question to search the company knowledge base for.")


class KnowledgeSearchTool(BaseTool):
    name: str = "search_company_knowledge"
    description: str = (
        "Search the company's knowledge base (policies, FAQs, product documentation) for "
        "information relevant to the customer's question. Use this for anything about "
        "policies, how-tos, product details, returns/refunds rules, warranty, etc. "
        "Input should be the customer's question in plain English."
    )
    args_schema: Type[BaseModel] = KnowledgeSearchInput

    def _run(self, query: str) -> str:
        try:
            model = _get_embedding_model()
            index = _get_faiss_index()
            chunks = _get_chunks()
        except FileNotFoundError as e:
            return f"Knowledge base is not available: {e}"

        query_embedding = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        k = 3
        scores, ids = index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = chunks.get(int(idx))
            if not chunk or not chunk["text"]:
                continue
            source_note = f" [source: {chunk['source_filename']}]" if chunk["source_filename"] != "unknown" else ""
            results.append(f"(relevance {score:.2f}){source_note} {chunk['text']}")

        if not results:
            return "No relevant information was found in the knowledge base for this question."
        return "\n\n".join(results)


# --------------------------------------------------------------------------------------
# Tool 2: Order lookup (Excel "database")
# --------------------------------------------------------------------------------------

class OrderLookupInput(BaseModel):
    identifier: str = Field(
        ..., description="The Order ID (e.g. ORD-1001) OR the customer's full name to look up an order."
    )


class OrderLookupTool(BaseTool):
    name: str = "lookup_order"
    description: str = (
        "Look up an order's details (product, order date, shipping address, status) by "
        "Order ID or by customer name. Use this whenever a customer asks about their order "
        "status, delivery, or order details."
    )
    args_schema: Type[BaseModel] = OrderLookupInput

    def _run(self, identifier: str) -> str:
        try:
            df = _get_orders_df()
        except FileNotFoundError as e:
            return f"Order database is not available: {e}"

        order_id_col = _resolve_column(df, "Order ID")
        name_col = _resolve_column(df, "Customer Name")
        status_col = _resolve_column(df, "Status", "Order Status")
        product_col = _resolve_column(df, "Product")
        date_col = _resolve_column(df, "Order Date")
        address_col = _resolve_column(df, "Shipping Address")
        contact_col = _resolve_column(df, "Contact Number")

        needle = identifier.strip().lower()

        matches = pd.DataFrame()
        if order_id_col:
            matches = df[df[order_id_col].astype(str).str.strip().str.lower() == needle]
        if matches.empty and name_col:
            matches = df[df[name_col].astype(str).str.lower().str.contains(needle, na=False)]

        if matches.empty:
            return f"No order found matching '{identifier}'. Ask the customer to double-check their Order ID."

        lines = []
        for _, row in matches.head(5).iterrows():
            parts = []
            if order_id_col:
                parts.append(f"Order ID: {row[order_id_col]}")
            if name_col:
                parts.append(f"Customer: {row[name_col]}")
            if product_col:
                parts.append(f"Product: {row[product_col]}")
            if date_col:
                parts.append(f"Order Date: {row[date_col]}")
            if status_col:
                parts.append(f"Status: {row[status_col]}")
            if address_col:
                parts.append(f"Shipping Address: {row[address_col]}")
            if contact_col:
                parts.append(f"Contact: {row[contact_col]}")
            lines.append(" | ".join(parts))

        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Tool 3: Escalate to human
# --------------------------------------------------------------------------------------

class EscalateInput(BaseModel):
    summary: str = Field(
        ..., description="A short summary of the customer's issue and what has already been tried."
    )
    customer_name: str = Field(default="", description="Customer's name if known.")
    order_id: str = Field(default="", description="Related Order ID if known.")


class EscalateToHumanTool(BaseTool):
    name: str = "escalate_to_human"
    description: str = (
        "Escalate the current issue to a human support agent. Use this ONLY when you cannot "
        "resolve the customer's issue with the knowledge base or order lookup tools, or when "
        "the customer explicitly asks to speak to a human or agent. This creates a pending "
        "ticket for a human to review. After calling this, tell the customer their request has "
        "been escalated."
    )
    args_schema: Type[BaseModel] = EscalateInput

    def _run(self, summary: str, customer_name: str = "", order_id: str = "") -> str:
        ticket = {
            "ticket_id": f"ESC-{int(datetime.now(timezone.utc).timestamp())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_name": customer_name or "Unknown",
            "order_id": order_id or "N/A",
            "summary": summary,
            "status": "pending",
        }

        escalations = []
        if ESCALATIONS_PATH.exists():
            try:
                with open(ESCALATIONS_PATH, "r", encoding="utf-8") as f:
                    escalations = json.load(f)
            except json.JSONDecodeError:
                escalations = []

        escalations.append(ticket)

        with open(ESCALATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(escalations, f, indent=2, ensure_ascii=False)

        return f"Escalated successfully. Ticket {ticket['ticket_id']} created and marked pending for human review."
