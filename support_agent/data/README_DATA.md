This folder contains the live data the app reads:

- chunks.json    — company knowledge chunks (refund/shipping/warranty/cancellation policies + FAQ), used by the search_company_knowledge tool
- faiss.index    — vector index built from chunks.json (all-MiniLM-L6-v2, 384-dim), used for semantic search
- pakistani_dummy_orders.xlsx — dummy order "database", used directly (no embeddings) by the lookup_order tool

If you ever regenerate chunks.json, you MUST also regenerate faiss.index from that same file in a
fresh Colab session — an index built from a different chunks.json than the one deployed here will
return mismatched results.
