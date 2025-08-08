# ISSUE-0006: Vector Database Integration Missing

- Impacted: RAG workflows, Document ingestion, Search
- Options:
  - A: Standardize on ChromaDB first, add Qdrant later (recommended)
  - B: Abstract layer supporting both from day one
  - C: Use FAISS in-process (limited persistence)
- Recommendation: A
- Consequences: Implement embeddings pipeline, collections, query API
- Sources: `/workspace/IMPORTANT/SUTAZAI_PRD.md`