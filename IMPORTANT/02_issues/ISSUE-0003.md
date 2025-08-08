# ISSUE-0003: Model Mismatch Between Backend and Ollama

- Impacted: Model management, Inference endpoints
- Options:
  - A: Change backend default to TinyLlama (fast) and add fallback list (recommended)
  - B: Load `gpt-oss` and ensure GPU fit
  - C: Implement model negotiation per request
- Recommendation: A (with optional preloading policy)
- Consequences: Update config, preload models, adjust tests
- Sources: `/workspace/IMPORTANT/SUTAZAI_PRD.md`