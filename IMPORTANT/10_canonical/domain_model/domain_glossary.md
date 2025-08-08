# Domain Model & Glossary (DDD)

- User: authenticated actor interacting via frontend/API.
- Agent: executable unit providing capabilities; registered with orchestrator.
- Task: unit of work assigned to an agent with parameters and result.
- Knowledge Document: ingested artifact with embeddings for RAG.
- Orchestrator: coordinates agent execution and resources.
- Capability: declared action an agent can perform.

Context Map: Core Domain (Orchestrator, RAG), Supporting (Auth, Docs), Generic (Observability, Gateway).