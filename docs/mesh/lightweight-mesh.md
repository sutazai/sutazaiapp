# Lightweight Mesh (Redis Streams)

A minimal, hardware‑friendly coordination layer using Redis Streams.

## Why
- Avoids heavy/unused components (Kong/Consul/RabbitMQ)
- Uses existing Redis service
- Adds backpressure and basic routing for multi‑agent workflows

## Concepts
- Tasks: `stream:tasks:<topic>` (e.g., `stream:tasks:nlp`)
- Results: `stream:results:<topic>`
- Dead‑letter: `stream:dead:<topic>`
- Agent registry (optional): `mesh:agent:<id>` keys with TTL

## API
- `POST /api/v1/mesh/enqueue` → `{ id }`
- `GET /api/v1/mesh/results?topic=...&count=N` → latest N results
- `GET /api/v1/mesh/agents` → alive agents (if registered)
 - `GET /api/v1/mesh/health` → connectivity + agent count

### Ollama Gateway (Rate-Limited)
- `POST /api/v1/mesh/ollama/generate` body:
  - `{ "model": "tinyllama", "prompt": "...", "options": { /* optional */ } }`
- Enforces token-bucket rate limit via Redis (defaults: capacity=10, refill=2 req/s)
- Returns 429 with `{ "retry_after_ms": N }` on backpressure

## CLI
- Enqueue: `python3 scripts/mesh/enqueue_task.py --topic nlp --task '{"prompt":"hi"}'`
- Tail: `python3 scripts/mesh/tail_results.py --topic nlp --count 5`
- Agent consumer (optional):
  `python3 scripts/mesh/agent_stream_consumer.py --topic nlp --group nlp --consumer local-01 --results nlp`

## Notes
- All endpoints and scripts are local‑only; no external services required.
- Consumer group is created automatically if absent.
- Rate limiting for Ollama can be implemented atop Redis with a token bucket (later step).
  - Implemented at `/api/v1/mesh/ollama/generate` with environment overrides:
    - `OLLAMA_BUCKET_CAPACITY`, `OLLAMA_BUCKET_REFILL_PER_SEC`
  - Target base URL configurable via `OLLAMA_BASE_URL` or `OLLAMA_URL`

## Quickstart (curl)
- Enqueue a task:
  - `curl -s -X POST http://localhost:8000/api/v1/mesh/enqueue -H 'Content-Type: application/json' -d '{"topic":"nlp","task":{"prompt":"hi"}}'`
- Read recent results:
  - `curl -s 'http://localhost:8000/api/v1/mesh/results?topic=nlp&count=5'`
- Check mesh health:
  - `curl -s http://localhost:8000/api/v1/mesh/health`
- Rate‑limited Ollama generate:
  - `curl -s -X POST http://localhost:8000/api/v1/mesh/ollama/generate -H 'Content-Type: application/json' -d '{"model":"tinyllama","prompt":"hello"}'`

If you exceed the configured rate, the Ollama endpoint returns `429` with `{ "retry_after_ms": N }`.
