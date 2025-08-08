# 🚀 SutazAI — Local AI Automation

Practical task automation with a local LLM (Ollama) backed by a FastAPI backend, Streamlit frontend, vector stores (Qdrant/Chroma/FAISS), and a full monitoring stack. No cloud keys required by default.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Docker](https://img.shields.io/badge/docker-20.0+-blue.svg)](https://www.docker.com)
[![Status](https://img.shields.io/badge/status-development-yellow.svg)](https://github.com)

## 🎯 What It Does

- Chat and generation via Ollama (default model: `tinyllama`)
- Code improvement workflow endpoint (background analysis + report)
- Lightweight task mesh over Redis Streams (enqueue/results/agents)
- Vector search services (standalone FAISS; LlamaIndex helper)
- Monitoring with Prometheus, Grafana, Loki/Promtail, system exporters
- **Optional Features** (disabled by default):
  - FSDP distributed training (`ENABLE_FSDP=true`)
  - TabbyML code completion (`ENABLE_TABBY=true`)
- Runs locally; no external API keys required for core flows

## 🚀 Quick Start

```bash
# 1) Start the stack (Docker)
docker compose up -d

# 2) Open core endpoints
# Backend API (FastAPI):
open http://localhost:10010/docs  # or visit manually

# Frontend (Streamlit):
open http://localhost:10011

# Health and metrics:
curl http://localhost:10010/health
curl http://localhost:10010/public/metrics

# Mesh health (Redis Streams):
curl http://localhost:10010/api/v1/mesh/health
```

Core published ports (external:internal)
- Backend 10010:8000, Frontend 10011:8501
- Postgres 10000:5432, Redis 10001:6379
- Qdrant 10101:6333 + 10102:6334, ChromaDB 10100:8000, FAISS 10103:8000
- Neo4j 10002:7474 + 10003:7687, Ollama 10104:10104
- Prometheus 10200:9090, Grafana 10201:3000, Loki 10202:3100

## 💻 Example Calls

Chat (XSS-hardened endpoint)
```bash
curl -sS -X POST http://localhost:10010/api/v1/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hello, SutazAI!"}' | jq .
```

Models
```bash
curl -sS http://localhost:10010/api/v1/models/ | jq .
```

Mesh enqueue/results
```bash
curl -sS -X POST http://localhost:10010/api/v1/mesh/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"topic": "code_tasks", "task": {"op": "scan", "path": "/opt/sutazaiapp"}}'
curl -sS 'http://localhost:10010/api/v1/mesh/results?topic=code_tasks&count=5' | jq .
```

## 🛠️ Requirements

- **Docker**: 20.0+ 
- **Docker Compose**: 2.0+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB for models and data
- **CPU**: 4+ cores recommended

## 📦 What's Included

Core services (compose)
- Backend (FastAPI), Frontend (Streamlit), Ollama, Postgres, Redis
- Vector DBs: Qdrant, ChromaDB; optional FAISS vector service
- Neo4j (for knowledge-graph features under flags)
- Monitoring: Prometheus, Grafana, Loki/Promtail, exporters

Agents and tools
- Hardware Resource Optimizer (privileged agent), GPT engineering tools, LlamaIndex helper, AI metrics exporter, and more. See the overview for the complete list and ports.

Local AI model
- Default: `tinyllama` via Ollama (`http://localhost:10104`). Other models can be pulled via the `/api/v1/models/pull` endpoint or directly with Ollama.

## 🎮 Commands

```bash
# Start the system
docker compose up -d

# Stop the system
docker compose down

# View logs
docker compose logs -f

# Check status
curl http://localhost:10010/health

# Enable optional features
ENABLE_FSDP=true ENABLE_TABBY=true docker-compose --profile optional up -d

# Or use the helper script
./scripts/start-with-features.sh
```

## 📚 Documentation

- SUTAZAI_CODEBASE_OVERVIEW.md — Architecture, components, ports, dependencies
- docs/OPTIONAL_FEATURES.md — Guide for optional features (FSDP, TabbyML)
- MIGRATION_NOTES.md — Migration guide for feature flags
- docs/backend_openapi.json — Exported OpenAPI schema (generated)
- docs/backend_endpoints.md — Endpoints list grouped by tag (generated)
- Live API Docs — http://localhost:10010/docs (after `docker compose up -d`)

Generate or refresh API docs
```bash
python3 scripts/export_openapi.py
python3 scripts/summarize_openapi.py
```

## 🔒 Privacy & Security

- **No External APIs**: Everything runs locally
- **No Data Collection**: Your data stays on your machine
- **No Internet Required**: Can run completely offline
- **Open Source**: Audit the code yourself

## 🤝 Contributing

Contributions are welcome! Please focus on:
- Practical automation workflows
- Performance improvements
- Bug fixes
- Documentation improvements

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

Note:
- Messaging is currently implemented via Redis Streams (lightweight mesh). RabbitMQ utilities exist in the repo but are not provisioned by default.
- Some services in docker-compose.yml are optional or disabled; see the overview for the accurate matrix of active ports.
