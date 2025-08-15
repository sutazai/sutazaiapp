Scope: Design, build, and operate agent services under /agents/ that integrate with the Backend Mesh and Ollama.

⚠️ Mandatory prerequisite

Read and comply with /opt/sutazaiapp/IMPORTANT/Enforcement_Rules before any change. This supersedes all docs. 
GitHub

Environment & Ports

Local LLM via Ollama at 10104 (default model: TinyLlama).

Core stack ports: backend 10010, frontend 10011, Postgres 10000, Redis 10001, Qdrant 10101/10102, Prometheus 10200, Grafana 10201. 
GitHub

Service contract (MUST)

All agents must expose:

GET /health → {status:"ok",version,uptime_s}

GET /ready → 200 only when deps are reachable and warm

GET /metrics → Prometheus text format

GET /version → {name,version,git_sha,build_time}

If HTTP job intake is supported:

POST /v1/jobs → 202 {job_id} (or 200 immediate result)

GET /v1/jobs/{job_id} → status/result

GET /v1/jobs/{job_id}/events → SSE (optional)

Message schema: agents/common/schemas.py (agentapi.v1)—do not break without a version bump.

Orchestration

Client → Backend Mesh → Orchestrator → Capability Agents (retrieve→plan→generate→verify) → Mesh → Client. Queue backpressure and idempotency keys are mandatory.

Configuration

Key env vars: AGENT_NAME, AGENT_PORT (11000+), AGENT_MAX_WORKERS, OLLAMA_BASE_URL, MODEL_ID, TEMPERATURE, MAX_TOKENS, REDIS_URL, PG_DSN, NEO4J_URI, VECTOR_PROVIDER, PROMETHEUS_NAMESPACE, LOG_LEVEL.

Security (MUST)

Least privilege credentials per agent.

No secrets in repo. Use env/secret store.

Strip PII in logs.

Allow‑list MCP tools per agent; denials logged.

Disable internet egress unless explicitly allowed by Enforcement Rules.

Observability

Metrics to emit:

sutazai_agents_tasks_total{agent,kind,status}

sutazai_agents_task_latency_ms_bucket{agent,kind}

sutazai_agents_queue_depth{agent}

sutazai_agents_model_tokens_total{agent,model,role}

sutazai_agents_errors_total{agent,type}

SLOs

Orchestrator decision latency ≤ 25 ms p50

Single agent task ≤ 500 ms p50 (non‑retrieval)

RAG end‑to‑end ≤ 8 s p50

Queue time ≤ 150 ms p50

Success rate ≥ 99.0% (non‑user errors)

Shipping a new agent (Checklist)

Reserve port in agents/PortRegistry.md.

Scaffold service with /health, /ready, /metrics, /version.

Enforce idempotency + deadlines.

Add tests (unit, contract, integration) — coverage ≥ 80%.

Register scrape in Prometheus and dashboards in Grafana.

Update this doc and PortRegistry.md.

Minimal template (FastAPI + Prometheus)

(same as in the previous message; keep in repo under agents/<name>/)

</details>
4) Make security & quality gates block PRs

Why: You already store scan outputs; convert them into blocking checks.
What to do:

New workflow .github/workflows/agents-ci.yml:

name: Agents CI (Tests + Lint + Security)
on:
  pull_request:
    paths:
      - "agents/**"
      - "backend/**"
      - ".github/workflows/**"
jobs:
  lint_test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Install
        run: pip install -r requirements/ci.txt || pip install black flake8 mypy pytest
      - name: Lint
        run: black --check . && flake8 && mypy agents backend
      - name: Tests
        run: pytest -m "unit or integration" --junitxml=reports/junit.xml --cov=agents --cov=backend
  container_scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Trivy image scan
        uses: aquasecurity/trivy-action@0.20.0
        with:
          scan-type: config
          exit-code: 1
          ignore-unfixed: true


This aligns with your existing structure (agents/, backend/, requirements/, security-scan-results/) and replaces ad‑hoc manual checks with CI gates. 
GitHub

5) Add idempotency + DLQ helpers to agents

Why: Reliable retries without duplicate side‑effects.
What to do: Add src/agents_utils/idempotency.py with Redis‑based locks, and standardize a Dead‑Letter Queue naming scheme agent.<name>.dlq (if you use AMQP, reflect in compose).

import time, hashlib, json
import redis

def idem_key(agent, request_body):
    digest = hashlib.sha256(json.dumps(request_body, sort_keys=True).encode()).hexdigest()
    return f"idem:{agent}:{digest}"

def with_idempotency(r: redis.Redis, agent: str, body: dict, ttl_s=900):
    key = idem_key(agent, body)
    ok = r.set(key, "1", nx=True, ex=ttl_s)
    return bool(ok)

6) Enforce Prometheus/Loki/OTEL consistency

Why: Agents drift unless you standardize labels and traces.
What to do: Add a shared logging module (src/agents_utils/logging.py) that emits JSON logs with trace_id, tenant_id, and agent labels, and a Prometheus helper that registers histograms/counters with a consistent namespace (sutazai_agents_*).

7) Clean the root: remove backups & debug artifacts; harden .gitignore

Why: Reduce noise and accidental drift.
What to do: Add these patterns:

Patch .gitignore:

+# Hygiene: ignore backups and local artifacts
+backups/
+*.backup
+*.backup.*
+*.bak
+*.orig
+.venvs/
+*.log
+*.html
+security-scan-results/*.json
+coverage.xml
+check-dashboard-live.html


The tree shows backups/, multiple docker-compose.yml.backup*, and HTML/scan reports committed; ignore going forward and remove from git. 
GitHub

8) Normalize secrets & MCP configs

Why: .mcp.json and secrets_secure/ exist—good signal, but lock them down.
What to do:

Encrypt secrets with SOPS (age) or use .env mounted from secret manager; never commit plaintext.

Keep one .mcp.json (drop debug backups), and run scripts/mcp/selfcheck_all.sh in CI to ensure MCP servers stay healthy. 
GitHub

9) Documentation: Cap matrix + auto‑inventory

Why: Let docs reflect reality automatically.
What to do: Ship scripts/agents/inventory.py that scans agents/*/ for Dockerfile/pyproject.toml, extracts ports, and emits a Markdown table into agents/PortRegistry.md. Run on CI and fail PR if the inventory changed but the table wasn’t updated.

10) Tests: formalize markers and contract tests

Why: Codify -m unit|integration|e2e|performance|security across the repo.
What to do: Add pytest.ini with marker definitions (if not present) and a small suite that probes /health and /metrics for each agent (spawn via compose).

11) Standardize Docker images (rootless + read‑only FS)

Why: Least privilege by default.
What to do: In each agent Dockerfile: add a non‑root user, set read_only: true in compose, and cap_drop: [ALL] unless explicitly required.

12) Tighten Nginx/Portainer exposure

Why: These are sharp edges.
What to do: Bind them to the Docker network only (no host publishing) unless you truly need host access; if public, enforce TLS and auth at the edge.