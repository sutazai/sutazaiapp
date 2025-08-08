# SutazAI Testing Suite Makefile
# Comprehensive testing automation and management

.PHONY: help install test test-unit test-integration test-e2e test-performance test-security test-docker test-health test-load test-all
.PHONY: coverage coverage-report lint format security-scan clean setup-dev setup-ci
.PHONY: docker-build docker-test docker-up docker-down services-up services-down
.PHONY: network mesh-up monitoring-up dbs-up core-up agents-up stack-up health
.PHONY: mcp-db-bootstrap
.PHONY: ensure-network up-minimal down-minimal ps-minimal logs-minimal restart-minimal health-minimal
.PHONY: deps-update deps-audit report-dashboard
.PHONY: docs-api docs-api-openapi docs-api-endpoints
.PHONY: onboarding-deck

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
POETRY := poetry
PYTEST := pytest
DOCKER_COMPOSE := docker-compose
PROJECT_ROOT := $(shell pwd)
VENV_DIR := venv-sutazaiapp

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)🧠 SutazAI Testing Suite$(NC)"
	@echo "$(CYAN)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment Setup
install: ## Install all dependencies
	@echo "$(YELLOW)📦 Installing dependencies...$(NC)"
	$(POETRY) install --with dev,security,performance
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

setup-dev: install ## Setup development environment
	@echo "$(YELLOW)🔧 Setting up development environment...$(NC)"
	$(POETRY) run pre-commit install
	@echo "$(GREEN)✅ Development environment ready$(NC)"

setup-ci: ## Setup CI environment
	@echo "$(YELLOW)🔧 Setting up CI environment...$(NC)"
	$(POETRY) install --no-dev
	@echo "$(GREEN)✅ CI environment ready$(NC)"

# Code Quality
lint: ## Run code linting
	@echo "$(YELLOW)🔍 Running linters...$(NC)"
	$(POETRY) run black --check backend/ frontend/ tests/ scripts/
	$(POETRY) run isort --check-only backend/ frontend/ tests/ scripts/
	$(POETRY) run flake8 backend/ frontend/ tests/ scripts/
	$(POETRY) run mypy backend/ --ignore-missing-imports
	@echo "$(GREEN)✅ Linting completed$(NC)"

format: ## Format code
	@echo "$(YELLOW)🎨 Formatting code...$(NC)"
	$(POETRY) run black backend/ frontend/ tests/ scripts/
	$(POETRY) run isort backend/ frontend/ tests/ scripts/
	@echo "$(GREEN)✅ Code formatted$(NC)"

security-scan: ## Run security scans
	@echo "$(YELLOW)🔒 Running security scans...$(NC)"
	$(POETRY) run bandit -r backend/ frontend/ -f json -o bandit-report.json
	$(POETRY) run safety check --json --output safety-report.json
	@echo "$(GREEN)✅ Security scan completed$(NC)"

# Testing Targets
test: test-unit ## Run default tests (unit tests)

test-unit: ## Run unit tests
	@echo "$(YELLOW)🧪 Running unit tests...$(NC)"
	$(POETRY) run python scripts/test_runner.py --type unit
	@echo "$(GREEN)✅ Unit tests completed$(NC)"

test-integration: services-up ## Run integration tests
	@echo "$(YELLOW)🔗 Running integration tests...$(NC)"
	$(POETRY) run python scripts/test_runner.py --type integration --services postgres,redis
	@echo "$(GREEN)✅ Integration tests completed$(NC)"

test-e2e: services-up ## Run end-to-end tests
	@echo "$(YELLOW)🌐 Running end-to-end tests...$(NC)"
	$(POETRY) run python scripts/test_runner.py --type e2e --browser chrome
	@echo "$(GREEN)✅ End-to-end tests completed$(NC)"

test-performance: services-up ## Run performance tests
	@echo "$(YELLOW)⚡ Running performance tests...$(NC)"
	$(POETRY) run python scripts/test_runner.py --type performance
	@echo "$(GREEN)✅ Performance tests completed$(NC)"

test-security: services-up ## Run security tests
	@echo "$(YELLOW)🛡️ Running security tests...$(NC)"
	$(POETRY) run python scripts/test_runner.py --type security
	@echo "$(GREEN)✅ Security tests completed$(NC)"

test-docker: ## Run Docker container tests
	@echo "$(YELLOW)🐳 Running Docker tests...$(NC)"
	$(POETRY) run python scripts/test_runner.py --type docker
	@echo "$(GREEN)✅ Docker tests completed$(NC)"

test-health: services-up ## Run health check tests
	@echo "$(YELLOW)🏥 Running health check tests...$(NC)"
	$(POETRY) run $(PYTEST) -m health tests/health/ -v
	@echo "$(GREEN)✅ Health check tests completed$(NC)"

test-load: services-up ## Run load tests
	@echo "$(YELLOW)📈 Running load tests...$(NC)"
	cd tests/load && python load_test_runner.py --host http://localhost:8000 --users 10 --run-time 300s
	@echo "$(GREEN)✅ Load tests completed$(NC)"

test-all: services-up ## Run all tests
	@echo "$(YELLOW)🚀 Running complete test suite...$(NC)"
	$(POETRY) run python scripts/test_runner.py --type all --quick
	@echo "$(GREEN)✅ All tests completed$(NC)"

test-comprehensive: services-up ## Run comprehensive test suite (long-running)
	@echo "$(YELLOW)🎯 Running comprehensive test suite...$(NC)"
	$(POETRY) run python scripts/test_runner.py --type all
	@echo "$(GREEN)✅ Comprehensive tests completed$(NC)"

# Coverage
coverage: ## Run tests with coverage
	@echo "$(YELLOW)📊 Running tests with coverage...$(NC)"
	$(POETRY) run python scripts/coverage_reporter.py --test-type all --threshold 80
	@echo "$(GREEN)✅ Coverage analysis completed$(NC)"

coverage-report: ## Generate coverage report
	@echo "$(YELLOW)📋 Generating coverage report...$(NC)"
	$(POETRY) run python scripts/coverage_reporter.py --no-tests
	@echo "$(GREEN)✅ Coverage report generated$(NC)"
	@echo "$(CYAN)📊 View report: file://$(PROJECT_ROOT)/htmlcov/index.html$(NC)"

report-dashboard: ## Generate test dashboard
	@echo "$(YELLOW)📈 Generating test dashboard...$(NC)"
	$(POETRY) run python scripts/coverage_reporter.py --test-type all
	@echo "$(CYAN)📊 View dashboard: file://$(PROJECT_ROOT)/coverage_reports/coverage_dashboard.html$(NC)"

# Docker Management
docker-build: ## Build Docker images
	@echo "$(YELLOW)🐳 Building Docker images...$(NC)"
	docker build -t sutazai-backend:latest -f backend/Dockerfile .
	docker build -t sutazai-frontend:latest -f frontend/Dockerfile .
	@echo "$(GREEN)✅ Docker images built$(NC)"

docker-test: docker-build ## Test Docker images
	@echo "$(YELLOW)🐳 Testing Docker images...$(NC)"
	docker run --rm sutazai-backend:latest python -c "import backend; print('Backend import successful')"
	docker run --rm sutazai-frontend:latest python -c "import streamlit; print('Frontend dependencies OK')"
	@echo "$(GREEN)✅ Docker images tested$(NC)"

network: ## Create docker network if missing
	@echo "$(YELLOW)📡 Ensuring docker network exists...$(NC)"
	@docker network create sutazai-network 2>/dev/null || true
	@echo "$(GREEN)✅ Network ready: sutazai-network$(NC)"

docker-up: network ## Start Docker services
	@echo "$(YELLOW)🐳 Starting Docker services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml up -d
	$(DOCKER_COMPOSE) -f docker-compose.yml ps
	@echo "$(GREEN)✅ Docker services started$(NC)"

# Minimal stack helpers (recommended)
ensure-network: ## Create external docker network if missing
	@echo "$(YELLOW)🔧 Ensuring external network 'sutazai-network' exists...$(NC)"
	@docker network ls | grep -q "sutazai-network" || docker network create sutazai-network
	@echo "$(GREEN)✅ Network ready$(NC)"

up-minimal: ensure-network ## Start minimal 8-service stack
	@echo "$(YELLOW)🐳 Starting minimal SutazAI stack...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.minimal.yml up -d
	$(DOCKER_COMPOSE) -f docker-compose.minimal.yml ps
	@echo "$(GREEN)✅ Minimal stack started$(NC)"

down-minimal: ## Stop minimal stack
	@echo "$(YELLOW)🐳 Stopping minimal SutazAI stack...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.minimal.yml down
	@echo "$(GREEN)✅ Minimal stack stopped$(NC)"

ps-minimal: ## Show minimal stack status
	$(DOCKER_COMPOSE) -f docker-compose.minimal.yml ps

logs-minimal: ## Tail minimal stack logs
	$(DOCKER_COMPOSE) -f docker-compose.minimal.yml logs -f --tail=200

restart-minimal: ## Restart minimal stack
	$(MAKE) down-minimal
	$(MAKE) up-minimal

health-minimal: ## Check minimal stack health endpoints
	@echo "$(YELLOW)🏥 Checking health endpoints...$(NC)"
	@echo "- Backend:    http://localhost:10010/health" && curl -sf http://localhost:10010/health | head -c 200 && echo || echo "$(RED)Backend health failed$(NC)"
	@echo "- Frontend:   http://localhost:10011/" && curl -sf http://localhost:10011/ | head -c 100 && echo || echo "$(RED)Frontend check failed$(NC)"
	@echo "- Ollama:     http://localhost:10104/" && curl -sf http://localhost:10104/ | head -c 100 && echo || echo "$(RED)Ollama check failed$(NC)"
	@echo "- Qdrant:     http://localhost:10101/healthz" && curl -sf http://localhost:10101/healthz | head -c 100 && echo || echo "$(RED)Qdrant check failed$(NC)"
	@echo "- Prometheus: http://localhost:10200/-/healthy" && curl -sf http://localhost:10200/-/healthy && echo || echo "$(RED)Prometheus health failed$(NC)"
	@echo "- Grafana:    http://localhost:10201/api/health" && curl -sf http://localhost:10201/api/health | head -c 200 && echo || echo "$(RED)Grafana health failed$(NC)"
	@echo "$(GREEN)✅ Health checks attempted$(NC)"

docker-down: ## Stop Docker services
	@echo "$(YELLOW)🐳 Stopping Docker services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml down
	@echo "$(GREEN)✅ Docker services stopped$(NC)"

# Service Management
services-up: ## Start test services (PostgreSQL, Redis, etc.)
	@echo "$(YELLOW)🔧 Starting test services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.test.yml up -d postgres redis
	@echo "$(CYAN)⏳ Waiting for services to be ready...$(NC)"
	@sleep 10
	@echo "$(GREEN)✅ Test services started$(NC)"

services-down: ## Stop test services
	@echo "$(YELLOW)🔧 Stopping test services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.test.yml down
	@echo "$(GREEN)✅ Test services stopped$(NC)"

services-status: ## Check service status
	@echo "$(YELLOW)🔍 Checking service status...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.test.yml ps
	@echo "$(CYAN)Health checks:$(NC)"
	@curl -s http://localhost:8000/health || echo "$(RED)Backend not responding$(NC)"
	@curl -s http://localhost:8501 || echo "$(RED)Frontend not responding$(NC)"

# Stack shortcuts
mesh-up: network ## Start Service Mesh (Kong, Consul, RabbitMQ)
	@echo "$(YELLOW)🕸  Starting service mesh (Kong, Consul, RabbitMQ)...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml up -d kong consul rabbitmq
	@echo "$(GREEN)✅ Service mesh started$(NC)"

monitoring-up: network ## Start Monitoring stack (Prometheus, Grafana, Loki, cAdvisor)
	@echo "$(YELLOW)📊 Starting monitoring stack...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml up -d prometheus grafana loki promtail cadvisor node-exporter alertmanager blackbox-exporter redis-exporter postgres-exporter
	@echo "$(GREEN)✅ Monitoring stack started$(NC)"

dbs-up: network ## Start databases (Postgres, Redis, Neo4j, ChromaDB, Qdrant, FAISS)
	@echo "$(YELLOW)🗄  Starting databases...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml up -d postgres redis neo4j chromadb qdrant faiss
	@echo "$(GREEN)✅ Databases started$(NC)"

core-up: network ## Start core app (Ollama, Backend, Frontend)
	@echo "$(YELLOW)🧩 Starting core services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml up -d ollama backend frontend
	@echo "$(GREEN)✅ Core services started$(NC)"

agents-up: network ## Start key agent services
	@echo "$(YELLOW)🤖 Starting agent services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml up -d agentgpt agentzero autogen autogpt crewai aider shellgpt context-framework jarvis-voice-interface jarvis-knowledge-management jarvis-automation-agent jarvis-multimodal-ai mcp-server
	@echo "$(GREEN)✅ Agent services started$(NC)"

stack-up: ## Start full platform (infra + mesh + monitoring + core + agents)
	@$(MAKE) dbs-up
	@$(MAKE) mesh-up
	@$(MAKE) monitoring-up
	@$(MAKE) core-up
	@$(MAKE) agents-up
	@echo "$(GREEN)✅ Full SutazAI stack started$(NC)"

health: ## Run platform health checks
	@echo "$(YELLOW)🩺 Running health checks...$(NC)"
	-@curl -fsS http://localhost:10010/health && echo " OK: backend" || echo "$(RED)FAIL: backend$(NC)"
	-@curl -fsS http://localhost:10104/ || echo "$(RED)WARN: ollama root responds (expected tags at /api/tags)$(NC)"
	-@curl -fsS http://localhost:10006/v1/status/leader && echo " OK: consul" || echo "$(RED)FAIL: consul$(NC)"
	-@curl -fsS http://localhost:10005/ || echo "$(RED)WARN: kong proxy reachable (admin at 10015)$(NC)"
	-@curl -fsS http://localhost:10008/api/health/checks/virtual-hosts && echo " OK: rabbitmq" || echo "$(RED)FAIL: rabbitmq$(NC)"
	-@curl -fsS http://localhost:10200/-/healthy && echo " OK: prometheus" || echo "$(RED)FAIL: prometheus$(NC)"
	-@curl -fsS http://localhost:10201/api/health && echo " OK: grafana" || echo "$(RED)FAIL: grafana$(NC)"
	-@curl -fsS http://localhost:10202/ready && echo " OK: loki" || echo "$(RED)FAIL: loki$(NC)"
	-@curl -fsS http://localhost:10206/ && echo " OK: cadvisor" || echo "$(RED)FAIL: cadvisor$(NC)"
	-@curl -fsS http://localhost:10100/api/v1/heartbeat && echo " OK: chromadb" || echo "$(RED)FAIL: chromadb$(NC)"
	-@curl -fsS http://localhost:10101/ && echo " OK: qdrant (HTTP 200 expected)" || echo "$(RED)FAIL: qdrant$(NC)"
	-@curl -fsS http://localhost:10002/ && echo " OK: neo4j UI" || echo "$(RED)FAIL: neo4j$(NC)"
	-@curl -fsS http://localhost:11190/health && echo " OK: mcp-server" || echo "$(RED)FAIL: mcp-server$(NC)"
	-@curl -fsS http://localhost:10005/mcp/health && echo " OK: kong->mcp route" || echo "$(RED)FAIL: kong->mcp route$(NC)"
	@echo "$(GREEN)✅ Health checks complete$(NC)"

mcp-db-bootstrap: ## Create MCP tables in Postgres (idempotent)
	@echo "$(YELLOW)🗄  Bootstrapping MCP DB schema...$(NC)"
	@docker cp scripts/mcp_bootstrap.sql sutazai-postgres:/tmp/mcp_bootstrap.sql
	@docker compose exec -T postgres psql -U $${POSTGRES_USER:-sutazai} -d $${POSTGRES_DB:-sutazai} -v ON_ERROR_STOP=1 -f /tmp/mcp_bootstrap.sql
	@echo "$(GREEN)✅ MCP DB schema ready$(NC)"

# Dependency Management
deps-update: ## Update dependencies
	@echo "$(YELLOW)📦 Updating dependencies...$(NC)"
	$(POETRY) update
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

deps-audit: ## Audit dependencies for security issues
	@echo "$(YELLOW)🔍 Auditing dependencies...$(NC)"
	$(POETRY) run safety check
	$(POETRY) run pip-audit
	@echo "$(GREEN)✅ Dependency audit completed$(NC)"

# Clean up
clean: ## Clean up test artifacts and caches
	@echo "$(YELLOW)🧹 Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .pytest_cache/ .mypy_cache/ .tox/ dist/ build/
	rm -f *.log *.xml *.json coverage.* mprofile_*
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

clean-docker: ## Clean up Docker resources
	@echo "$(YELLOW)🐳 Cleaning Docker resources...$(NC)"
	docker system prune -af
	docker volume prune -f
	@echo "$(GREEN)✅ Docker cleanup completed$(NC)"

# CI/CD Helpers
ci-install: ## Install dependencies for CI
	@echo "$(YELLOW)🔧 Installing CI dependencies...$(NC)"
	$(PIP) install poetry
	$(POETRY) install --no-dev
	@echo "$(GREEN)✅ CI dependencies installed$(NC)"

ci-test: ## Run CI test suite
	@echo "$(YELLOW)🚀 Running CI test suite...$(NC)"
	$(MAKE) lint
	$(MAKE) security-scan
	$(MAKE) test-unit
	$(MAKE) coverage
	@echo "$(GREEN)✅ CI tests completed$(NC)"

ci-test-full: ## Run full CI test suite
	@echo "$(YELLOW)🎯 Running full CI test suite...$(NC)"
	$(MAKE) ci-test
	$(MAKE) test-integration
	$(MAKE) test-security
	$(MAKE) test-docker
	@echo "$(GREEN)✅ Full CI tests completed$(NC)"

# Quality Gates
quality-gate: ## Run quality gate checks
	@echo "$(YELLOW)🚪 Running quality gate checks...$(NC)"
	$(MAKE) lint
	$(MAKE) security-scan
	$(MAKE) test-unit
	$(MAKE) coverage
	@echo "$(GREEN)✅ Quality gate passed$(NC)"

quality-gate-strict: ## Run strict quality gate checks
	@echo "$(YELLOW)🚪 Running strict quality gate checks...$(NC)"
	$(MAKE) quality-gate
	$(MAKE) test-integration
	$(MAKE) test-security
	$(MAKE) test-performance
	@echo "$(GREEN)✅ Strict quality gate passed$(NC)"

# Status and Information
status: ## Show project status
	@echo "$(BLUE)🧠 SutazAI Project Status$(NC)"
	@echo "$(CYAN)Python Version:$(NC) $(shell python --version)"
	@echo "$(CYAN)Poetry Version:$(NC) $(shell poetry --version)"
	@echo "$(CYAN)Docker Version:$(NC) $(shell docker --version)"
	@echo "$(CYAN)Project Root:$(NC) $(PROJECT_ROOT)"
	@echo "$(CYAN)Last Test Run:$(NC) $(shell ls -la test-results.xml 2>/dev/null | awk '{print $$6, $$7, $$8}' || echo 'No recent tests')"

info: status ## Alias for status

# Legacy deployment targets (preserved)
deploy-dev: ## Deploy to development
	@echo "$(YELLOW)🚀 Deploying to development...$(NC)"
	./deploy_sutazai_baseline.sh

deploy-prod: ## Deploy to production
	@echo "$(RED)⚠️  WARNING: This will deploy to production. Continue? [y/N]$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ]; then \
		./deploy_sutazai_v9_complete.sh; \
	else \
		echo "$(YELLOW)Deployment cancelled.$(NC)"; \
	fi

# Database operations
db-migrate: ## Run database migrations
	@echo "$(YELLOW)🗃️ Running database migrations...$(NC)"
	docker-compose exec backend alembic upgrade head

db-rollback: ## Roll back last migration
	@echo "$(YELLOW)⏪ Rolling back last migration...$(NC)"
	docker-compose exec backend alembic downgrade -1

# Development helpers
shell-backend: ## Open backend shell
	@echo "$(YELLOW)🐚 Opening backend shell...$(NC)"
	docker-compose exec backend /bin/bash

shell-db: ## Open database shell
	@echo "$(YELLOW)🗃️ Opening database shell...$(NC)"
	docker-compose exec postgres psql -U sutazai -d sutazai

logs: ## Show service logs
	@echo "$(YELLOW)📋 Showing logs...$(NC)"
	docker-compose logs -f

# Performance monitoring
monitor: ## Open monitoring dashboards
	@echo "$(YELLOW)📊 Opening monitoring dashboards...$(NC)"
	@echo "$(CYAN)Prometheus: http://localhost:9090$(NC)"
	@echo "$(CYAN)Grafana: http://localhost:3000$(NC)"
	@xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000"

# Generate documentation
docs: ## Generate documentation
	@echo "$(YELLOW)📚 Generating documentation...$(NC)"
	cd backend && python -m sphinx -b html docs/ docs/_build/html
	@echo "$(GREEN)✅ Documentation generated at backend/docs/_build/html/$(NC)"

docs-api: docs-api-openapi docs-api-endpoints ## Export OpenAPI and endpoint summary
	@echo "$(GREEN)✅ API documentation artifacts updated$(NC)"

docs-api-openapi: ## Export backend OpenAPI to docs/backend_openapi.json
	@echo "$(YELLOW)📜 Exporting OpenAPI...$(NC)"
	python3 scripts/export_openapi.py

docs-api-endpoints: ## Generate Markdown endpoints summary from OpenAPI
	@echo "$(YELLOW)🗂  Generating endpoints summary...$(NC)"
	python3 scripts/summarize_openapi.py

# Version management
version: ## Show current version
	@echo "$(CYAN)Current version:$(NC)"
	@grep -E "^version" backend/pyproject.toml || echo "version = \"9.0.0\""

# Complete system check
check: lint test security-scan ## Run complete system check
	@echo "$(GREEN)✅ All checks passed!$(NC)"

# Onboarding deck generation (requires python-pptx)
onboarding-deck: ## Generate onboarding PPTX deck from overview
	@echo "$(YELLOW)📑 Generating onboarding deck...$(NC)"
	python scripts/onboarding/generate_kickoff_deck.py
	@echo "$(GREEN)✅ Deck generated at docs/onboarding/kickoff_deck_v1.pptx$(NC)"
# --- MCP Orchestration ---
.PHONY: mcp-build mcp-up mcp-down mcp-logs mcp-restart mcp-health e2e-mcp

mcp-build:
	docker compose -f docker-compose.mcp.yml build

mcp-up:
	docker compose -f docker-compose.mcp.yml up -d

mcp-down:
	docker compose -f docker-compose.mcp.yml down

mcp-logs:
	docker compose -f docker-compose.mcp.yml logs -f mcp-server

mcp-restart: mcp-down mcp-up

mcp-health:
	curl -sSf http://localhost:3030/health | jq .

e2e-mcp:
	npx playwright test e2e/mcp-health.spec.ts

.PHONY: mcp-bootstrap mcp-teardown
mcp-bootstrap:
	bash scripts/mcp_bootstrap.sh

mcp-teardown:
	bash scripts/mcp_teardown.sh

.PHONY: add-mcp-tool
add-mcp-tool:
	@if [ -z "$(IMG)" ]; then echo "Usage: make add-mcp-tool IMG=<docker-image>"; exit 1; fi
	bash scripts/add_mcp_tool.sh $(IMG)
