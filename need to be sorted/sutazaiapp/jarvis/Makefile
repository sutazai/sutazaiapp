# SutazAI Testing Suite Makefile
# Comprehensive testing automation and management

.PHONY: help install test test-unit test-integration test-e2e test-performance test-security test-docker test-health test-load test-all
.PHONY: coverage coverage-report lint format security-scan clean setup-dev setup-ci
.PHONY: docker-build docker-test docker-up docker-down services-up services-down
.PHONY: deps-update deps-audit report-dashboard

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

docker-up: ## Start Docker services
	@echo "$(YELLOW)🐳 Starting Docker services...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml up -d
	$(DOCKER_COMPOSE) -f docker-compose.yml ps
	@echo "$(GREEN)✅ Docker services started$(NC)"

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

# Version management
version: ## Show current version
	@echo "$(CYAN)Current version:$(NC)"
	@grep -E "^version" backend/pyproject.toml || echo "version = \"9.0.0\""

# Complete system check
check: lint test security-scan ## Run complete system check
	@echo "$(GREEN)✅ All checks passed!$(NC)"