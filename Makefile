# SutazAI Makefile for Development and Testing

.PHONY: help build test test-unit test-integration test-security test-performance clean docker-up docker-down lint format security-scan

# Default target
help:
	@echo "SutazAI Development Commands:"
	@echo "  make build          - Build all Docker images"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-security  - Run security tests"
	@echo "  make test-performance - Run performance tests"
	@echo "  make lint           - Run code linters"
	@echo "  make format         - Format code"
	@echo "  make security-scan  - Run security scans"
	@echo "  make docker-up      - Start all services"
	@echo "  make docker-down    - Stop all services"
	@echo "  make clean          - Clean up generated files"
	@echo "  make deploy-dev     - Deploy to development"
	@echo "  make deploy-prod    - Deploy to production"

# Build Docker images
build:
	@echo "Building Docker images..."
	docker-compose build
	docker-compose -f docker-compose.test.yml build

# Run all tests
test: test-unit test-integration test-security

# Run unit tests
test-unit:
	@echo "Running unit tests..."
	cd backend && python -m pytest tests/unit -v --cov=app --cov-report=term-missing

# Run integration tests
test-integration: docker-up-test
	@echo "Running integration tests..."
	docker-compose -f docker-compose.test.yml run --rm test-runner
	$(MAKE) docker-down-test

# Run security tests
test-security:
	@echo "Running security tests..."
	cd backend && python -m pytest tests/unit/test_security_real.py -v -m security
	@echo "Running vulnerability scans..."
	docker run --rm -v "$(PWD)":/src aquasec/trivy fs /src

# Run performance tests
test-performance: docker-up
	@echo "Running performance tests..."
	docker-compose -f docker-compose.test.yml run --rm k6-runner
	@echo "Performance test results saved to test-results/"

# Code quality checks
lint:
	@echo "Running linters..."
	cd backend && python -m flake8 app/ --max-line-length=120 --extend-ignore=E203,W503
	cd backend && python -m pylint app/ --fail-under=8.0
	cd backend && python -m mypy app/ --ignore-missing-imports

# Format code
format:
	@echo "Formatting code..."
	cd backend && python -m black app/ tests/
	cd backend && python -m isort app/ tests/

# Security scanning
security-scan:
	@echo "Running security scans..."
	cd backend && python -m bandit -r app/ -ll
	cd backend && python -m safety check -r requirements.txt
	@echo "Scanning Docker images..."
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		-v "$(PWD)":/src aquasec/trivy image sutazai-backend:latest

# Docker operations
docker-up:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@docker-compose ps

docker-down:
	@echo "Stopping services..."
	docker-compose down -v

docker-up-test:
	@echo "Starting test services..."
	docker-compose -f docker-compose.test.yml up -d postgres-test redis-test chromadb-test
	@sleep 10

docker-down-test:
	@echo "Stopping test services..."
	docker-compose -f docker-compose.test.yml down -v

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf backend/htmlcov
	rm -rf backend/test-results
	rm -rf performance-results
	rm -rf .pytest_cache
	rm -rf .mypy_cache

# Deployment targets
deploy-dev:
	@echo "Deploying to development..."
	./deploy_sutazai_baseline.sh

deploy-prod:
	@echo "Deploying to production..."
	@echo "WARNING: This will deploy to production. Continue? [y/N]"
	@read -r response; \
	if [ "$$response" = "y" ]; then \
		./deploy_sutazai_v9_complete.sh; \
	else \
		echo "Deployment cancelled."; \
	fi

# Database operations
db-migrate:
	@echo "Running database migrations..."
	docker-compose exec backend alembic upgrade head

db-rollback:
	@echo "Rolling back last migration..."
	docker-compose exec backend alembic downgrade -1

# Development helpers
shell-backend:
	@echo "Opening backend shell..."
	docker-compose exec backend /bin/bash

shell-db:
	@echo "Opening database shell..."
	docker-compose exec postgres psql -U sutazai -d sutazai

logs:
	@echo "Showing logs..."
	docker-compose logs -f

# CI/CD helpers
ci-test:
	@echo "Running CI tests..."
	docker-compose -f docker-compose.test.yml up --abort-on-container-exit --exit-code-from backend-test

ci-build:
	@echo "Building for CI..."
	docker build -t sutazai-backend:ci ./backend --target test
	docker build -t sutazai-frontend:ci ./frontend

# Performance monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000"
	@xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000"

# Generate documentation
docs:
	@echo "Generating documentation..."
	cd backend && python -m sphinx -b html docs/ docs/_build/html
	@echo "Documentation generated at backend/docs/_build/html/"

# Version management
version:
	@echo "Current version:"
	@grep -E "^version" backend/pyproject.toml || echo "version = \"9.0.0\""

# Complete system check
check: lint test security-scan
	@echo "All checks passed!"