# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ MANDATORY PREREQUISITE - READ FIRST

**🚨 CRITICAL:** Before executing ANY code changes, modifications, or development tasks in this repository, you MUST first read and understand the complete Enforcement Rules document:

**👉 `/opt/sutazaiapp/IMPORTANT/Enforcement_Rules`**

This 356KB comprehensive document contains:
- 🔧 Professional Codebase Standards & Hygiene Guide
- 🎯 Core Non-Negotiable Standards 
- 🧼 Detailed Codebase Hygiene Requirements
- 🔒 Security and Quality Enforcement Rules
- 📋 Complete Rule Set with Implementation Details

**NO EXCEPTIONS:** All work in this codebase MUST comply with the standards defined in the Enforcement Rules. Failure to review these rules prior to execution will result in code that violates established patterns and may be rejected.

**Last Updated:** 2024-12-20 17:00:00 UTC  
**System Version:** SutazAI v89 - Local AI Automation Platform  
**Status:** Production Ready ✅  
**Architecture:** Multi-tier containerized system with 25 operational services

## Project Overview

SutazAI is a comprehensive local AI automation platform designed for enterprise deployment without external AI service dependencies. The system provides:

- **Local AI Processing**: Ollama with TinyLlama model for on-premises AI capabilities
- **Multi-Database Architecture**: PostgreSQL, Redis, Neo4j, ChromaDB, Qdrant, FAISS
- **FastAPI Backend**: High-performance API server with async support
- **Streamlit Frontend**: Modern web interface for AI automation
- **Agent System**: 7+ operational AI agents for various automation tasks
- **Vector Intelligence**: Multiple vector databases for semantic search and AI workflows
- **Monitoring Stack**: Prometheus, Grafana, Loki for comprehensive observability
- **MCP Integration**: 17+ Model Context Protocol servers for extended AI capabilities

## Quick Start Commands

### Core System Operations
```bash
# Start core services (database, cache, backend, frontend)
make up-core

# Start full stack including monitoring and agents
make stack-up

# View system status
make status

# Stop all services
make down
```

### Development Workflow
```bash
# Install dependencies and setup development environment
make install
make setup-dev

# Run all tests
make test

# Run specific test types
make test-unit           # Unit tests only
make test-integration    # Integration tests
make test-e2e           # End-to-end tests
make test-performance   # Performance benchmarks

# Code quality checks
make lint               # Run linting (black, flake8, mypy)
make format            # Auto-format code
make coverage          # Generate test coverage report

# Run single test file
pytest backend/tests/test_specific.py -v

# Run specific test function
pytest backend/tests/test_file.py::test_function_name -v
```

### System Monitoring
```bash
# Start monitoring stack
make monitoring-up

# View system metrics
curl http://localhost:10010/health      # Backend health
curl http://localhost:10104/api/tags    # Ollama models

# Access monitoring dashboards
open http://localhost:10201             # Grafana (admin/admin)
open http://localhost:10200             # Prometheus
open http://localhost:10202             # Loki logs
```

## Architecture Overview

### Port Allocation (Complete Registry)
The system uses structured port allocation in the 10000+ range:

**Infrastructure Services (10000-10199)**
- 10000: PostgreSQL database
- 10001: Redis cache  
- 10002-10003: Neo4j graph database (bolt/http)
- 10007-10008: RabbitMQ (amqp/mgmt)
- 10010: FastAPI backend
- 10011: Streamlit frontend

**Vector & AI Services (10100-10199)**  
- 10100: ChromaDB vector database
- 10101-10102: Qdrant vector database
- 10103: FAISS vector service
- 10104: Ollama AI model server (**RESERVED - CRITICAL**)

**Monitoring Stack (10200-10299)**
- 10200: Prometheus metrics
- 10201: Grafana dashboards
- 10202: Loki log aggregation
- 10203: AlertManager
- 10220-10221: Node exporters and cAdvisor

**AI Agents (11000+)**
- 11000+: Various AI automation agents (see PortRegistry.md for complete list)

### Service Dependencies
Understanding the service startup order and dependencies:

1. **Core Infrastructure**: PostgreSQL → Redis → Neo4j
2. **Vector Databases**: ChromaDB, Qdrant, FAISS (can start in parallel)
3. **AI Services**: Ollama model server (independent)
4. **Application Layer**: Backend API (depends on databases) → Frontend (depends on backend)
5. **Monitoring**: Independent services that can start in any order
6. **Agent Services**: Depend on core infrastructure and AI services

### Key Configuration Files

**Docker & Deployment:**
- `docker-compose.yml` - Core service definitions with resource limits
- `Makefile` - Build, test, and deployment automation
- `.env.example` - Environment variables template

**Backend Configuration:**  
- `backend/requirements.txt` - Python dependencies with security patches
- `backend/app/main.py` - FastAPI application with performance optimizations
- `backend/app/core/` - Core configurations and utilities

**Frontend Configuration:**
- Uses Streamlit with modular architecture
- Custom components for AI interactions
- Real-time monitoring dashboard integration

## Development Guidelines

### Code Quality Requirements
- **Python 3.11+** required for all backend components
- **Type hints** mandatory for new code
- **Test coverage** minimum 80% for new features
- **Security scanning** with bandit and safety
- **Linting** with black, flake8, mypy (configured in pyproject.toml)

### Database Schema Management
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Database connection testing
python -c "from backend.app.core.database import get_db; print('DB OK')"
```

### AI Model Management
```bash
# Check loaded models
curl http://localhost:10104/api/tags

# Model health check  
curl http://localhost:10104/api/version

# Backend AI integration test
curl -X POST http://localhost:10010/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "model": "tinyllama"}'
```

## MCP Server Management

The system includes 17+ Model Context Protocol servers for extended AI capabilities:

### Critical MCP Servers
- **file operations**: Local file system access
- **postgres**: Database operations and queries  
- **github**: GitHub repository management
- **browser automation**: Playwright/Puppeteer web automation
- **extended-memory**: Persistent memory across sessions

### MCP Commands
```bash
# Test all MCP servers
scripts/mcp/selfcheck_all.sh

# Test individual MCP server
scripts/mcp/wrappers/[server-name].sh --selfcheck

# List MCP status (in Claude)
/mcp list
```

⚠️ **MCP Protection**: MCP servers are protected infrastructure. Never modify `.mcp.json` or wrapper scripts without explicit authorization.

## Troubleshooting Guide

### Common Issues

**Service Won't Start:**
```bash
# Check service logs
docker-compose logs [service-name]

# Verify port availability
ss -tulpn | grep [port-number]

# Check resource usage
docker stats
```

**Database Connection Issues:**
```bash
# Test PostgreSQL connection
docker exec -it sutazai-postgres pg_isready -U sutazai

# Test Redis connection
docker exec -it sutazai-redis redis-cli ping

# Reset database (CAUTION: destroys data)
make db-reset
```

**AI Model Issues:**
```bash
# Check Ollama status
curl http://localhost:10104/api/tags

# Restart Ollama service
docker-compose restart ollama

# Check model loading
docker exec -it ollama ollama list
```

**Performance Issues:**
```bash
# Monitor resource usage
make monitoring-up
# Then access Grafana at http://localhost:10201

# Check container resource limits
docker inspect [container-name] | grep -A 10 Resources
```

### Health Check Endpoints
- Backend API: `GET http://localhost:10010/health`
- Frontend: `GET http://localhost:10011/` 
- PostgreSQL: `docker exec sutazai-postgres pg_isready`
- Redis: `docker exec sutazai-redis redis-cli ping`
- Ollama: `GET http://localhost:10104/api/version`

## Security Considerations

**Current Security Status: 88% Hardened**
- 22/25 containers run as non-root users
- JWT authentication with bcrypt password hashing
- Environment-based secrets management
- Regular security patches in requirements.txt

**Remaining Security Tasks:**
- Migrate 3 remaining containers to non-root users
- Enable SSL/TLS for production deployment
- Implement advanced secrets management

## Performance Characteristics

**System Requirements:**
- **Minimum**: 8GB RAM, 4 CPU cores, 50GB storage
- **Recommended**: 16GB RAM, 8 CPU cores, 100GB storage
- **Docker**: Version 20.0+ with Docker Compose

**Expected Performance:**
- Backend API response: <100ms for standard requests
- AI model inference: 5-8 seconds (TinyLlama)
- Database operations: <50ms for standard queries
- Full system startup: 2-3 minutes

## Container Architecture

The system runs 25 operational containers organized in tiers:

**Tier 1: Core Infrastructure (5 containers)**
- PostgreSQL, Redis, Neo4j databases
- FastAPI backend, Streamlit frontend

**Tier 2: AI & Vector Services (6 containers)**  
- Ollama AI model server
- ChromaDB, Qdrant, FAISS vector databases
- Vector processing services

**Tier 3: Agent Services (7 containers)**
- Hardware Resource Optimizer
- Jarvis Automation Agent
- Task Assignment Coordinator
- Resource Arbitration Agent
- AI Agent Orchestrator
- Ollama Integration Agent
- Jarvis Hardware Optimizer

**Tier 4: Monitoring Stack (7 containers)**
- Prometheus, Grafana, Loki
- Various exporters and monitoring tools

## API Documentation

**Primary API Endpoints:**
- `POST /api/v1/chat/` - Chat with AI models
- `GET /api/v1/models/` - List available models
- `POST /api/v1/mesh/enqueue` - Task queue operations
- `GET /api/v1/mesh/results` - Task results
- `GET /health` - System health status
- `GET /metrics` - Prometheus metrics

**Interactive Documentation:**
- Swagger UI: http://localhost:10010/docs
- ReDoc: http://localhost:10010/redoc

## Important Files and Directories

### Configuration Files
- `/CLAUDE.md` - This developer guidance file
- `/README.md` - Project overview and quick start
- `/docker-compose.yml` - Service definitions
- `/Makefile` - Build and deployment automation
- `/.env.example` - Environment variables template

### Source Code Structure
- `/backend/` - FastAPI application source
  - `/app/main.py` - Application entry point
  - `/app/api/` - API route definitions  
  - `/app/core/` - Core configurations
  - `/app/models/` - Database models
  - `/tests/` - Backend test suite
- `/frontend/` - Streamlit application
- `/agents/` - AI agent implementations
- `/scripts/` - Utility and deployment scripts
- `/monitoring/` - Monitoring configurations

### Critical Directories
- `/IMPORTANT/` - Critical system documentation and architecture diagrams
- `/scripts/mcp/` - MCP server wrapper scripts (**PROTECTED**)
- `/config/` - Service configuration files
- `/logs/` - Application logs (development)

## Best Practices

### Development Workflow
1. Always run `make test` before committing code
2. Use `make lint` to ensure code quality
3. Update tests when modifying functionality  
4. Follow semantic versioning for releases
5. Document API changes in OpenAPI specs

### Deployment Guidelines
1. Use `make up-core` for basic development
2. Use `make stack-up` for full feature testing
3. Monitor resource usage with Grafana
4. Backup databases before major changes
5. Test rollback procedures

### Monitoring Best Practices
1. Check Grafana dashboards for system health
2. Monitor container resource usage
3. Review application logs in Loki
4. Set up alerts for critical metrics
5. Regular performance baseline updates

This CLAUDE.md file serves as the complete developer reference for the SutazAI system. For additional technical details, refer to the README.md and documentation in the `/IMPORTANT/` directory.