# Backend + API Configuration (DEEP DIVE - 2025-08-21)

## Architecture
- **Framework**: FastAPI + Uvicorn
- **Main API**: `/backend/app/api/v1/api.py`
- **URL**: http://localhost:10010
- **Container**: sutazai-backend

## API Endpoints (23 Found)
```
/api/v1/agents        - Agent management
/api/v1/models        - Model operations
/api/v1/documents     - Document handling
/api/v1/chat          - Chat interface
/api/v1/system        - System operations
/api/v1/hardware      - Hardware monitoring
/api/v1/cache         - Caching layer
/api/v1/cache-optimized - Optimized cache
/api/v1/circuit-breaker - Fault tolerance
/api/v1/mesh          - Legacy Redis mesh
/api/v1/mesh/v2       - Real service mesh
/api/v1/mcp           - MCP server integration
/api/v1/features      - Feature flags
```

## Services (12 Core Services)
- agent_registry.py - Agent registration
- consolidated_ollama_service.py - Ollama integration
- faiss_manager.py - FAISS vector DB
- mcp_client.py - MCP client
- mcp_service_discovery.py - Service discovery
- memory_migration.py - Memory management
- rate_limiter.py - Rate limiting
- self_improvement.py - Self-optimization
- vector_context_injector.py - Context injection
- vector_db_manager.py - Vector DB management

## AI Agents (12 Modules)
- agent_factory.py - Agent creation
- agent_manager.py - Lifecycle management
- workflow_orchestrator.py - Multi-agent workflows
- universal_agent_adapter.py - Adapter pattern
- communication_protocols.py - Agent communication
- discovery_service.py - Agent discovery
- health_check.py - Health monitoring

## Mesh Integration (20 Files)
- Service mesh with Consul
- DinD (Docker-in-Docker) bridge
- MCP container orchestration
- Load balancing
- Protocol translation
- Resource isolation
- Distributed tracing

## Database Models
- Pydantic models for validation
- SQLAlchemy for ORM
- Support for PostgreSQL, Redis, Neo4j
- Vector databases (Qdrant, ChromaDB)

## Dependencies (Key)
- fastapi==0.115.6
- sqlalchemy==2.0.36
- redis==5.2.1
- neo4j==5.27.0
- qdrant-client==1.12.1
- docker==7.1.0 (for DinD)