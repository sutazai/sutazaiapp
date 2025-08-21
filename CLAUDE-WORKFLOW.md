# Development Workflow (DEEP DIVE - 2025-08-21)

## Project Structure
```
/opt/sutazaiapp/
├── backend/          # FastAPI backend
│   ├── ai_agents/    # 12 AI agent modules
│   ├── app/
│   │   ├── api/v1/   # 23 API endpoints
│   │   ├── mesh/     # 20 service mesh files
│   │   └── services/ # 12 core services
│   └── edge_inference/ # Edge AI capabilities
├── frontend/         # Streamlit UI
│   ├── pages/        # 4 main features
│   └── components/   # UI components
├── tests/            # 313 test files
├── scripts/          # Automation scripts
├── docker/           # Docker configs
└── .claude/agents/   # 200+ agent definitions
```

## Testing Infrastructure
- **Test Files**: 313 Python test files
- **Testing Tools**: 
  - Playwright for E2E
  - Newman for API testing
  - K6 for load testing
  - Jest for unit tests

## Scripts & Automation
- Hardware optimization scripts
- MCP wrapper scripts
- Database migration scripts
- Docker consolidation scripts
- Analysis and compliance tools

## Agent System (200+ Agents)
Actual agent definition files in `.claude/agents/`:
- Each agent has 20-300KB markdown definition
- Categories: Development, Testing, Security, Infrastructure, AI/ML
- Examples: adversarial-attack-detector, agent-architect, ai-engineer

## Build & Deploy
```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && pip install -r requirements.txt
streamlit run app.py

# Docker
docker-compose up -d

# Tests
npm test  # API tests
npm run test:e2e  # E2E tests
npm run test:load  # Load tests
```

## Environment Configuration
Required `.env` variables:
- POSTGRES_PASSWORD
- NEO4J_PASSWORD
- JWT_SECRET
- GRAFANA_PASSWORD
- CHROMADB_API_KEY

## Development Tools
- SPARC methodology integration
- Claude-Flow orchestration
- MCP server management
- Service mesh with Consul
- DinD for container isolation