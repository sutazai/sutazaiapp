# Docker Service Analysis and Categorization
**Date**: 2025-08-08  
**Total Services Defined**: 59  
**Currently Running**: 28 (based on system reality)  

## Service Classification

### ESSENTIAL Services (Keep - 20 services)
*These are core infrastructure and working services that must be retained*

#### Core Infrastructure (7)
1. **postgres** - PostgreSQL database (port 10000)
2. **redis** - Cache layer (port 10001) 
3. **neo4j** - Graph database (port 10002/10003)
4. **ollama** - LLM service with tinyllama (port 10104)
5. **backend** - FastAPI application (port 10010)
6. **frontend** - Streamlit UI (port 10011)
7. **kong** - API gateway (port 10005/10015) - for future routing

#### Service Mesh & Orchestration (3)  
8. **consul** - Service discovery (port 10006)
9. **rabbitmq** - Message queue (port 10007/10008)
10. **mcp-server** - Message Control Protocol server (port 11190)

#### Monitoring Stack (6)
11. **prometheus** - Metrics collection (port 10200)
12. **grafana** - Visualization dashboards (port 10201) 
13. **loki** - Log aggregation (port 10202)
14. **alertmanager** - Alert routing (port 11108)
15. **node-exporter** - System metrics (port 10205)
16. **cadvisor** - Container metrics (port 10206)

#### Vector Databases (3)
17. **chromadb** - Vector storage (port 10100)
18. **qdrant** - Vector search (port 10101/10102) 
19. **faiss** - Vector similarity (port 10103)

#### Working Agent (1)
20. **ollama-integration** - Ollama wrapper agent (port 8090) - only functional agent

### STUB Services (Comment out - 21 services)
*These are placeholder services returning hardcoded responses*

#### Agent Stubs (7)
1. **agentgpt** - AgentGPT stub (port 11066)
2. **agentzero** - AgentZero stub (port 11067)  
3. **hardware-resource-optimizer** - Hardware monitoring stub (port 11110)
4. **jarvis-voice-interface** - Voice interface stub (port 11150)
5. **jarvis-knowledge-management** - Knowledge management stub (port 11101)
6. **jarvis-automation-agent** - Automation stub (port 11102)
7. **jarvis-multimodal-ai** - Multimodal AI stub (port 11103)

#### AI Framework Stubs (14)
8. **autogen** - AutoGen framework (port 10405)
9. **autogpt** - AutoGPT implementation (containerized)
10. **crewai** - CrewAI team framework (port 10300)
11. **aider** - AI pair programming (port 10301)
12. **gpt-engineer** - Code generation (port 11109)
13. **letta** - Memory-enabled agent (containerized)
14. **browser-use** - Web browser automation (port 10304)
15. **context-framework** - Context management (port 10404)
16. **flowise** - Low-code AI workflows (port 10401)
17. **langflow** - Visual AI flow builder (port 10400)
18. **llamaindex** - Data framework (port 10402)
19. **n8n** - Workflow automation (port 10403)
20. **dify** - LLM app platform (port 10412)
21. **privategpt** - Private document chat (port 10306)

### UNUSED Services (Remove - 15 services)  
*These services are not needed or experimental*

#### Development/Testing Tools (5)
1. **semgrep** - Static analysis (one-time run)
2. **health-monitor** - Redundant health checking (port 10210)
3. **blackbox-exporter** - External monitoring (port 10204) 
4. **ai-metrics-exporter** - Broken metrics exporter (port 11068)
5. **service-hub** - Generic service hub (port 10409)

#### Heavy ML Frameworks (4) - Profile: ml-heavy
6. **pytorch** - PyTorch notebook (port 10500)
7. **tensorflow** - TensorFlow notebook (port 10501)
8. **jax** - JAX ML framework (port 10502)
9. **fsdp** - Distributed training (profile: fsdp)

#### Specialized/Niche Tools (6)
10. **pentestgpt** - Penetration testing (containerized)
11. **documind** - Document processing (port 10308)
12. **shellgpt** - Shell command generation (port 10307)
13. **skyvern** - Web automation (containerized)  
14. **finrobot** - Financial analysis (port 10407)
15. **code-improver** - Code improvement automation (port 10408)

### FUTURE Services (Keep but comment out - 3 services)
*These have potential but need implementation*

1. **tabbyml** - Code completion (port 10303) - Profile: tabby
2. **awesome-code-ai** - Code AI tools (port 10410)
3. **opendevin** - Development assistant (port 10406)

## Additional Components

### Mesh Workers (Profile-based - 2 services)
- **mesh-worker** - NLP processing worker (profile: mesh)
- **mesh-worker-2** - Additional worker (profile: mesh)

### Database Exporters (Keep - 2 services)
- **postgres-exporter** - PostgreSQL metrics (port 10207)  
- **redis-exporter** - Redis metrics (port 10208)

### Log Collection
- **promtail** - Log shipping to Loki (no exposed port)

### Hardware Optimization
- **jarvis-hardware-resource-optimizer** - System optimization (port 11104)

## Summary by Category
- **ESSENTIAL**: 20 services (keep active)
- **STUB**: 21 services (comment out, keep definitions)  
- **UNUSED**: 15 services (remove entirely)
- **FUTURE**: 3 services (comment out, keep for later)

**Total Reduction**: 39 services can be removed/commented (66% reduction)  
**Minimal Configuration**: 20 active services (34% of original)

## Port Conflicts and Issues
- Multiple services trying to use similar ports
- Some agents have hardcoded port expectations
- Kong gateway has no routes configured
- Vector databases are isolated (not integrated with backend)

## Model Configuration Issues  
- Backend expects "gpt-oss" model
- Only "tinyllama" model is available in Ollama
- This mismatch causes degraded status in health checks

## Next Steps
1. Create docker-compose.minimal.yml with 20 essential services
2. Fix model configuration mismatch  
3. Apply PostgreSQL schema
4. Configure Kong gateway routes
5. Integrate vector databases with backend
6. Remove unused Docker directories