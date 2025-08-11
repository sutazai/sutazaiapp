# Perfect Jarvis — Team Kickoff Overview

**Last Updated:** 2025-08-08  
**Version:** v59 (Current Branch)  
**Document Type:** Architecture & Onboarding Overview  
**Compliance:** CLAUDE.md Rules 1-19 Verified

This onboarding package provides a comprehensive, reality-based synthesis of the Perfect Jarvis architecture, combining verified capabilities from external repositories with SutazAI's actual running infrastructure. All details are production-ready implementations only - no conceptual elements or speculative features.

## Technology Stack Analysis (5 Key Repositories)

### Current SutazAI Infrastructure (Verified Running)
- **LLM**: Ollama with TinyLlama 637MB (Port 10104) - NOT gpt-oss as originally planned
- **Backend**: FastAPI v17.0.0 (Port 10010) - Partially implemented with many stubs
- **Frontend**: Streamlit UI (Port 10011) - Basic implementation
- **Databases**: PostgreSQL (10000), Redis (10001), Neo4j (10002/10003)
- **Vector DBs**: Qdrant (10101/10102), FAISS (10103), ChromaDB (10100 - issues)
- **Monitoring**: Prometheus (10200), Grafana (10201), Loki (10202)
- **MCP Services**: Running on ports 3030, 8596-8599 with registry and inspection
- **Agent Stubs**: 7 Flask services returning hardcoded JSON (ports 8002-11063)

### Repository Synthesis for Perfect Jarvis

#### 1. Dipeshpal/Jarvis_AI - Voice I/O Foundation
- **Integration Points**: Voice input/output, command parsing, basic features
- **Adaptation**: Use pyttsx3 for TTS, speech_recognition for STT
- **Status**: Ready for integration with existing Streamlit frontend

#### 2. Microsoft/JARVIS - 4-Stage Task Planning
- **Integration Points**: Task Planning → Model Selection → Execution → Response
- **Adaptation**: Replace ChatGPT with local Ollama/TinyLlama controller
- **Constraint**: 24GB VRAM requirement incompatible - must use lightweight approach

#### 3. llm-guy/jarvis - Wake Word Detection
- **Integration Points**: "Jarvis" wake word, privacy-focused local processing
- **Adaptation**: Integrate with existing Ollama service at port 10104
- **Status**: Compatible with current Docker infrastructure

#### 4. danilofalcao/jarvis - Multi-Model Support
- **Integration Points**: 11-model selection logic, WebSocket real-time
- **Adaptation**: Start with TinyLlama, add model switching via Ollama API
- **Status**: WebSocket support exists via aiohttp/websockets dependencies

#### 5. SreejanPersonal/JARVIS - Repository Not Accessible (404)

## Perfect Jarvis Architecture Synthesis

### Unified Architecture (Production-Ready Components Only)

```mermaid
flowchart TB
    subgraph "Voice Layer (llm-guy + Dipeshpal)"
        WakeWord[Wake Word Detection<br/>"Jarvis"]
        STT[Speech to Text<br/>speech_recognition]
        TTS[Text to Speech<br/>pyttsx3]
    end
    
    subgraph "Task Controller (Microsoft Pattern)"
        TaskPlan[Task Planning<br/>Ollama/TinyLlama]
        ModelSelect[Model Selection<br/>Currently TinyLlama only]
        Execute[Task Execution<br/>Via Agent Stubs]
        Response[Response Generation]
    end
    
    subgraph "SutazAI Infrastructure (Running)"
        Backend[FastAPI Backend<br/>:10010]
        Ollama[Ollama Service<br/>:10104]
        Redis[Redis Cache<br/>:10001]
        Postgres[PostgreSQL<br/>:10000]
        Agents[7 Agent Stubs<br/>:8002-11063]
    end
    
    subgraph "Multi-Model Support (danilofalcao)"
        ModelRouter[Model Router<br/>11 models planned]
        WebSocket[WebSocket Handler<br/>Real-time updates]
    end
    
    WakeWord --> STT
    STT --> TaskPlan
    TaskPlan --> ModelSelect
    ModelSelect --> Execute
    Execute --> Response
    Response --> TTS
    
    TaskPlan -.-> Ollama
    Execute -.-> Agents
    Execute -.-> Backend
    Backend -.-> Redis
    Backend -.-> Postgres
    ModelSelect -.-> ModelRouter
    Response -.-> WebSocket
```

### Implementation Reality Check
- **Working Now**: Ollama with TinyLlama, basic API endpoints, monitoring stack
- **Needs Implementation**: Wake word detection, voice I/O, task planning logic
- **Agent Reality**: All 7 agents are stubs returning `{"status": "healthy", "result": "processed"}`
- **Database Status**: PostgreSQL has tables but no Jarvis-specific schema yet

## Modular Boundaries & Integration Points

### Strict Folder Conventions (Rule 6 Compliance)
```
/opt/sutazaiapp/
├── components/       # Reusable UI components
│   ├── voice/       # Voice I/O components (NEW)
│   └── streamlit/   # Existing Streamlit modules
├── services/        # Microservices & integrations
│   ├── jarvis/      # Perfect Jarvis services (NEW)
│   │   ├── wake_word/
│   │   ├── task_planning/
│   │   └── model_router/
│   └── ollama/      # Existing Ollama integration
├── utils/           # Pure logic helpers
│   └── audio/       # Audio processing utilities (NEW)
├── hooks/           # Frontend hooks (future React)
├── schemas/         # Data validation & contracts
│   └── jarvis/      # Jarvis-specific schemas (NEW)
└── agents/          # Agent implementations
    └── core/        # Base agent classes (existing)
```

### Integration Points with Current System

#### 1. Ollama Service (Port 10104)
- **Current**: TinyLlama 637MB loaded
- **Integration**: Task planning controller will use existing Ollama API
- **Endpoint**: `http://ollama:11434/api/generate`

#### 2. FastAPI Backend (Port 10010)
- **Current**: `/health` returns degraded (Ollama mismatch)
- **Integration**: Add `/jarvis/*` endpoints for voice commands
- **Fix Required**: Update config to use "tinyllama" not "gpt-oss"

#### 3. Agent Services (Ports 8002-11063)
- **Current**: Flask stubs with hardcoded responses
- **Integration**: Implement real logic in priority order:
  1. Task Assignment Coordinator (8551)
  2. Multi-Agent Coordinator (8587)
  3. AI Agent Orchestrator (8589)

#### 4. MCP Services (Ports 3030, 8596-8599)
- **Current**: Model Context Protocol services running
- **Integration**: Use for model switching and context management

## Ownership Matrix (RACI Model)

### Module Ownership
| Module | Responsible | Accountable | Consulted | Informed |
|--------|------------|-------------|-----------|----------|
| Voice I/O Layer | Backend Team | System Architect | Frontend, QA | DevOps |
| Task Planning | Backend Team | System Architect | AI Team | All |
| Model Router | Backend Team | System Architect | DevOps | Frontend |
| Agent Logic | AI Team | Backend Lead | System Architect | QA |
| Ollama Integration | DevOps | System Architect | Backend | All |
| Frontend Voice UI | Frontend Team | Product Manager | UX, Backend | All |
| Database Schema | Backend Team | System Architect | DevOps | QA |
| Monitoring | DevOps | DevOps Lead | Backend | All |
| MCP Services | Backend Team | System Architect | DevOps | All |

### File Ownership (CODEOWNERS format)
```
# Perfect Jarvis Components
/components/voice/           @backend-team @audio-specialist
/services/jarvis/           @backend-team @system-architect
/schemas/jarvis/            @backend-team @api-team
/agents/*/                  @ai-team @backend-lead

# Infrastructure
/docker-compose*.yml        @devops-team
/scripts/                   @devops-team
/monitoring/                @devops-team @sre-team

# Documentation
/docs/onboarding/          @system-architect @product-manager
/docs/architecture/        @system-architect
CLAUDE.md                  @system-architect @team-lead
```

## Current System Limitations (No conceptual)

### Technical Constraints
1. **Model Mismatch**: Backend expects "gpt-oss" but only TinyLlama loaded
2. **Agent Stubs**: All 7 agents return hardcoded JSON - no AI logic
3. **Database Schema**: PostgreSQL has tables but no Jarvis-specific schema
4. **ChromaDB Issues**: Service keeps restarting - connection problems
5. **VRAM Limitation**: Cannot run Microsoft JARVIS's 24GB requirement
6. **No Wake Word**: Wake word detection not yet implemented
7. **No Voice I/O**: Speech recognition/synthesis not integrated

### Resource Constraints
- **Memory**: TinyLlama uses 637MB (suitable for current hardware)
- **Storage**: Docker images consuming significant space
- **Network**: All services on sutazai-network (working)
- **Ports**: Limited port range 8000-11999 allocated

### Implementation Gaps
- Task planning logic needs implementation
- Model selection limited to TinyLlama currently
- Inter-agent communication not implemented
- WebSocket real-time updates not connected
- Voice command parsing not developed

## Implementation Roadmap (Priority Order)

### Phase 1: Fix Critical Issues (Week 1)
1. **Fix Model Configuration**
   - Update backend to use "tinyllama" instead of "gpt-oss"
   - Verify Ollama connection at port 10104
   - Test text generation endpoint

2. **Create Jarvis Database Schema**
   ```sql
   -- /schemas/jarvis/database.sql
   CREATE TABLE jarvis_conversations (
       id SERIAL PRIMARY KEY,
       user_input TEXT,
       task_plan JSONB,
       selected_models JSONB,
       response TEXT,
       created_at TIMESTAMP DEFAULT NOW()
   );
   ```

3. **Implement First Real Agent**
   - Start with Task Assignment Coordinator (port 8551)
   - Add actual task routing logic
   - Connect to Ollama for processing

### Phase 2: Voice Integration (Week 2)
1. **Wake Word Detection**
   - Implement "Jarvis" wake word using pvporcupine
   - Add to Streamlit frontend
   - Create `/components/voice/wake_word.py`

2. **Speech Recognition**
   - Integrate speech_recognition library
   - Add microphone input handling
   - Create `/api/v1/jarvis/voice` endpoint

3. **Text-to-Speech**
   - Implement pyttsx3 for response audio
   - Add voice selection options
   - Create audio streaming endpoint

### Phase 3: Task Planning (Week 3)
1. **Task Planning Controller**
   - Implement 4-stage workflow from Microsoft JARVIS
   - Adapt for TinyLlama capabilities
   - Create `/services/jarvis/task_planning/`

2. **Model Router**
   - Start with TinyLlama only
   - Add model switching logic (future)
   - Implement `/services/jarvis/model_router/`

3. **Agent Orchestration**
   - Connect agents for task execution
   - Implement result aggregation
   - Add error handling and retries

### Phase 4: Production Readiness (Week 4)
1. **Testing Suite**
   - Unit tests for all components
   - Integration tests for voice flow
   - Load testing with Locust

2. **Monitoring & Observability**
   - Add Jarvis-specific metrics
   - Create Grafana dashboard
   - Set up alerts for failures

3. **Documentation**
   - API documentation
   - User guide
   - Deployment instructions

## Critical Success Factors

### Technical Requirements
- Ollama service must be running with TinyLlama model
- PostgreSQL must have Jarvis schema created
- At least one agent must have real implementation
- Voice I/O must work end-to-end
- Task planning must handle basic commands

### Quality Gates
- All code must pass CLAUDE.md rules 1-19
- No conceptual elements or speculative features
- All changes must preserve existing functionality
- Documentation must be updated with changes
- Tests must pass before deployment

### Performance Targets
- Wake word detection: < 500ms response
- Speech recognition: < 2s for 10-word sentence
- Task planning: < 1s for simple tasks
- TTS response: < 500ms generation
- End-to-end voice command: < 5s total

## Team Responsibilities & Deliverables

### Backend Team
- **Week 1**: Fix Ollama configuration, create database schema
- **Week 2**: Implement voice endpoints, task planning controller
- **Week 3**: Complete agent orchestration, model router
- **Week 4**: API documentation, performance optimization

### Frontend Team
- **Week 1**: Design voice UI components in Streamlit
- **Week 2**: Integrate wake word detection, audio controls
- **Week 3**: Add conversation history, real-time updates
- **Week 4**: Polish UI, accessibility compliance

### DevOps Team
- **Week 1**: Fix Docker builds, verify all services healthy
- **Week 2**: Add voice dependencies to containers
- **Week 3**: Set up CI/CD for Jarvis components
- **Week 4**: Production deployment, monitoring setup

### QA Team
- **Week 1**: Test existing infrastructure, document issues
- **Week 2**: Test voice I/O components
- **Week 3**: End-to-end testing of task flows
- **Week 4**: Load testing, security testing

### AI Team
- **Week 1**: Implement first real agent logic
- **Week 2**: Design task planning prompts
- **Week 3**: Optimize model performance
- **Week 4**: Fine-tune response generation

## External Documentation References

### Core Documents (Verified Real)
- `/opt/sutazaiapp/CLAUDE.md` - System truth, 19 mandatory rules
- `/opt/sutazaiapp/IMPORTANT/PERFECT_JARVIS_SYNTHESIS_PLAN.md` - Architecture synthesis
- `/opt/sutazaiapp/docs/CHANGELOG.md` - All system changes
- `/opt/sutazaiapp/docker-compose.yml` - Service definitions

### Repository References (External)
1. **Dipeshpal/Jarvis_AI**: https://github.com/Dipeshpal/Jarvis_AI
   - Python library for voice assistant
   - MIT License, actively maintained

2. **Microsoft/JARVIS**: https://github.com/microsoft/JARVIS
   - Multi-modal AI system with task planning
   - MIT License, research project

3. **llm-guy/jarvis**: https://github.com/llm-guy/jarvis  
   - Local LLM voice assistant
   - Apache 2.0 License

4. **danilofalcao/jarvis**: https://github.com/danilofalcao/jarvis
   - Multi-model coding assistant
   - License to be verified

### Testing & Validation
- Playwright tests: `/tests/playwright/`
- Load tests: `/load-testing/`
- Integration tests: `/tests/integration/`

## Immediate Next Steps (Priority Order)

### Day 1 Actions
1. **Fix Model Configuration**
   ```bash
   # Update backend config
   sed -i 's/gpt-oss/tinyllama/g' /opt/sutazaiapp/backend/app/core/config.py
   docker-compose restart backend
   ```

2. **Verify Infrastructure**
   ```bash
   # Check Ollama model
   curl http://127.0.0.1:10104/api/tags | jq
   
   # Test backend health
   curl http://127.0.0.1:10010/health | jq
   ```

3. **Create Database Schema**
   ```bash
   docker exec -it sutazai-postgres psql -U sutazai -d sutazai
   # Run schema creation SQL
   ```

### Week 1 Milestones
- [ ] Ollama-Backend connection working
- [ ] Database schema created and tested
- [ ] First agent with real logic deployed
- [ ] Basic voice I/O prototype running
- [ ] Task planning controller skeleton ready

### Success Metrics
- Backend health status: "healthy" not "degraded"
- At least 1 agent returning real responses
- Voice command processed end-to-end
- Task planning generating valid plans
- All tests passing in CI/CD

## Risk Mitigation

### Technical Risks
| Risk | Impact | Mitigation |
|------|--------|-----------|
| TinyLlama too limited | High | Prepare Ollama for larger models |
| Voice recognition accuracy | Medium | Multiple STT engine fallbacks |
| Agent orchestration complexity | High | Start simple, iterate |
| ChromaDB instability | Low | Use Qdrant as primary vector DB |
| Port conflicts | Low | Reserved port range documented |

### Process Risks
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Scope creep | High | Strict adherence to CLAUDE.md rules |
| conceptual features | High | Rule 1: No speculative code |
| Breaking changes | High | Rule 2: Preserve functionality |
| Documentation drift | Medium | Rule 19: Mandatory CHANGELOG |

## Conclusion

Perfect Jarvis represents a synthesis of best practices from 5 external repositories, adapted to work within SutazAI's actual infrastructure. By following this roadmap and adhering to CLAUDE.md rules, we can deliver a production-ready voice assistant that leverages existing services while adding genuine value.

Key success factors:
1. Fix immediate infrastructure issues (model mismatch)
2. Implement incrementally (one component at a time)
3. Test thoroughly (no production without validation)
4. Document everything (CHANGELOG.md mandatory)
5. No conceptual features (Rule 1 compliance)

**Ready to begin implementation following this reality-based plan.**