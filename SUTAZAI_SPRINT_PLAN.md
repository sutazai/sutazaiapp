# SUTAZAI System - Sprint Plan

## Sprint 1: Foundation Fix
**Duration**: 2 weeks (10 working days)  
**Goal**: Fix critical blockers and establish core agent functionality  
**Team Capacity**: 40 story points (assuming 4 developers × 10 days × 1 point/day)

---

## Sprint Backlog

### Week 1: Critical Blockers & Infrastructure (20 points)

#### Day 1-2: Model & Database Foundation
- **Story 1.1**: Fix TinyLlama Configuration Mismatch [P0, 2 points]
  - Owner: Backend Developer
  - Update all configuration files
  - Test Ollama connectivity
  - Verify health endpoints

- **Story 4.1**: Create PostgreSQL Schema and Migrations [P0, 3 points]
  - Owner: Database Developer
  - Design schema for users, agents, tasks
  - Create Alembic migrations
  - Deploy and verify

#### Day 3-4: Messaging Infrastructure
- **Story 3.1**: Create Message Schema Definitions [P0, 3 points]
  - Owner: Backend Developer
  - Define Pydantic schemas
  - Create validation utilities
  - Document message formats

- **Story 3.2**: Implement RabbitMQ Connection Pool [P0, 3 points]
  - Owner: Backend Developer
  - Create connection manager
  - Implement retry logic
  - Add health monitoring

#### Day 5: Agent Foundation
- **Story 2.1**: Implement Task Assignment Coordinator Logic [P0, 5 points]
  - Owner: Senior Developer (continues into Day 6)
  - Core routing logic
  - Queue consumption
  - Assignment strategies

### Week 1 Review & Planning (End of Day 5)
- Demo working model connection
- Show database schema
- Review message flow
- Plan Week 2 details

### Week 2: Core Agent Implementation (20 points)

#### Day 6-7: Complete Coordinator & Start Orchestrator
- **Story 2.1** (continued): Complete Task Assignment Coordinator
  - Integration tests
  - Documentation
  
- **Story 2.2**: Implement AI Agent Orchestrator Core Logic [P0, 8 points]
  - Owner: Senior Developer (continues to Day 9)
  - Task decomposition logic
  - Agent selection algorithm

#### Day 8-9: Service Discovery & Monitoring
- **Story 5.2**: Fix Consul Health Checks [P1, 2 points]
  - Owner: DevOps Engineer
  - Update registration scripts
  - Fix network issues
  - Verify all services green

- **Story 3.3**: Implement Agent Heartbeat System [P1, 3 points]
  - Owner: Backend Developer
  - Heartbeat sender in base agent
  - Stale detection logic
  - AlertManager integration

#### Day 10: Integration & Documentation
- **Story 7.1**: Create API Documentation [P1, 3 points]
  - Owner: Technical Writer + Developer
  - OpenAPI spec
  - Postman collection
  - Usage examples

- **Sprint Retrospective** (PM)
  - What worked well
  - What needs improvement
  - Velocity calculation

---

## Definition of Done

Each story is considered DONE when:
1. ✅ Code is written and reviewed
2. ✅ Unit tests pass (where applicable)
3. ✅ Integration tests pass
4. ✅ Documentation updated
5. ✅ Deployed to development environment
6. ✅ Acceptance criteria verified
7. ✅ No critical bugs remaining
8. ✅ CHANGELOG.md updated (Rule 19)

---

## Risk Management

### Identified Risks & Mitigations

1. **Risk**: RabbitMQ integration complexity
   - **Impact**: High - blocks agent communication
   - **Mitigation**: Pair programming, use existing coordinator code as reference

2. **Risk**: Network segmentation between Docker networks
   - **Impact**: Medium - Consul health checks fail
   - **Mitigation**: Document workaround, plan network consolidation for Sprint 2

3. **Risk**: Ollama model performance
   - **Impact**: Low - slow responses
   - **Mitigation**: Use TinyLlama for now, plan model optimization for later

4. **Risk**: Team unfamiliar with agent architecture
   - **Impact**: Medium - slower development
   - **Mitigation**: Architecture review session Day 1, pair programming

---

## Daily Standup Schedule

**Time**: 9:30 AM  
**Duration**: 15 minutes  
**Format**: 
- What I completed yesterday
- What I'm working on today
- Any blockers
- Need for pair programming

---

## Success Metrics

### Sprint 1 Goals
- ✅ 100% of P0 stories completed (6 stories, 21 points)
- ✅ At least 3 P1 stories completed (8 points)
- ✅ Zero critical bugs in production
- ✅ All agents have working health endpoints
- ✅ Task assignment working end-to-end
- ✅ Team velocity established for future planning

### Key Performance Indicators
- **Velocity**: Target 40 points, Minimum 29 points
- **Bug Rate**: < 1 critical bug
- **Test Coverage**: > 60% for new code
- **Documentation**: 100% of public APIs documented

---

## Sprint 2 Preview (Tentative)

Based on Sprint 1 velocity, Sprint 2 will focus on:

### High Priority Candidates (P1)
- Story 2.3: Hardware Resource Optimizer [5 points]
- Story 2.4: Multi-Agent Coordinator [5 points]
- Story 2.5: Resource Arbitration Agent [5 points]
- Story 4.2: Integrate Qdrant Vector Database [5 points]
- Story 4.3: Implement Redis Caching [3 points]
- Story 5.1: Configure Kong API Routes [3 points]
- Story 7.2: Integration Test Suite [5 points]

### Technical Debt (P2) - If capacity allows
- Story 6.1: Consolidate Requirements Files [5 points]
- Story 6.2: Remove Non-Running Services [2 points]

---

## Team Assignments

### Proposed Team Structure
- **Senior Developer**: Agent implementation (Stories 2.1, 2.2)
- **Backend Developer**: Messaging & schemas (Stories 1.1, 3.1, 3.2, 3.3)
- **Database Developer**: Database setup (Story 4.1)
- **DevOps Engineer**: Infrastructure (Story 5.2)
- **Technical Writer**: Documentation (Story 7.1)

### Pair Programming Sessions
- Day 3: Message schemas (Backend + Senior)
- Day 5-6: Task Coordinator (Senior + Backend)
- Day 8: Heartbeat system (Backend + DevOps)

---

## Communication Plan

### Meetings
- **Sprint Planning**: Day 0, 2 hours
- **Daily Standup**: Days 1-10, 15 minutes
- **Mid-Sprint Review**: Day 5, 1 hour
- **Sprint Review**: Day 10 AM, 1 hour
- **Sprint Retrospective**: Day 10 PM, 1 hour

### Documentation
- Daily updates to team wiki
- Blockers reported immediately in Slack
- Technical decisions recorded in ADRs
- Progress tracked in project board

---

## Acceptance Test Scenarios

### Scenario 1: End-to-End Task Assignment
```
GIVEN the system is running with all P0 stories complete
WHEN a task is submitted via API
THEN it should be:
  1. Received by the backend
  2. Published to RabbitMQ
  3. Consumed by Task Coordinator
  4. Assigned to appropriate agent
  5. Confirmation sent back
  6. Status queryable via API
```

### Scenario 2: Model Integration
```
GIVEN TinyLlama is properly configured
WHEN a text generation request is made
THEN Ollama should:
  1. Receive the request
  2. Generate response using TinyLlama
  3. Return response within 5 seconds
  4. Health check shows "connected"
```

### Scenario 3: Agent Health Monitoring
```
GIVEN all agents are running
WHEN checking Consul dashboard
THEN all services should:
  1. Show green health status
  2. Have accurate IP addresses
  3. Respond to health checks
  4. Send heartbeats every 30s
```

---

## Notes for Sprint Planning Meeting

### Discussion Items
1. Confirm team capacity (40 points realistic?)
2. Review story estimates with team
3. Identify technical dependencies not captured
4. Assign story owners
5. Schedule pair programming sessions
6. Review Definition of Done
7. Set up development environment for all team members

### Pre-Sprint Checklist
- [ ] Development environments ready
- [ ] Access to all required services
- [ ] Project board configured
- [ ] CI/CD pipelines working
- [ ] Slack channels created
- [ ] Documentation wiki set up
- [ ] Test data prepared

---

## Appendix: Technical Notes

### Critical File Locations
- Backend config: `/opt/sutazaiapp/backend/app/core/config.py`
- Agent configs: `/opt/sutazaiapp/config/agents.yaml`
- Message schemas: `/opt/sutazaiapp/schemas/`
- Docker compose: `/opt/sutazaiapp/docker-compose.yml`
- Consul config: `/opt/sutazaiapp/config/consul-services-config.json`

### Quick Commands
```bash
# Start system
docker-compose up -d

# Check agent health
curl http://localhost:8589/health

# View Consul dashboard
open http://localhost:10006/ui

# Check RabbitMQ queues
open http://localhost:10008

# View logs
docker-compose logs -f backend
```

### Known Issues Going Into Sprint
1. ChromaDB connection issues (not prioritized)
2. Docker network segmentation (workaround documented)
3. Some monitoring containers unhealthy (not blocking)
4. Frontend takes time to initialize (expected)