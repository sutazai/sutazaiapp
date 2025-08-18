# ARCH-001 Prioritized Cleanup Action Plan
**Agent ID**: ARCH-001 (System Architect)  
**Plan Date**: 2025-08-08  
**Target Completion**: 4 weeks  
**System Version**: SutazAI v67  

## Priority Matrix

| Priority | Category | Impact | Effort | Timeline |
|----------|----------|--------|--------|----------|
| P0 | Critical Security | Prevent breach | Low | Today |
| P1 | Core Functionality | Enable basic ops | Medium | Week 1 |
| P2 | Integration | Connect components | Medium | Week 2 |
| P3 | Optimization | Improve performance | High | Week 3 |
| P4 | Cleanup | Reduce complexity | Low | Week 4 |

## Week 1: Critical Foundation

### Day 1: Security & Configuration
**Owner**: Backend Architect  
**Dependencies**: None  

- [ ] **P0-001**: Move all secrets to environment variables
  ```bash
  # Create .env file
  DATABASE_URL=postgresql://sutazai:${DB_PASSWORD}@postgres:5432/sutazai
  JWT_SECRET_KEY=${JWT_SECRET}
  REDIS_URL=redis://redis:6379
  OLLAMA_BASE_URL=http://ollama:11434
  DEFAULT_MODEL=tinyllama
  ```

- [ ] **P0-002**: Implement JWT authentication
  - Location: `/backend/app/auth/`
  - Add middleware to all endpoints
  - Create login/logout endpoints
  - Test with Postman collection

- [ ] **P1-001**: Fix model configuration
  - Update `/backend/app/core/config.py`
  - Change DEFAULT_MODEL to "tinyllama"
  - Add MODEL_OVERRIDE env variable
  - Test inference endpoints

### Day 2: Database Foundation
**Owner**: Backend Architect  
**Dependencies**: P0-001  

- [ ] **P1-002**: Create UUID-based schema
  ```sql
  -- /backend/migrations/001_initial_schema.sql
  CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
  
  CREATE TABLE users (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      email VARCHAR(255) UNIQUE NOT NULL,
      password_hash VARCHAR(255) NOT NULL,
      role VARCHAR(50) DEFAULT 'user',
      created_at TIMESTAMP DEFAULT NOW()
  );
  
  CREATE TABLE agents (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      name VARCHAR(100) UNIQUE NOT NULL,
      type VARCHAR(50) NOT NULL,
      config JSONB DEFAULT '{}',
      status VARCHAR(50) DEFAULT 'inactive'
  );
  
  CREATE TABLE tasks (
      id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
      agent_id UUID REFERENCES agents(id),
      user_id UUID REFERENCES users(id),
      payload JSONB NOT NULL,
      result JSONB,
      status VARCHAR(50) DEFAULT 'pending',
      created_at TIMESTAMP DEFAULT NOW()
  );
  ```

- [ ] **P1-003**: Apply migrations
  - Run migration script
  - Verify tables created
  - Add indexes for foreign keys
  - Test CRUD operations

### Day 3: Service Mesh Configuration
**Owner**: DevOps Lead  
**Dependencies**: P1-001, P1-002  

- [ ] **P1-004**: Configure Kong Gateway
  ```yaml
  # /config/kong/kong.yml
  services:
    - name: backend-api
      url: http://backend:8000
      routes:
        - name: api-v1
          paths: ["/api/v1"]
          methods: ["GET", "POST", "PUT", "DELETE"]
      plugins:
        - name: jwt
        - name: cors
        - name: rate-limiting
          config:
            minute: 100
  ```

- [ ] **P1-005**: Configure RabbitMQ
  - Create exchanges for agent communication
  - Define routing keys
  - Set up dead letter queues
  - Test message flow

## Week 2: Core Implementation

### Day 4-5: Implement Real Agent
**Owner**: AI Engineer  
**Dependencies**: P1-004, P1-005  

- [ ] **P2-001**: Implement AI Orchestrator Agent
  - Location: `/agents/ai_agent_orchestrator/app.py`
  - Connect to RabbitMQ
  - Integrate with Ollama
  - Add task processing logic
  - Implement result callbacks

- [ ] **P2-002**: Connect agents via message bus
  - Update all agent base classes
  - Implement message handlers
  - Add retry logic
  - Test inter-agent communication

### Day 6-7: Vector Database Integration
**Owner**: Backend Architect  
**Dependencies**: P1-002  

- [ ] **P2-003**: Wire ChromaDB to backend
  - Create `/backend/app/services/vector_service.py`
  - Add document upload endpoint
  - Implement embedding generation
  - Create similarity search API
  - Test with sample documents

- [ ] **P2-004**: Create RAG pipeline
  - Document ingestion flow
  - Query processing
  - Context injection
  - Response generation

## Week 3: Stabilization

### Day 8-10: Monitoring & Observability
**Owner**: DevOps Lead  
**Dependencies**: All P1, P2  

- [ ] **P3-001**: Import Grafana dashboards
  - System overview dashboard
  - Agent performance metrics
  - API latency tracking
  - Error rate monitoring

- [ ] **P3-002**: Configure alerts
  - High error rate
  - Low disk space
  - Service down
  - High latency

- [ ] **P3-003**: Implement structured logging
  - JSON log format
  - Correlation IDs
  - Log aggregation
  - Search interface

### Day 11-12: Error Handling
**Owner**: Backend Architect  
**Dependencies**: P3-001  

- [ ] **P3-004**: Add circuit breakers
  - For external service calls
  - For agent communication
  - For database queries
  - Fallback mechanisms

- [ ] **P3-005**: Implement retry strategies
  - Exponential backoff
  - Dead letter queues
  - Manual retry interface
  - Failure notifications

## Week 4: Optimization & Cleanup

### Day 13-14: Container Cleanup
**Owner**: DevOps Lead  
**Dependencies**: All P3  

- [ ] **P4-001**: Remove unused services
  ```yaml
  # Services to remove from docker-compose.yml:
  - agentgpt
  - autogpt
  - agentzero
  - tensorflow
  - pytorch
  - jax
  - dify
  - flowise
  - langflow
  # ... (31 total)
  ```

- [ ] **P4-002**: Optimize resource allocation
  - Right-size container limits
  - Configure health checks
  - Add restart policies
  - Implement graceful shutdown

### Day 15: Documentation Update
**Owner**: Documentation Lead  
**Dependencies**: All P4  

- [ ] **P4-003**: Update all documentation
  - Remove conceptual features
  - Document actual capabilities
  - Create API documentation
  - Write deployment guide
  - Update README

- [ ] **P4-004**: Create runbooks
  - Deployment procedures
  - Troubleshooting guide
  - Monitoring playbook
  - Incident response

## Success Metrics

### Week 1 Targets
- ✅ All secrets in environment variables
- ✅ JWT authentication working
- ✅ Database schema applied
- ✅ Kong routes configured
- ✅ Model configuration fixed

### Week 2 Targets
- ✅ One agent fully functional
- ✅ Agent communication working
- ✅ Vector search operational
- ✅ RAG pipeline tested
- ✅ 50% reduction in stubs

### Week 3 Targets
- ✅ Grafana dashboards live
- ✅ Alerts configured
- ✅ Error handling implemented
- ✅ 95% uptime achieved
- ✅ <2s API response time

### Week 4 Targets
- ✅ 20 containers (from 59)
- ✅ 6GB RAM usage (from 12GB)
- ✅ Documentation complete
- ✅ 25% test coverage
- ✅ Production-ready MVP

## Risk Mitigation

### Rollback Plans
1. **Database changes**: Keep backup before migration
2. **Service changes**: Use feature flags
3. **Container removal**: Keep compose backup
4. **Code changes**: Git branching strategy

### Testing Strategy
1. **Unit tests**: For new code (target 50%)
2. **Integration tests**: For agent communication
3. **E2E tests**: For critical user flows
4. **Load tests**: Before production

### Communication Plan
1. Daily standup for progress
2. Weekly architecture review
3. Blocker escalation process
4. Documentation updates

## Resource Requirements

### Team Allocation
- 1 System Architect (20%)
- 1 Backend Engineer (100%)
- 1 AI Engineer (100%)
- 1 DevOps Engineer (50%)
- 1 QA Engineer (50%)

### Infrastructure
- Development environment
- Staging environment
- CI/CD pipeline
- Monitoring tools

## Deliverables

### Week 1
- [ ] Security audit report
- [ ] Database migration scripts
- [ ] Kong configuration
- [ ] Updated .env.example

### Week 2
- [ ] Working agent demo
- [ ] Vector search API docs
- [ ] Integration test suite
- [ ] Performance baseline

### Week 3
- [ ] Monitoring dashboards
- [ ] Alert runbook
- [ ] Error handling guide
- [ ] Load test results

### Week 4
- [ ] Optimized docker-compose
- [ ] Complete documentation
- [ ] Deployment guide
- [ ] MVP release notes

## Next Steps

1. **Immediate** (Today):
   - Review and approve plan
   - Assign team members
   - Set up tracking dashboard
   - Create feature branches

2. **Tomorrow**:
   - Begin P0 security fixes
   - Start database migration
   - Update model configuration
   - Create test environment

3. **This Week**:
   - Complete Week 1 targets
   - Daily progress updates
   - Address blockers immediately
   - Prepare Week 2 resources

---
*Generated by ARCH-001 System Architect Agent*  
*Based on comprehensive system analysis*  
*Aligned with CLAUDE.md reality check*