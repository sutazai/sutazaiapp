# SUTAZAI FEATURES AND USER STORIES BIBLE

**Version:** 1.0.0  
**Created:** August 7, 2025  
**Status:** AUTHORITATIVE - This is the definitive feature and user story guide  
**Classification:** ENGINEERING SPECIFICATION  

---

## EXECUTIVE SUMMARY

This document defines the complete feature set and user stories for the SutazAI system transformation from a proof-of-concept with stub agents to a fully functional AI orchestration platform. Based on verified system state as of August 6, 2025, this document provides a realistic, implementable roadmap grounded in existing infrastructure.

### Current System Reality (Verified)
- **28 containers operational** with core infrastructure healthy
- **Backend API v17.0.0** running with 70+ endpoints (many stubs)
- **PostgreSQL with 14 tables** created and ready for data
- **TinyLlama model (637MB)** loaded and functional in Ollama
- **7 agent services** running but returning fake JSON responses
- **Full monitoring stack** operational (Prometheus, Grafana, Loki)
- **Service mesh components** running but NOT configured

### Document Purpose
1. Define clear, measurable feature specifications
2. Provide detailed user stories with acceptance criteria
3. Establish implementation priorities based on value delivery
4. Create testable success metrics for each phase
5. Align all development efforts with realistic capabilities

---

## STRATEGIC ROADMAP PHASES

### Phase 0: Critical Fixes (72 Hours)
**Objective:** Stabilize system and fix breaking issues  
**Timeline:** 3 days  
**Risk Level:** Critical - System unstable without these fixes  

### Phase 1: MVP - First Working Agent (2 Weeks)
**Objective:** Transform one stub agent into fully functional service  
**Timeline:** 14 days  
**Success Metric:** One agent processing real requests with measurable results  

### Phase 2: Agent Expansion (4 Weeks)
**Objective:** All 7 agents operational with basic orchestration  
**Timeline:** 28 days  
**Success Metric:** Multi-agent task execution with coordination  

### Phase 3: System Integration (6 Weeks)
**Objective:** Full service mesh configuration and vector database integration  
**Timeline:** 42 days  
**Success Metric:** End-to-end workflow automation with knowledge persistence  

### Phase 4: Production Readiness (8 Weeks)
**Objective:** Authentication, scaling, and operational excellence  
**Timeline:** 56 days  
**Success Metric:** System handling 100+ concurrent users with 99.9% uptime  

---

## CROSS-FUNCTIONAL FEATURE PILLARS

### 1. Agent Implementation Pillar
Transform stub agents into intelligent, functional services capable of real processing.

**Core Features:**
- Agent lifecycle management
- Task processing pipelines
- Inter-agent communication
- Result validation and error handling
- Performance monitoring per agent

### 2. Service Integration Pillar
Connect and configure existing but disconnected services.

**Core Features:**
- Kong API Gateway routing
- RabbitMQ message queue implementation
- Consul service discovery activation
- Vector database integration
- Cache optimization with Redis

### 3. Data Pipeline Pillar
Leverage existing databases for persistent, intelligent operations.

**Core Features:**
- PostgreSQL data models and migrations
- Neo4j relationship mapping
- Vector embedding storage
- Time-series metrics collection
- Event sourcing implementation

### 4. Monitoring & Observability Pillar
Maximize existing monitoring stack for operational excellence.

**Core Features:**
- Custom Grafana dashboards
- Alert rule configuration
- Log aggregation pipelines
- Performance baselines
- Distributed tracing setup

### 5. API Development Pillar
Build on FastAPI backend to deliver robust endpoints.

**Core Features:**
- RESTful API completion
- WebSocket real-time updates
- GraphQL interface (optional)
- API versioning strategy
- Rate limiting and throttling

### 6. Security & Authentication Pillar
Implement enterprise-grade security from ground up.

**Core Features:**
- JWT authentication
- Role-based access control (RBAC)
- API key management
- Audit logging
- Secrets management

---

## PHASE 0: CRITICAL FIXES (72 HOURS)

### Epic 0.1: Model Configuration Alignment

#### STORY ID: P0-MODEL-001
**Title:** Fix TinyLlama vs GPT-OSS Model Mismatch  
**As a:** System Administrator  
**I want:** The backend to use the correct LLM model  
**So that:** All LLM operations work without configuration errors  

**Acceptance Criteria:**
- [ ] Backend configuration updated to use "tinyllama" model
- [ ] All agent configurations aligned with TinyLlama
- [ ] Ollama integration tests passing with correct model
- [ ] No model mismatch errors in logs

**Technical Implementation:**
- File: `/opt/sutazaiapp/backend/app/core/config.py`
- Changes: Update `DEFAULT_MODEL = "tinyllama"`
- Tests: `curl http://127.0.0.1:10104/api/generate -d '{"model": "tinyllama", "prompt": "test"}'`

**Dependencies:** None  
**Effort:** 4 hours  
**Priority:** P0 (Blocker)  

---

#### STORY ID: P0-MODEL-002
**Title:** Validate Ollama Connection Health  
**As a:** Backend Service  
**I want:** Reliable connection to Ollama service  
**So that:** LLM operations don't fail intermittently  

**Acceptance Criteria:**
- [ ] Health check endpoint returns "healthy" for Ollama
- [ ] Connection retry logic implemented
- [ ] Circuit breaker pattern for Ollama calls
- [ ] Proper error messages when Ollama unavailable

**Technical Implementation:**
- File: `/opt/sutazaiapp/backend/app/core/ollama_client.py`
- Changes: Add connection pooling, retry logic, health checks
- Tests: Disconnect Ollama, verify graceful degradation

**Dependencies:** P0-MODEL-001  
**Effort:** 6 hours  
**Priority:** P0  

---

### Epic 0.2: Database Data Initialization

#### STORY ID: P0-DATA-001
**Title:** Initialize PostgreSQL with Seed Data  
**As a:** Developer  
**I want:** Basic data in all 14 tables  
**So that:** I can test application functionality  

**Acceptance Criteria:**
- [ ] All 14 tables contain sample data
- [ ] User accounts created for testing
- [ ] Agent records populated
- [ ] Task templates available
- [ ] Migration scripts idempotent

**Technical Implementation:**
- File: `/opt/sutazaiapp/backend/app/db/seeds.py`
- Changes: Create comprehensive seed data script
- Tests: `docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT COUNT(*) FROM users;"`

**Dependencies:** None  
**Effort:** 8 hours  
**Priority:** P0  

---

#### STORY ID: P0-DATA-002
**Title:** Implement Database Migration System  
**As a:** DevOps Engineer  
**I want:** Automated database migrations  
**So that:** Schema changes are version controlled  

**Acceptance Criteria:**
- [ ] Alembic or similar migration tool configured
- [ ] Initial migration captures current schema
- [ ] Rollback capability tested
- [ ] Migration runs on container startup

**Technical Implementation:**
- File: `/opt/sutazaiapp/backend/alembic/`
- Changes: Setup Alembic, create initial migration
- Tests: Run migration up/down cycle

**Dependencies:** P0-DATA-001  
**Effort:** 6 hours  
**Priority:** P0  

---

### Epic 0.3: Container Health Stabilization

#### STORY ID: P0-HEALTH-001
**Title:** Fix ChromaDB Connection Issues  
**As a:** Backend Service  
**I want:** Stable ChromaDB connection  
**So that:** Vector operations work reliably  

**Acceptance Criteria:**
- [ ] ChromaDB container stays healthy
- [ ] Connection from backend successful
- [ ] Basic vector storage/retrieval working
- [ ] Proper error handling for ChromaDB failures

**Technical Implementation:**
- File: `/opt/sutazaiapp/docker-compose.yml`
- Changes: Update ChromaDB configuration, add health checks
- Tests: Store and retrieve test vectors

**Dependencies:** None  
**Effort:** 4 hours  
**Priority:** P0  

---

#### STORY ID: P0-HEALTH-002
**Title:** Implement Comprehensive Health Monitoring  
**As a:** Site Reliability Engineer  
**I want:** All services reporting detailed health status  
**So that:** I can quickly identify and fix issues  

**Acceptance Criteria:**
- [ ] All 28 containers have health check endpoints
- [ ] Unified health dashboard in Grafana
- [ ] Alert rules for unhealthy services
- [ ] Automatic restart for failed containers

**Technical Implementation:**
- File: `/opt/sutazaiapp/monitoring/health_checks.py`
- Changes: Create unified health monitoring system
- Tests: Kill a container, verify detection and recovery

**Dependencies:** None  
**Effort:** 12 hours  
**Priority:** P0  

---

### Epic 0.4: Requirements Consolidation

#### STORY ID: P0-REQ-001
**Title:** Consolidate 75+ Requirements Files  
**As a:** Development Team  
**I want:** Three clean requirements files  
**So that:** Dependency management is maintainable  

**Acceptance Criteria:**
- [ ] requirements-base.txt for core dependencies
- [ ] requirements-dev.txt for development tools
- [ ] requirements-prod.txt for production only
- [ ] All duplicates and conflicts resolved
- [ ] Versions pinned appropriately

**Technical Implementation:**
- Files: `/opt/sutazaiapp/requirements*.txt`
- Changes: Analyze, deduplicate, consolidate all requirements
- Tests: Fresh install in clean environment

**Dependencies:** None  
**Effort:** 8 hours  
**Priority:** P0  

---

## PHASE 1: MVP - FIRST WORKING AGENT (2 WEEKS)

### Epic 1.1: Hardware Resource Optimizer Implementation

#### STORY ID: P1-AGENT-001
**Title:** Implement Real Hardware Monitoring  
**As a:** System Administrator  
**I want:** Real-time hardware metrics from the agent  
**So that:** I can monitor and optimize resource usage  

**Acceptance Criteria:**
- [ ] CPU usage reported accurately
- [ ] Memory metrics include used/free/available
- [ ] Disk I/O statistics collected
- [ ] Network throughput measured
- [ ] Process count and top consumers identified

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/agents/hardware-resource-optimizer/app.py
import psutil
from datetime import datetime

@app.route('/process', methods=['POST'])
def process():
    """Return REAL system metrics"""
    return jsonify({
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "cores": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current
            },
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "total_gb": psutil.virtual_memory().total / (1024**3)
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "used_gb": psutil.disk_usage('/').used / (1024**3),
                "free_gb": psutil.disk_usage('/').free / (1024**3)
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
                "packets_sent": psutil.net_io_counters().packets_sent,
                "packets_recv": psutil.net_io_counters().packets_recv
            },
            "processes": {
                "total": len(psutil.pids()),
                "top_cpu": [p.info for p in psutil.process_iter(['pid', 'name', 'cpu_percent']) 
                           if p.info['cpu_percent'] > 0][:5],
                "top_memory": [p.info for p in psutil.process_iter(['pid', 'name', 'memory_percent'])
                             if p.info['memory_percent'] > 0][:5]
            }
        }
    })
```

**Dependencies:** P0-MODEL-001  
**Effort:** 16 hours  
**Priority:** P1  

---

#### STORY ID: P1-AGENT-002
**Title:** Implement Resource Optimization Actions  
**As a:** System Administrator  
**I want:** The agent to optimize resources automatically  
**So that:** System performance improves without manual intervention  

**Acceptance Criteria:**
- [ ] Cache clearing functionality implemented
- [ ] Process priority adjustment working
- [ ] Memory garbage collection triggered
- [ ] Disk cleanup for temp files
- [ ] Docker container optimization

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/agents/hardware-resource-optimizer/optimizer.py

def optimize_memory():
    """Free up system memory"""
    gc.collect()
    # Clear system caches
    if os.path.exists('/proc/sys/vm/drop_caches'):
        subprocess.run(['sync'])
        subprocess.run(['echo', '1', '>', '/proc/sys/vm/drop_caches'])
    return {"freed_mb": get_freed_memory()}

def optimize_disk():
    """Clean temporary files and docker artifacts"""
    freed_space = 0
    # Clean temp directories
    for temp_dir in ['/tmp', '/var/tmp']:
        freed_space += clean_directory(temp_dir)
    # Docker cleanup
    subprocess.run(['docker', 'system', 'prune', '-f'])
    return {"freed_gb": freed_space / (1024**3)}

def optimize_processes():
    """Optimize running processes"""
    optimized = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        if proc.info['cpu_percent'] > 80:
            proc.nice(10)  # Lower priority
            optimized.append(proc.info['name'])
    return {"optimized_processes": optimized}
```

**Dependencies:** P1-AGENT-001  
**Effort:** 20 hours  
**Priority:** P1  

---

### Epic 1.2: Ollama Integration Enhancement

#### STORY ID: P1-OLLAMA-001
**Title:** Create Ollama Integration Service  
**As a:** AI Developer  
**I want:** Robust Ollama integration with queuing  
**So that:** Multiple LLM requests can be handled efficiently  

**Acceptance Criteria:**
- [ ] Request queue implemented with Redis
- [ ] Concurrent request handling (up to 5)
- [ ] Request timeout and retry logic
- [ ] Model switching capability
- [ ] Response caching for common prompts

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/services/ollama_service.py
import asyncio
from redis import Redis
import hashlib

class OllamaService:
    def __init__(self):
        self.redis = Redis(host='sutazai-redis', port=6379)
        self.model = "tinyllama"
        self.max_concurrent = 5
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
    async def generate(self, prompt: str, cache_ttl: int = 3600):
        # Check cache first
        cache_key = f"ollama:{hashlib.md5(prompt.encode()).hexdigest()}"
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
            
        async with self.semaphore:
            response = await self._call_ollama(prompt)
            # Cache the response
            self.redis.setex(cache_key, cache_ttl, json.dumps(response))
            return response
            
    async def _call_ollama(self, prompt: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://sutazai-ollama:11434/api/generate",
                json={"model": self.model, "prompt": prompt},
                timeout=30.0
            )
            return response.json()
```

**Dependencies:** P0-MODEL-001  
**Effort:** 16 hours  
**Priority:** P1  

---

### Epic 1.3: Basic API Endpoints

#### STORY ID: P1-API-001
**Title:** Complete Core CRUD Operations  
**As a:** Frontend Developer  
**I want:** All CRUD endpoints functional  
**So that:** The UI can perform all basic operations  

**Acceptance Criteria:**
- [ ] User CRUD operations working
- [ ] Agent CRUD operations working
- [ ] Task CRUD operations working
- [ ] Proper validation on all inputs
- [ ] Consistent error responses

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/api/v1/users.py
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List

router = APIRouter(prefix="/api/v1/users", tags=["users"])

@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.post("/", response_model=UserResponse)
async def create_user(
    user: UserCreate, 
    db: Session = Depends(get_db)
):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user: UserUpdate,
    db: Session = Depends(get_db)
):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    for key, value in user.dict(exclude_unset=True).items():
        setattr(db_user, key, value)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    return {"status": "deleted"}
```

**Dependencies:** P0-DATA-001  
**Effort:** 24 hours  
**Priority:** P1  

---

### Epic 1.4: Frontend Dashboard Connection

#### STORY ID: P1-UI-001
**Title:** Create Real-Time System Dashboard  
**As a:** System Operator  
**I want:** Live dashboard showing all system metrics  
**So that:** I can monitor system health at a glance  

**Acceptance Criteria:**
- [ ] WebSocket connection for real-time updates
- [ ] CPU, Memory, Disk metrics displayed
- [ ] Agent status indicators
- [ ] Task queue visualization
- [ ] Auto-refresh every 5 seconds

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/frontend/pages/dashboard.py
import streamlit as st
import asyncio
import websockets
import json

def render_dashboard():
    st.title("SutazAI System Dashboard")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_metric = st.metric("CPU Usage", "0%", "0%")
    with col2:
        mem_metric = st.metric("Memory", "0 GB", "0%")
    with col3:
        disk_metric = st.metric("Disk", "0 GB", "0%")
    with col4:
        agents_metric = st.metric("Active Agents", "0", "0")
    
    # WebSocket connection for real-time updates
    async def update_metrics():
        uri = "ws://sutazai-backend:10010/ws/metrics"
        async with websockets.connect(uri) as websocket:
            while True:
                data = await websocket.recv()
                metrics = json.loads(data)
                
                cpu_metric.metric(
                    "CPU Usage", 
                    f"{metrics['cpu']['percent']:.1f}%",
                    f"{metrics['cpu']['delta']:.1f}%"
                )
                mem_metric.metric(
                    "Memory",
                    f"{metrics['memory']['used_gb']:.1f} GB",
                    f"{metrics['memory']['percent']:.1f}%"
                )
                disk_metric.metric(
                    "Disk",
                    f"{metrics['disk']['used_gb']:.1f} GB",
                    f"{metrics['disk']['percent']:.1f}%"
                )
                agents_metric.metric(
                    "Active Agents",
                    metrics['agents']['active'],
                    metrics['agents']['delta']
                )
                
    # Run async update loop
    if st.button("Start Monitoring"):
        asyncio.run(update_metrics())
```

**Dependencies:** P1-API-001  
**Effort:** 16 hours  
**Priority:** P1  

---

## PHASE 2: AGENT EXPANSION (4 WEEKS)

### Epic 2.1: Task Assignment Logic

#### STORY ID: P2-TASK-001
**Title:** Implement Intelligent Task Router  
**As a:** System Architect  
**I want:** Tasks automatically routed to appropriate agents  
**So that:** Workload is distributed efficiently  

**Acceptance Criteria:**
- [ ] Task classification algorithm implemented
- [ ] Agent capability registry maintained
- [ ] Load balancing across agents
- [ ] Priority queue for urgent tasks
- [ ] Task retry on failure

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/agents/task-assignment-coordinator/task_router.py
from typing import Dict, List, Optional
import heapq

class TaskRouter:
    def __init__(self):
        self.agent_capabilities = {
            "hardware-optimizer": ["resource", "performance", "cleanup"],
            "ai-orchestrator": ["ml", "nlp", "generation"],
            "multi-agent": ["coordination", "workflow", "pipeline"],
            "resource-arbitration": ["allocation", "scheduling", "conflict"],
            "ollama-integration": ["llm", "prompt", "completion"]
        }
        self.task_queue = []  # Priority queue
        self.agent_load = {}  # Current load per agent
        
    def classify_task(self, task: Dict) -> str:
        """Classify task and return best agent"""
        task_type = task.get("type", "").lower()
        keywords = task.get("keywords", [])
        
        scores = {}
        for agent, capabilities in self.agent_capabilities.items():
            score = sum(1 for cap in capabilities 
                       if cap in task_type or cap in keywords)
            scores[agent] = score
            
        # Consider current load
        best_agent = min(scores.items(), 
                        key=lambda x: (x[1], self.agent_load.get(x[0], 0)))
        return best_agent[0]
        
    def assign_task(self, task: Dict) -> Dict:
        """Assign task to appropriate agent"""
        agent = self.classify_task(task)
        priority = task.get("priority", 5)
        
        # Add to priority queue
        heapq.heappush(self.task_queue, (priority, task["id"], agent, task))
        
        # Update load
        self.agent_load[agent] = self.agent_load.get(agent, 0) + 1
        
        return {
            "task_id": task["id"],
            "assigned_to": agent,
            "priority": priority,
            "estimated_completion": self.estimate_completion(agent)
        }
```

**Dependencies:** P1-AGENT-001  
**Effort:** 24 hours  
**Priority:** P2  

---

#### STORY ID: P2-TASK-002
**Title:** Implement Task Execution Pipeline  
**As a:** Backend Developer  
**I want:** Complete task lifecycle management  
**So that:** Tasks are tracked from creation to completion  

**Acceptance Criteria:**
- [ ] Task states: created, assigned, running, completed, failed
- [ ] State transitions logged
- [ ] Result storage in PostgreSQL
- [ ] Webhook notifications on completion
- [ ] Task cancellation capability

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/services/task_service.py
from enum import Enum
from sqlalchemy.orm import Session
import asyncio

class TaskState(Enum):
    CREATED = "created"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskService:
    def __init__(self, db: Session):
        self.db = db
        self.router = TaskRouter()
        
    async def create_task(self, task_data: Dict) -> Task:
        """Create and assign new task"""
        task = Task(
            **task_data,
            state=TaskState.CREATED,
            created_at=datetime.utcnow()
        )
        self.db.add(task)
        self.db.commit()
        
        # Assign to agent
        assignment = self.router.assign_task(task_data)
        task.assigned_to = assignment["assigned_to"]
        task.state = TaskState.ASSIGNED
        self.db.commit()
        
        # Trigger execution
        asyncio.create_task(self.execute_task(task))
        
        return task
        
    async def execute_task(self, task: Task):
        """Execute task on assigned agent"""
        try:
            task.state = TaskState.RUNNING
            task.started_at = datetime.utcnow()
            self.db.commit()
            
            # Call agent API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{task.assigned_to}:8000/process",
                    json=task.to_dict(),
                    timeout=300.0
                )
                result = response.json()
                
            task.state = TaskState.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow()
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            
        finally:
            self.db.commit()
            await self.send_webhook(task)
```

**Dependencies:** P2-TASK-001  
**Effort:** 20 hours  
**Priority:** P2  

---

### Epic 2.2: Multi-Agent Coordination

#### STORY ID: P2-COORD-001
**Title:** Implement Agent Communication Protocol  
**As a:** System Architect  
**I want:** Agents to communicate directly  
**So that:** Complex multi-step workflows can be executed  

**Acceptance Criteria:**
- [ ] Message passing via RabbitMQ
- [ ] Request-response pattern implemented
- [ ] Broadcast capability for announcements
- [ ] Message acknowledgment and retry
- [ ] Dead letter queue for failed messages

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/agents/multi-agent-coordinator/communication.py
import pika
import json
from typing import Dict, Callable

class AgentCommunicator:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('sutazai-rabbitmq', 5672)
        )
        self.channel = self.connection.channel()
        
        # Declare exchanges and queues
        self.channel.exchange_declare(exchange='agents', exchange_type='topic')
        self.channel.queue_declare(queue=f'agent.{agent_id}', durable=True)
        self.channel.queue_bind(
            exchange='agents',
            queue=f'agent.{agent_id}',
            routing_key=f'agent.{agent_id}.#'
        )
        
    def send_message(self, target_agent: str, message: Dict):
        """Send message to specific agent"""
        self.channel.basic_publish(
            exchange='agents',
            routing_key=f'agent.{target_agent}.message',
            body=json.dumps({
                'from': self.agent_id,
                'to': target_agent,
                'timestamp': datetime.utcnow().isoformat(),
                'message': message
            }),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
                expiration='300000'  # 5 minutes
            )
        )
        
    def broadcast(self, message: Dict):
        """Broadcast to all agents"""
        self.channel.basic_publish(
            exchange='agents',
            routing_key='agent.*.broadcast',
            body=json.dumps({
                'from': self.agent_id,
                'timestamp': datetime.utcnow().isoformat(),
                'message': message
            })
        )
        
    def listen(self, callback: Callable):
        """Listen for incoming messages"""
        def wrapper(ch, method, properties, body):
            message = json.loads(body)
            callback(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        self.channel.basic_consume(
            queue=f'agent.{self.agent_id}',
            on_message_callback=wrapper
        )
        self.channel.start_consuming()
```

**Dependencies:** P2-TASK-002  
**Effort:** 20 hours  
**Priority:** P2  

---

#### STORY ID: P2-COORD-002
**Title:** Create Workflow Orchestration Engine  
**As a:** Business Analyst  
**I want:** Complex workflows with multiple agents  
**So that:** Business processes can be automated end-to-end  

**Acceptance Criteria:**
- [ ] Workflow definition language (YAML/JSON)
- [ ] Sequential and parallel execution
- [ ] Conditional branching
- [ ] Error handling and compensation
- [ ] Workflow state persistence

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/workflow/engine.py
from typing import Dict, List, Optional
import yaml
import asyncio

class WorkflowEngine:
    def __init__(self):
        self.workflows = {}
        self.executions = {}
        
    def load_workflow(self, workflow_yaml: str) -> str:
        """Load workflow definition"""
        workflow = yaml.safe_load(workflow_yaml)
        workflow_id = workflow['id']
        self.workflows[workflow_id] = workflow
        return workflow_id
        
    async def execute_workflow(self, workflow_id: str, inputs: Dict) -> Dict:
        """Execute complete workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        execution_id = str(uuid.uuid4())
        self.executions[execution_id] = {
            'workflow_id': workflow_id,
            'state': 'running',
            'current_step': 0,
            'context': inputs,
            'results': {}
        }
        
        try:
            for step in workflow['steps']:
                await self.execute_step(execution_id, step)
                
            self.executions[execution_id]['state'] = 'completed'
            return self.executions[execution_id]['results']
            
        except Exception as e:
            self.executions[execution_id]['state'] = 'failed'
            self.executions[execution_id]['error'] = str(e)
            
            # Execute compensation if defined
            if 'on_failure' in workflow:
                await self.execute_compensation(execution_id, workflow['on_failure'])
                
            raise
            
    async def execute_step(self, execution_id: str, step: Dict):
        """Execute single workflow step"""
        execution = self.executions[execution_id]
        
        if step['type'] == 'parallel':
            # Execute steps in parallel
            tasks = [self.call_agent(s['agent'], s['action'], execution['context'])
                    for s in step['steps']]
            results = await asyncio.gather(*tasks)
            execution['results'][step['name']] = results
            
        elif step['type'] == 'sequential':
            # Execute steps sequentially
            for s in step['steps']:
                result = await self.call_agent(s['agent'], s['action'], execution['context'])
                execution['context'][s['output']] = result
                execution['results'][s['name']] = result
                
        elif step['type'] == 'conditional':
            # Evaluate condition and branch
            condition = eval(step['condition'], {'context': execution['context']})
            branch = step['then'] if condition else step.get('else', [])
            for s in branch:
                await self.execute_step(execution_id, s)
```

**Dependencies:** P2-COORD-001  
**Effort:** 32 hours  
**Priority:** P2  

---

### Epic 2.3: Resource Arbitration

#### STORY ID: P2-RESOURCE-001
**Title:** Implement Resource Allocation System  
**As a:** System Administrator  
**I want:** Automatic resource allocation to agents  
**So that:** System resources are used efficiently  

**Acceptance Criteria:**
- [ ] Resource pool management
- [ ] Fair allocation algorithm
- [ ] Priority-based allocation
- [ ] Resource reservation system
- [ ] Resource usage tracking

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/agents/resource-arbitration-agent/allocator.py
from typing import Dict, List, Optional
import threading

class ResourceAllocator:
    def __init__(self):
        self.resources = {
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_memory_gb': self.get_gpu_memory(),
            'disk_iops': 1000,
            'network_mbps': 1000
        }
        self.allocations = {}
        self.lock = threading.Lock()
        
    def request_resources(self, agent_id: str, requirements: Dict) -> Optional[Dict]:
        """Request resource allocation"""
        with self.lock:
            # Check availability
            available = self.get_available_resources()
            
            can_allocate = all(
                available.get(resource, 0) >= amount
                for resource, amount in requirements.items()
            )
            
            if not can_allocate:
                return None
                
            # Allocate resources
            allocation = {
                'agent_id': agent_id,
                'resources': requirements,
                'allocated_at': datetime.utcnow(),
                'lease_duration': 3600  # 1 hour default
            }
            
            self.allocations[agent_id] = allocation
            
            return allocation
            
    def release_resources(self, agent_id: str):
        """Release allocated resources"""
        with self.lock:
            if agent_id in self.allocations:
                del self.allocations[agent_id]
                return True
            return False
            
    def get_available_resources(self) -> Dict:
        """Calculate available resources"""
        available = self.resources.copy()
        
        for allocation in self.allocations.values():
            for resource, amount in allocation['resources'].items():
                available[resource] = available.get(resource, 0) - amount
                
        return available
        
    def enforce_limits(self):
        """Enforce resource limits on agents"""
        for agent_id, allocation in self.allocations.items():
            # Use cgroups to enforce CPU and memory limits
            cpu_limit = allocation['resources'].get('cpu_cores', 1)
            mem_limit = allocation['resources'].get('memory_gb', 1)
            
            subprocess.run([
                'docker', 'update',
                f'--cpus={cpu_limit}',
                f'--memory={mem_limit}g',
                f'sutazai-{agent_id}'
            ])
```

**Dependencies:** P2-COORD-001  
**Effort:** 20 hours  
**Priority:** P2  

---

### Epic 2.4: AI Orchestration

#### STORY ID: P2-AI-001
**Title:** Build AI Task Orchestration Framework  
**As a:** AI Engineer  
**I want:** Coordinated AI model execution  
**So that:** Complex AI pipelines can be executed  

**Acceptance Criteria:**
- [ ] Model registry with capabilities
- [ ] Pipeline definition and execution
- [ ] Model chaining and composition
- [ ] Result aggregation
- [ ] Fallback strategies

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/agents/ai-agent-orchestrator/orchestrator.py
from typing import Dict, List, Any
import asyncio

class AIOrchestrator:
    def __init__(self):
        self.models = {
            'tinyllama': {
                'type': 'llm',
                'capabilities': ['generation', 'completion', 'chat'],
                'max_tokens': 2048,
                'endpoint': 'http://sutazai-ollama:11434'
            }
        }
        self.pipelines = {}
        
    def create_pipeline(self, pipeline_def: Dict) -> str:
        """Create AI processing pipeline"""
        pipeline_id = str(uuid.uuid4())
        self.pipelines[pipeline_id] = {
            'id': pipeline_id,
            'name': pipeline_def['name'],
            'steps': pipeline_def['steps'],
            'created_at': datetime.utcnow()
        }
        return pipeline_id
        
    async def execute_pipeline(self, pipeline_id: str, input_data: Any) -> Dict:
        """Execute AI pipeline"""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
            
        context = {'input': input_data, 'results': {}}
        
        for step in pipeline['steps']:
            result = await self.execute_step(step, context)
            context['results'][step['name']] = result
            
            # Update context for next step
            if 'output_key' in step:
                context[step['output_key']] = result
                
        return context['results']
        
    async def execute_step(self, step: Dict, context: Dict) -> Any:
        """Execute single pipeline step"""
        step_type = step['type']
        
        if step_type == 'llm_generate':
            return await self.llm_generate(
                step['model'],
                step['prompt_template'].format(**context),
                step.get('parameters', {})
            )
            
        elif step_type == 'vector_search':
            return await self.vector_search(
                step['collection'],
                context[step['query_key']],
                step.get('top_k', 5)
            )
            
        elif step_type == 'aggregate':
            results = [context['results'][key] for key in step['inputs']]
            return self.aggregate_results(results, step['method'])
            
        elif step_type == 'classify':
            return await self.classify_text(
                context[step['text_key']],
                step['categories']
            )
            
    async def llm_generate(self, model: str, prompt: str, parameters: Dict) -> str:
        """Generate text using LLM"""
        model_config = self.models[model]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{model_config['endpoint']}/api/generate",
                json={
                    'model': model,
                    'prompt': prompt,
                    **parameters
                },
                timeout=60.0
            )
            return response.json()['response']
```

**Dependencies:** P2-COORD-002  
**Effort:** 28 hours  
**Priority:** P2  

---

## PHASE 3: SYSTEM INTEGRATION (6 WEEKS)

### Epic 3.1: Kong Gateway Configuration

#### STORY ID: P3-KONG-001
**Title:** Configure API Gateway Routes  
**As a:** API Developer  
**I want:** All services accessible through Kong  
**So that:** We have centralized API management  

**Acceptance Criteria:**
- [ ] Routes configured for all services
- [ ] Rate limiting implemented
- [ ] API key authentication
- [ ] Request/response transformation
- [ ] Load balancing configured

**Technical Implementation:**
```yaml
# File: /opt/sutazaiapp/kong/kong.yml
_format_version: "3.0"

services:
  - name: backend-api
    url: http://sutazai-backend:10010
    routes:
      - name: backend-route
        paths:
          - /api
        strip_path: false
    plugins:
      - name: rate-limiting
        config:
          second: 10
          minute: 100
      - name: key-auth
        config:
          key_names: ["apikey", "X-API-Key"]
          
  - name: agent-hardware
    url: http://sutazai-hardware-optimizer:8002
    routes:
      - name: hardware-route
        paths:
          - /agents/hardware
    plugins:
      - name: request-transformer
        config:
          add:
            headers:
              X-Agent-Type: hardware
              
  - name: agent-orchestrator
    url: http://sutazai-ai-orchestrator:8589
    routes:
      - name: orchestrator-route
        paths:
          - /agents/orchestrator
    plugins:
      - name: cors
        config:
          origins: ["*"]
          methods: ["GET", "POST", "PUT", "DELETE"]
```

**Dependencies:** P2-AI-001  
**Effort:** 16 hours  
**Priority:** P3  

---

### Epic 3.2: RabbitMQ Queue Setup

#### STORY ID: P3-RABBIT-001
**Title:** Implement Message Queue Architecture  
**As a:** System Architect  
**I want:** Reliable message passing between services  
**So that:** System components are loosely coupled  

**Acceptance Criteria:**
- [ ] Exchange topology defined
- [ ] Queue durability configured
- [ ] Dead letter exchanges
- [ ] Message TTL settings
- [ ] Consumer acknowledgments

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/messaging/rabbitmq_setup.py
import pika
from typing import Dict

class RabbitMQSetup:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('sutazai-rabbitmq', 5672)
        )
        self.channel = self.connection.channel()
        
    def setup_topology(self):
        """Setup complete RabbitMQ topology"""
        
        # Main exchanges
        self.channel.exchange_declare(
            exchange='tasks',
            exchange_type='topic',
            durable=True
        )
        
        self.channel.exchange_declare(
            exchange='agents',
            exchange_type='topic',
            durable=True
        )
        
        self.channel.exchange_declare(
            exchange='dlx',
            exchange_type='fanout',
            durable=True
        )
        
        # Task queues with DLX
        self.channel.queue_declare(
            queue='task.high_priority',
            durable=True,
            arguments={
                'x-dead-letter-exchange': 'dlx',
                'x-message-ttl': 300000,  # 5 minutes
                'x-max-length': 1000
            }
        )
        
        self.channel.queue_declare(
            queue='task.normal_priority',
            durable=True,
            arguments={
                'x-dead-letter-exchange': 'dlx',
                'x-message-ttl': 600000,  # 10 minutes
                'x-max-length': 5000
            }
        )
        
        # Agent queues
        for agent in ['hardware', 'orchestrator', 'coordinator', 'arbitration']:
            self.channel.queue_declare(
                queue=f'agent.{agent}',
                durable=True,
                arguments={
                    'x-dead-letter-exchange': 'dlx',
                    'x-max-priority': 10
                }
            )
            
        # Bindings
        self.channel.queue_bind(
            exchange='tasks',
            queue='task.high_priority',
            routing_key='task.*.high'
        )
        
        self.channel.queue_bind(
            exchange='tasks',
            queue='task.normal_priority',
            routing_key='task.*.normal'
        )
        
        # Dead letter queue
        self.channel.queue_declare(queue='dead_letters', durable=True)
        self.channel.queue_bind(exchange='dlx', queue='dead_letters')
```

**Dependencies:** P3-KONG-001  
**Effort:** 20 hours  
**Priority:** P3  

---

### Epic 3.3: Vector Database Integration

#### STORY ID: P3-VECTOR-001
**Title:** Integrate ChromaDB for Embeddings  
**As a:** AI Developer  
**I want:** Vector storage and retrieval working  
**So that:** We can implement semantic search  

**Acceptance Criteria:**
- [ ] ChromaDB client configured
- [ ] Embedding generation pipeline
- [ ] Collection management
- [ ] Similarity search working
- [ ] Metadata filtering

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/vector/chroma_service.py
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional

class ChromaService:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_server_host="sutazai-chromadb",
            chroma_server_http_port=8000
        ))
        self.embedding_model = self.load_embedding_model()
        
    def create_collection(self, name: str, metadata: Dict = None) -> chromadb.Collection:
        """Create or get collection"""
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {},
            embedding_function=self.embedding_function
        )
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        # Use sentence-transformers or similar
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, 'model'):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model.encode(text).tolist()
        
    def add_documents(self, collection_name: str, documents: List[Dict]):
        """Add documents to collection"""
        collection = self.create_collection(collection_name)
        
        embeddings = [self.embed_text(doc['text']) for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        ids = [doc['id'] for doc in documents]
        
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            documents=[doc['text'] for doc in documents]
        )
        
    def search(self, collection_name: str, query: str, 
              n_results: int = 5, where: Dict = None) -> Dict:
        """Search for similar documents"""
        collection = self.client.get_collection(collection_name)
        
        query_embedding = self.embed_text(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0],
            'ids': results['ids'][0]
        }
```

**Dependencies:** P3-RABBIT-001  
**Effort:** 24 hours  
**Priority:** P3  

---

#### STORY ID: P3-VECTOR-002
**Title:** Integrate Qdrant for Advanced Vector Operations  
**As a:** Data Scientist  
**I want:** Advanced vector search capabilities  
**So that:** We can implement recommendation systems  

**Acceptance Criteria:**
- [ ] Qdrant client configured
- [ ] Point insertion and updates
- [ ] Filtered search working
- [ ] Batch operations optimized
- [ ] Payload indexing

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/vector/qdrant_service.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List, Dict, Optional

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host="sutazai-qdrant",
            port=6333
        )
        
    def create_collection(self, name: str, vector_size: int = 384):
        """Create collection with vector configuration"""
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
    def insert_vectors(self, collection: str, vectors: List[Dict]):
        """Insert vectors with payloads"""
        points = []
        for vector_data in vectors:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_data['embedding'],
                payload=vector_data.get('metadata', {})
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=collection,
            points=points
        )
        
    def search_similar(self, collection: str, query_vector: List[float],
                       limit: int = 10, score_threshold: float = 0.7) -> List[Dict]:
        """Search for similar vectors"""
        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload
            }
            for hit in results
        ]
        
    def recommend(self, collection: str, positive: List[str], 
                 negative: List[str] = None, limit: int = 10) -> List[Dict]:
        """Get recommendations based on examples"""
        results = self.client.recommend(
            collection_name=collection,
            positive=positive,
            negative=negative or [],
            limit=limit
        )
        
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload
            }
            for hit in results
        ]
```

**Dependencies:** P3-VECTOR-001  
**Effort:** 20 hours  
**Priority:** P3  

---

### Epic 3.4: Service Discovery via Consul

#### STORY ID: P3-CONSUL-001
**Title:** Implement Service Registration  
**As a:** DevOps Engineer  
**I want:** Automatic service discovery  
**So that:** Services can find each other dynamically  

**Acceptance Criteria:**
- [ ] Service registration on startup
- [ ] Health check configuration
- [ ] Service deregistration on shutdown
- [ ] DNS integration
- [ ] Key-value configuration store

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/discovery/consul_service.py
import consul
from typing import Dict, Optional
import socket

class ConsulService:
    def __init__(self):
        self.consul = consul.Consul(
            host='sutazai-consul',
            port=8500
        )
        self.service_id = None
        
    def register_service(self, name: str, port: int, 
                        tags: List[str] = None, meta: Dict = None):
        """Register service with Consul"""
        hostname = socket.gethostname()
        service_id = f"{name}-{hostname}-{port}"
        
        # Register service
        self.consul.agent.service.register(
            name=name,
            service_id=service_id,
            address=hostname,
            port=port,
            tags=tags or [],
            meta=meta or {},
            check=consul.Check.http(
                f"http://{hostname}:{port}/health",
                interval="10s",
                timeout="5s",
                deregister="30s"
            )
        )
        
        self.service_id = service_id
        return service_id
        
    def discover_service(self, name: str) -> List[Dict]:
        """Discover available service instances"""
        _, services = self.consul.health.service(name, passing=True)
        
        return [
            {
                'id': service['Service']['ID'],
                'address': service['Service']['Address'],
                'port': service['Service']['Port'],
                'tags': service['Service']['Tags'],
                'meta': service['Service']['Meta']
            }
            for service in services
        ]
        
    def get_config(self, key: str) -> Optional[str]:
        """Get configuration from KV store"""
        _, data = self.consul.kv.get(key)
        if data:
            return data['Value'].decode('utf-8')
        return None
        
    def set_config(self, key: str, value: str):
        """Set configuration in KV store"""
        return self.consul.kv.put(key, value)
        
    def watch_config(self, key: str, callback):
        """Watch configuration changes"""
        index = None
        while True:
            index, data = self.consul.kv.get(key, index=index)
            if data:
                callback(data['Value'].decode('utf-8'))
```

**Dependencies:** P3-VECTOR-002  
**Effort:** 16 hours  
**Priority:** P3  

---

## PHASE 4: PRODUCTION READINESS (8 WEEKS)

### Epic 4.1: Authentication & Authorization

#### STORY ID: P4-AUTH-001
**Title:** Implement JWT Authentication  
**As a:** Security Engineer  
**I want:** Secure token-based authentication  
**So that:** APIs are protected from unauthorized access  

**Acceptance Criteria:**
- [ ] JWT token generation
- [ ] Token validation middleware
- [ ] Refresh token mechanism
- [ ] Token revocation
- [ ] Multi-factor authentication support

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/auth/jwt_service.py
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict
import redis

class JWTService:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(minutes=30)
        self.refresh_token_expire = timedelta(days=7)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.redis_client = redis.Redis(host='sutazai-redis', port=6379)
        
    def create_access_token(self, data: Dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.access_token_expire
        to_encode.update({"exp": expire, "type": "access"})
        
        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Store in Redis for revocation tracking
        self.redis_client.setex(
            f"token:access:{token[:10]}",
            int(self.access_token_expire.total_seconds()),
            token
        )
        
        return token
        
    def create_refresh_token(self, data: Dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.refresh_token_expire
        to_encode.update({"exp": expire, "type": "refresh"})
        
        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Store in Redis
        self.redis_client.setex(
            f"token:refresh:{data['sub']}",
            int(self.refresh_token_expire.total_seconds()),
            token
        )
        
        return token
        
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            # Check if token is revoked
            token_key = f"token:*:{token[:10]}"
            if not self.redis_client.keys(token_key):
                return None
                
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
            
        except JWTError:
            return None
            
    def revoke_token(self, token: str):
        """Revoke a token"""
        # Delete from Redis
        token_key = f"token:*:{token[:10]}"
        for key in self.redis_client.keys(token_key):
            self.redis_client.delete(key)
            
        # Add to blacklist
        self.redis_client.setex(
            f"blacklist:{token[:20]}",
            86400,  # 24 hours
            "revoked"
        )
```

**Dependencies:** P3-CONSUL-001  
**Effort:** 24 hours  
**Priority:** P4  

---

#### STORY ID: P4-AUTH-002
**Title:** Implement Role-Based Access Control  
**As a:** Security Administrator  
**I want:** Fine-grained permission control  
**So that:** Users only access authorized resources  

**Acceptance Criteria:**
- [ ] Role hierarchy defined
- [ ] Permission matrix implemented
- [ ] Resource-based permissions
- [ ] Role assignment API
- [ ] Audit logging for access

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/auth/rbac.py
from typing import List, Dict, Set
from enum import Enum

class Permission(Enum):
    # Agent permissions
    AGENT_VIEW = "agent:view"
    AGENT_CREATE = "agent:create"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_EXECUTE = "agent:execute"
    
    # Task permissions
    TASK_VIEW = "task:view"
    TASK_CREATE = "task:create"
    TASK_UPDATE = "task:update"
    TASK_DELETE = "task:delete"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"

class Role:
    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions
        
class RBACService:
    def __init__(self):
        self.roles = {
            'admin': Role('admin', {p for p in Permission}),
            'operator': Role('operator', {
                Permission.AGENT_VIEW,
                Permission.AGENT_EXECUTE,
                Permission.TASK_VIEW,
                Permission.TASK_CREATE,
                Permission.SYSTEM_MONITOR
            }),
            'viewer': Role('viewer', {
                Permission.AGENT_VIEW,
                Permission.TASK_VIEW,
                Permission.SYSTEM_MONITOR
            }),
            'developer': Role('developer', {
                Permission.AGENT_VIEW,
                Permission.AGENT_CREATE,
                Permission.AGENT_UPDATE,
                Permission.TASK_VIEW,
                Permission.TASK_CREATE,
                Permission.TASK_UPDATE
            })
        }
        
    def check_permission(self, user_roles: List[str], 
                        required_permission: Permission) -> bool:
        """Check if user has required permission"""
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if role and required_permission in role.permissions:
                return True
        return False
        
    def get_user_permissions(self, user_roles: List[str]) -> Set[Permission]:
        """Get all permissions for user"""
        permissions = set()
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if role:
                permissions.update(role.permissions)
        return permissions
        
    def require_permission(self, permission: Permission):
        """Decorator for permission checking"""
        def decorator(func):
            async def wrapper(request, *args, **kwargs):
                user = request.state.user
                if not self.check_permission(user.roles, permission):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Permission {permission.value} required"
                    )
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator
```

**Dependencies:** P4-AUTH-001  
**Effort:** 20 hours  
**Priority:** P4  

---

### Epic 4.2: Performance Optimization

#### STORY ID: P4-PERF-001
**Title:** Implement Response Caching  
**As a:** Performance Engineer  
**I want:** Intelligent response caching  
**So that:** System response times improve  

**Acceptance Criteria:**
- [ ] Redis cache implementation
- [ ] Cache invalidation strategy
- [ ] Cache hit rate monitoring
- [ ] Configurable TTL per endpoint
- [ ] Cache warming on startup

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/cache/cache_service.py
import redis
import hashlib
import json
from typing import Optional, Any, Callable
from functools import wraps

class CacheService:
    def __init__(self):
        self.redis = redis.Redis(
            host='sutazai-redis',
            port=6379,
            decode_responses=True
        )
        self.default_ttl = 300  # 5 minutes
        
    def cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        params = json.dumps(kwargs, sort_keys=True)
        hash_digest = hashlib.md5(params.encode()).hexdigest()
        return f"cache:{prefix}:{hash_digest}"
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None
        
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        self.redis.setex(
            key,
            ttl,
            json.dumps(value)
        )
        
    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        for key in self.redis.scan_iter(f"cache:{pattern}*"):
            self.redis.delete(key)
            
    def cached(self, prefix: str = None, ttl: int = None):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_prefix = prefix or func.__name__
                cache_key = self.cache_key(cache_prefix, args=args, kwargs=kwargs)
                
                # Check cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                    
                # Call function
                result = await func(*args, **kwargs)
                
                # Store in cache
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
        
    def warm_cache(self, warming_tasks: List[Dict]):
        """Warm cache with predefined queries"""
        for task in warming_tasks:
            endpoint = task['endpoint']
            params = task['params']
            
            # Call endpoint to populate cache
            asyncio.create_task(
                self._warm_endpoint(endpoint, params)
            )
```

**Dependencies:** P4-AUTH-002  
**Effort:** 16 hours  
**Priority:** P4  

---

#### STORY ID: P4-PERF-002
**Title:** Optimize Database Queries  
**As a:** Database Administrator  
**I want:** Optimized database performance  
**So that:** Query response times are minimized  

**Acceptance Criteria:**
- [ ] Query performance analysis
- [ ] Index optimization
- [ ] Connection pooling configured
- [ ] Query result caching
- [ ] Slow query logging

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/db/optimization.py
from sqlalchemy import create_engine, event, Index
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
import time
import logging

class DatabaseOptimizer:
    def __init__(self):
        self.engine = create_engine(
            "postgresql://sutazai:password@sutazai-postgres:5432/sutazai",
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.setup_listeners()
        
    def setup_listeners(self):
        """Setup query performance monitoring"""
        @event.listens_for(Engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault('query_start_time', []).append(time.time())
            
        @event.listens_for(Engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - conn.info['query_start_time'].pop(-1)
            if total > 1.0:  # Log slow queries
                logging.warning(f"Slow query ({total:.2f}s): {statement[:100]}")
                
    def create_indexes(self):
        """Create optimized indexes"""
        indexes = [
            Index('idx_users_email', 'users', 'email'),
            Index('idx_tasks_status_created', 'tasks', 'status', 'created_at'),
            Index('idx_agents_type_status', 'agents', 'type', 'status'),
            Index('idx_agent_executions_agent_task', 'agent_executions', 'agent_id', 'task_id'),
            Index('idx_system_metrics_timestamp', 'system_metrics', 'timestamp'),
        ]
        
        for index in indexes:
            index.create(self.engine, checkfirst=True)
            
    def analyze_tables(self):
        """Run ANALYZE on all tables"""
        with self.engine.connect() as conn:
            conn.execute("ANALYZE;")
            
    def vacuum_tables(self):
        """Run VACUUM on all tables"""
        with self.engine.connect() as conn:
            conn.execute("VACUUM ANALYZE;")
```

**Dependencies:** P4-PERF-001  
**Effort:** 20 hours  
**Priority:** P4  

---

### Epic 4.3: High Availability

#### STORY ID: P4-HA-001
**Title:** Implement Service Health Monitoring  
**As a:** Site Reliability Engineer  
**I want:** Comprehensive health monitoring  
**So that:** System availability is maximized  

**Acceptance Criteria:**
- [ ] Health check endpoints for all services
- [ ] Automatic failover mechanisms
- [ ] Circuit breaker implementation
- [ ] Service degradation handling
- [ ] Recovery procedures

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/reliability/health_monitor.py
from typing import Dict, List
import asyncio
from datetime import datetime, timedelta
from enum import Enum

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    
class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class HealthMonitor:
    def __init__(self):
        self.services = {}
        self.circuit_breakers = {}
        self.check_interval = 10  # seconds
        
    def register_service(self, name: str, health_check: Callable):
        """Register service for monitoring"""
        self.services[name] = {
            'health_check': health_check,
            'status': ServiceStatus.HEALTHY,
            'last_check': None,
            'consecutive_failures': 0
        }
        
        self.circuit_breakers[name] = {
            'state': CircuitBreakerState.CLOSED,
            'failure_count': 0,
            'last_failure': None,
            'success_count': 0
        }
        
    async def check_service_health(self, name: str) -> ServiceStatus:
        """Check health of a specific service"""
        service = self.services.get(name)
        if not service:
            return ServiceStatus.UNHEALTHY
            
        try:
            result = await service['health_check']()
            service['consecutive_failures'] = 0
            
            # Update circuit breaker
            cb = self.circuit_breakers[name]
            if cb['state'] == CircuitBreakerState.HALF_OPEN:
                cb['success_count'] += 1
                if cb['success_count'] >= 3:
                    cb['state'] = CircuitBreakerState.CLOSED
                    cb['failure_count'] = 0
                    
            return ServiceStatus.HEALTHY
            
        except Exception as e:
            service['consecutive_failures'] += 1
            
            # Update circuit breaker
            cb = self.circuit_breakers[name]
            cb['failure_count'] += 1
            cb['last_failure'] = datetime.utcnow()
            
            if cb['failure_count'] >= 5:
                cb['state'] = CircuitBreakerState.OPEN
                
            if service['consecutive_failures'] >= 3:
                return ServiceStatus.UNHEALTHY
            return ServiceStatus.DEGRADED
            
    async def monitor_all_services(self):
        """Continuously monitor all services"""
        while True:
            for name in self.services:
                status = await self.check_service_health(name)
                self.services[name]['status'] = status
                self.services[name]['last_check'] = datetime.utcnow()
                
                # Trigger alerts if needed
                if status == ServiceStatus.UNHEALTHY:
                    await self.trigger_alert(name, status)
                    
            await asyncio.sleep(self.check_interval)
            
    def can_call_service(self, name: str) -> bool:
        """Check if service can be called (circuit breaker)"""
        cb = self.circuit_breakers.get(name)
        if not cb:
            return True
            
        if cb['state'] == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if datetime.utcnow() - cb['last_failure'] > timedelta(seconds=30):
                cb['state'] = CircuitBreakerState.HALF_OPEN
                cb['success_count'] = 0
                return True
            return False
            
        return True
```

**Dependencies:** P4-PERF-002  
**Effort:** 24 hours  
**Priority:** P4  

---

### Epic 4.4: Documentation & Testing

#### STORY ID: P4-DOC-001
**Title:** Create Comprehensive API Documentation  
**As a:** Developer  
**I want:** Complete API documentation  
**So that:** I can integrate with the system easily  

**Acceptance Criteria:**
- [ ] OpenAPI/Swagger specification
- [ ] Interactive API explorer
- [ ] Code examples in multiple languages
- [ ] Authentication guide
- [ ] Rate limit documentation

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/backend/app/docs/api_docs.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html

def custom_openapi(app: FastAPI):
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title="SutazAI API",
        version="1.0.0",
        description="""
        ## Overview
        The SutazAI API provides programmatic access to AI agent orchestration capabilities.
        
        ## Authentication
        All API requests require authentication using JWT tokens or API keys.
        
        ### JWT Authentication
        ```bash
        curl -X POST https://api.sutazai.com/auth/token \\
          -H "Content-Type: application/json" \\
          -d '{"username": "user", "password": "pass"}'
        ```
        
        ### API Key Authentication
        ```bash
        curl -X GET https://api.sutazai.com/api/v1/agents \\
          -H "X-API-Key: your-api-key"
        ```
        
        ## Rate Limiting
        - 100 requests per minute for authenticated users
        - 10 requests per minute for unauthenticated users
        
        ## Endpoints
        """,
        routes=app.routes,
        tags=[
            {
                "name": "agents",
                "description": "AI Agent management and execution"
            },
            {
                "name": "tasks",
                "description": "Task creation and monitoring"
            },
            {
                "name": "workflows",
                "description": "Complex workflow orchestration"
            }
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "JWT": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "APIKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
```

**Dependencies:** P4-HA-001  
**Effort:** 16 hours  
**Priority:** P4  

---

#### STORY ID: P4-TEST-001
**Title:** Implement Comprehensive Test Suite  
**As a:** QA Engineer  
**I want:** Automated testing for all components  
**So that:** System reliability is ensured  

**Acceptance Criteria:**
- [ ] Unit tests with 80% coverage
- [ ] Integration tests for all APIs
- [ ] End-to-end workflow tests
- [ ] Performance tests
- [ ] Continuous integration pipeline

**Technical Implementation:**
```python
# File: /opt/sutazaiapp/tests/test_integration.py
import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy import create_engine
from typing import Dict

@pytest.fixture
async def client():
    """Create test client"""
    from backend.app.main import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def db():
    """Create test database"""
    engine = create_engine("postgresql://test:test@localhost:5432/test")
    # Create tables
    Base.metadata.create_all(engine)
    yield engine
    # Cleanup
    Base.metadata.drop_all(engine)

class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, client: AsyncClient, db):
        """Test complete agent lifecycle"""
        # Create agent
        response = await client.post(
            "/api/v1/agents",
            json={
                "name": "test-agent",
                "type": "hardware",
                "capabilities": ["monitoring", "optimization"]
            }
        )
        assert response.status_code == 201
        agent_id = response.json()["id"]
        
        # Get agent
        response = await client.get(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "test-agent"
        
        # Update agent
        response = await client.put(
            f"/api/v1/agents/{agent_id}",
            json={"status": "active"}
        )
        assert response.status_code == 200
        
        # Execute task on agent
        response = await client.post(
            f"/api/v1/agents/{agent_id}/execute",
            json={"action": "monitor", "parameters": {}}
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        
        # Wait for completion
        await asyncio.sleep(2)
        
        # Check task result
        response = await client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        assert response.json()["status"] in ["completed", "running"]
        
        # Delete agent
        response = await client.delete(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 204

class TestWorkflowIntegration:
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, client: AsyncClient):
        """Test complex multi-agent workflow"""
        # Create workflow
        workflow = {
            "name": "data-processing",
            "steps": [
                {
                    "type": "parallel",
                    "steps": [
                        {"agent": "hardware", "action": "optimize"},
                        {"agent": "ai-orchestrator", "action": "prepare"}
                    ]
                },
                {
                    "type": "sequential",
                    "steps": [
                        {"agent": "task-coordinator", "action": "assign"},
                        {"agent": "multi-agent", "action": "execute"}
                    ]
                }
            ]
        }
        
        response = await client.post("/api/v1/workflows", json=workflow)
        assert response.status_code == 201
        workflow_id = response.json()["id"]
        
        # Execute workflow
        response = await client.post(
            f"/api/v1/workflows/{workflow_id}/execute",
            json={"input": {"data": "test"}}
        )
        assert response.status_code == 202
        execution_id = response.json()["execution_id"]
        
        # Monitor execution
        max_wait = 30
        while max_wait > 0:
            response = await client.get(f"/api/v1/executions/{execution_id}")
            if response.json()["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(1)
            max_wait -= 1
            
        assert response.json()["status"] == "completed"
```

**Dependencies:** P4-DOC-001  
**Effort:** 32 hours  
**Priority:** P4  

---

## SUCCESS METRICS

### Phase 0 Metrics (72 Hours)
- All containers status: HEALTHY
- Backend /health returns status: "healthy"
- Model configuration: tinyllama correctly configured
- Database: All 14 tables contain data
- Zero configuration errors in logs

### Phase 1 Metrics (2 Weeks)
- Hardware optimizer agent processing real metrics
- API response time: < 500ms for CRUD operations
- Ollama integration: 95% success rate
- Frontend dashboard: Real-time updates working
- Test coverage: > 60%

### Phase 2 Metrics (4 Weeks)
- All 7 agents returning real data (not stubs)
- Task completion rate: > 90%
- Agent communication latency: < 100ms
- Workflow execution success: > 85%
- Concurrent task handling: 50+ tasks

### Phase 3 Metrics (6 Weeks)
- Kong routing all traffic successfully
- RabbitMQ message throughput: 1000+ msg/sec
- Vector search latency: < 200ms
- Service discovery time: < 50ms
- System integration tests: 100% passing

### Phase 4 Metrics (8 Weeks)
- System uptime: 99.9%
- Concurrent users: 100+
- API response time (p95): < 1 second
- Authentication success rate: 99.9%
- Test coverage: > 80%
- Documentation completeness: 100%

---

## IMPLEMENTATION PRIORITIES

### Immediate (Must Fix Now)
1. Model configuration alignment (P0-MODEL-001)
2. Database initialization (P0-DATA-001)
3. Container health stabilization (P0-HEALTH-001)

### Week 1 (Foundation)
1. Requirements consolidation (P0-REQ-001)
2. Hardware optimizer implementation (P1-AGENT-001)
3. Basic API endpoints (P1-API-001)

### Week 2-3 (Core Features)
1. Ollama integration (P1-OLLAMA-001)
2. Task assignment logic (P2-TASK-001)
3. Frontend dashboard (P1-UI-001)

### Week 4-6 (Integration)
1. Multi-agent coordination (P2-COORD-001)
2. Kong gateway configuration (P3-KONG-001)
3. Vector database integration (P3-VECTOR-001)

### Week 7-8 (Production)
1. Authentication system (P4-AUTH-001)
2. Performance optimization (P4-PERF-001)
3. High availability (P4-HA-001)
4. Documentation and testing (P4-DOC-001, P4-TEST-001)

---

## RISK MITIGATION

### Technical Risks
- **Risk:** TinyLlama limitations for complex tasks
- **Mitigation:** Design modular system to swap models easily

- **Risk:** Agent stub replacement breaks system
- **Mitigation:** Implement one agent at a time with rollback capability

- **Risk:** Performance degradation with scale
- **Mitigation:** Early performance testing and optimization

### Operational Risks
- **Risk:** System instability during development
- **Mitigation:** Feature flags and gradual rollout

- **Risk:** Data loss during migrations
- **Mitigation:** Automated backups before any schema changes

---

## CONCLUSION

This document represents the authoritative feature and user story guide for transforming SutazAI from a proof-of-concept to a production-ready AI orchestration platform. Every story is grounded in the actual system state, with realistic timelines and measurable success criteria.

The phased approach ensures:
1. Critical issues are fixed immediately
2. Value is delivered incrementally
3. Each phase builds on stable foundation
4. System remains operational throughout development
5. All work is testable and measurable

Follow this guide strictly. Do not deviate into fantasy features. Build on what exists. Test everything. Document all changes.

---

**Document Control:**
- Version: 1.0.0
- Created: August 7, 2025
- Author: System Architect
- Review: Required before any feature implementation
- Updates: Must be approved by technical lead

**END OF DOCUMENT**