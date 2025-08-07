# SutazAI Architecture Enhancement Plan - Building on Existing Infrastructure

> Deployment Reality: This document references components such as Kong, Consul, and RabbitMQ. In the current docker-compose.yml they are not provisioned. Active messaging uses Redis Streams and HTTP between services. Treat these references as planned/optional; provision them explicitly if you intend to use them.

## Executive Summary

This document provides a comprehensive enhancement strategy for the SutazAI system that builds upon ALL existing components without removing anything. The plan focuses on configuring the service mesh, implementing real AI logic in agents, deploying missing services, and creating proper interconnections between all components.

## Current System Assessment

### ✅ Running Services (14 containers)
- **Core Infrastructure**: PostgreSQL, Redis, Neo4j
- **AI Platform**: Ollama (TinyLlama model)
- **Application Layer**: Backend (FastAPI), Frontend (Streamlit)
- **Vector Databases**: ChromaDB, Qdrant
- **Monitoring Stack**: Prometheus, Grafana, Loki, Promtail, Blackbox Exporter, cAdvisor

### ⚠️ Missing Critical Services (Need Deployment)
- **Service Mesh**: Kong Gateway, Consul, RabbitMQ
- **AI Agents**: 7 agent stubs need real implementation
- **Advanced Services**: 42 additional services defined but not running

## Phase 1: Service Mesh Configuration

### 1.1 Kong API Gateway Setup

```yaml
# /opt/sutazaiapp/config/kong/kong.yml
_format_version: "3.0"
_transform: true

services:
  - name: backend-api
    url: http://backend:8000
    routes:
      - name: backend-route
        paths:
          - /api/v1
        strip_path: false
        methods:
          - GET
          - POST
          - PUT
          - DELETE
    plugins:
      - name: rate-limiting
        config:
          minute: 100
          policy: local
      - name: cors
        config:
          origins:
            - http://localhost:10011
          credentials: true
      - name: request-transformer
        config:
          add:
            headers:
              - X-Kong-Proxy:true

  - name: ai-orchestrator
    url: http://ai-agent-orchestrator:8589
    routes:
      - name: orchestrator-route
        paths:
          - /ai/orchestrate
        strip_path: true
    plugins:
      - name: jwt
      - name: request-size-limiting
        config:
          allowed_payload_size: 10

  - name: vector-search
    url: http://qdrant:6333
    routes:
      - name: vector-route
        paths:
          - /vector
        strip_path: true
    plugins:
      - name: ip-restriction
        config:
          allow:
            - 172.0.0.0/8
            - 10.0.0.0/8

upstreams:
  - name: ai-agents-pool
    targets:
      - target: task-assignment-coordinator:8551
        weight: 100
      - target: multi-agent-coordinator:8587
        weight: 100
      - target: resource-arbitration-agent:8588
        weight: 100
    healthchecks:
      active:
        healthy:
          interval: 10
          successes: 2
        unhealthy:
          interval: 5
          tcp_failures: 3
      passive:
        healthy:
          successes: 5
        unhealthy:
          tcp_failures: 3
```

### 1.2 Consul Service Discovery Configuration

```javascript
// /opt/sutazaiapp/config/consul/service-registration.js
const consul = require('consul')();

// Register all services with Consul
const services = [
  {
    name: 'backend-api',
    id: 'backend-api-1',
    address: 'backend',
    port: 8000,
    tags: ['api', 'fastapi', 'primary'],
    check: {
      http: 'http://backend:8000/health',
      interval: '10s',
      timeout: '5s'
    }
  },
  {
    name: 'ollama-llm',
    id: 'ollama-1',
    address: 'ollama',
    port: 10104,
    tags: ['llm', 'ai', 'inference'],
    check: {
      http: 'http://ollama:10104/api/tags',
      interval: '30s',
      timeout: '10s'
    }
  },
  {
    name: 'ai-orchestrator',
    id: 'ai-orchestrator-1',
    address: 'ai-agent-orchestrator',
    port: 8589,
    tags: ['ai', 'agent', 'orchestrator'],
    check: {
      http: 'http://ai-agent-orchestrator:8589/health',
      interval: '15s',
      timeout: '5s'
    }
  }
];

// Service registration function
async function registerServices() {
  for (const service of services) {
    await consul.agent.service.register(service);
    console.log(`Registered service: ${service.name}`);
  }
}

// Health check watcher
consul.health.state('critical', (err, result) => {
  if (!err && result.length > 0) {
    console.log('Critical services:', result.map(s => s.ServiceName));
  }
});
```

### 1.3 RabbitMQ Message Queue Configuration

```python
# /opt/sutazaiapp/config/rabbitmq/setup.py
import pika
import json
from typing import Dict, Any

class RabbitMQSetup:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('rabbitmq', 5672)
        )
        self.channel = self.connection.channel()
        
    def setup_exchanges_and_queues(self):
        """Create all necessary exchanges and queues"""
        
        # Task distribution exchange
        self.channel.exchange_declare(
            exchange='ai.tasks',
            exchange_type='topic',
            durable=True
        )
        
        # Event broadcasting exchange
        self.channel.exchange_declare(
            exchange='ai.events',
            exchange_type='fanout',
            durable=True
        )
        
        # Agent coordination exchange
        self.channel.exchange_declare(
            exchange='ai.coordination',
            exchange_type='direct',
            durable=True
        )
        
        # Create queues for each agent
        agents = [
            'orchestrator',
            'task_assignment',
            'resource_arbitration',
            'multi_agent_coordinator',
            'hardware_optimizer',
            'ollama_integration',
            'metrics_exporter'
        ]
        
        for agent in agents:
            # Task queue
            queue_name = f'agent.{agent}.tasks'
            self.channel.queue_declare(queue=queue_name, durable=True)
            self.channel.queue_bind(
                exchange='ai.tasks',
                queue=queue_name,
                routing_key=f'task.{agent}.*'
            )
            
            # Event subscription
            event_queue = f'agent.{agent}.events'
            self.channel.queue_declare(queue=event_queue, durable=False)
            self.channel.queue_bind(
                exchange='ai.events',
                queue=event_queue
            )
            
            # Direct coordination
            coord_queue = f'agent.{agent}.coordination'
            self.channel.queue_declare(queue=coord_queue, durable=True)
            self.channel.queue_bind(
                exchange='ai.coordination',
                queue=coord_queue,
                routing_key=agent
            )
        
        # Priority queue for urgent tasks
        self.channel.queue_declare(
            queue='ai.priority.tasks',
            durable=True,
            arguments={'x-max-priority': 10}
        )
        
        print("RabbitMQ setup complete")
```

## Phase 2: Implementing Real AI Agent Logic

### 2.1 AI Agent Orchestrator - Real Implementation

```python
# /opt/sutazaiapp/agents/ai-agent-orchestrator/enhanced_app.py
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pika
import json
import redis
from datetime import datetime
import logging

class TaskRequest(BaseModel):
    task_type: str
    payload: Dict[str, Any]
    priority: int = 5
    timeout: int = 300

class AIAgentOrchestrator:
    """Real AI Agent Orchestrator with actual intelligence"""
    
    def __init__(self):
        self.app = FastAPI(title="AI Agent Orchestrator", version="2.0.0")
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.ollama_url = "http://ollama:10104"
        self.agents_registry = {}
        self.setup_rabbitmq()
        self.setup_routes()
        self.logger = logging.getLogger(__name__)
        
    def setup_rabbitmq(self):
        """Setup RabbitMQ connections"""
        self.rabbit_connection = pika.BlockingConnection(
            pika.ConnectionParameters('rabbitmq', 5672)
        )
        self.rabbit_channel = self.rabbit_connection.channel()
        
    async def analyze_task(self, request: TaskRequest) -> Dict[str, Any]:
        """Use Ollama to analyze and plan task execution"""
        
        prompt = f"""
        Analyze this task and create an execution plan:
        
        Task Type: {request.task_type}
        Payload: {json.dumps(request.payload, indent=2)}
        Priority: {request.priority}
        
        Determine:
        1. Which agents should handle this task
        2. The optimal execution order
        3. Resource requirements
        4. Expected completion time
        5. Potential risks or issues
        
        Respond in JSON format with:
        {{
            "agents_required": ["agent1", "agent2"],
            "execution_plan": [
                {{"agent": "name", "action": "description", "dependencies": []}}
            ],
            "resources": {{"cpu": "percent", "memory": "MB", "gpu": bool}},
            "estimated_time": "seconds",
            "confidence": 0.0-1.0,
            "risks": ["risk1", "risk2"]
        }}
        """
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "tinyllama",
                        "prompt": prompt,
                        "format": "json",
                        "stream": False
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.loads(result.get('response', '{}'))
                    
            except Exception as e:
                self.logger.error(f"Ollama analysis failed: {e}")
                return self.fallback_analysis(request)
    
    def fallback_analysis(self, request: TaskRequest) -> Dict[str, Any]:
        """Fallback logic when AI analysis fails"""
        # Intelligent routing based on task type
        agent_mapping = {
            "code_generation": ["code-generator", "qa-validator"],
            "system_optimization": ["hardware-optimizer", "resource-arbitrator"],
            "data_processing": ["data-processor", "vector-indexer"],
            "security_scan": ["security-scanner", "vulnerability-assessor"],
            "deployment": ["deployment-agent", "monitoring-agent"]
        }
        
        agents = agent_mapping.get(request.task_type, ["general-agent"])
        
        return {
            "agents_required": agents,
            "execution_plan": [
                {"agent": agent, "action": f"Process {request.task_type}", "dependencies": []}
                for agent in agents
            ],
            "resources": {"cpu": "50", "memory": "1024", "gpu": False},
            "estimated_time": 60,
            "confidence": 0.7,
            "risks": ["Fallback mode - reduced accuracy"]
        }
    
    async def coordinate_execution(self, task_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multi-agent task execution"""
        
        execution_results = []
        
        for step in plan['execution_plan']:
            agent = step['agent']
            action = step['action']
            
            # Check dependencies
            if step.get('dependencies'):
                await self.wait_for_dependencies(task_id, step['dependencies'])
            
            # Dispatch to agent via RabbitMQ
            message = {
                'task_id': task_id,
                'agent': agent,
                'action': action,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.rabbit_channel.basic_publish(
                exchange='ai.tasks',
                routing_key=f'task.{agent}.execute',
                body=json.dumps(message)
            )
            
            # Track in Redis
            self.redis_client.hset(
                f"task:{task_id}",
                f"step:{agent}",
                json.dumps({"status": "dispatched", "time": datetime.utcnow().isoformat()})
            )
            
            execution_results.append({
                "agent": agent,
                "status": "dispatched",
                "action": action
            })
        
        return {
            "task_id": task_id,
            "execution_status": "in_progress",
            "steps": execution_results
        }
    
    async def monitor_task(self, task_id: str) -> Dict[str, Any]:
        """Monitor task execution progress"""
        
        task_data = self.redis_client.hgetall(f"task:{task_id}")
        
        if not task_data:
            raise HTTPException(status_code=404, detail="Task not found")
        
        steps_status = {}
        for key, value in task_data.items():
            if key.startswith("step:"):
                agent_name = key.replace("step:", "")
                steps_status[agent_name] = json.loads(value)
        
        # Calculate overall progress
        total_steps = len(steps_status)
        completed_steps = sum(1 for s in steps_status.values() if s.get('status') == 'completed')
        
        progress = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        return {
            "task_id": task_id,
            "progress": progress,
            "steps": steps_status,
            "status": "completed" if progress == 100 else "in_progress"
        }
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/orchestrate")
        async def orchestrate_task(request: TaskRequest):
            """Main orchestration endpoint"""
            
            # Generate task ID
            task_id = f"task_{datetime.utcnow().timestamp()}"
            
            # Analyze task with AI
            plan = await self.analyze_task(request)
            
            # Store task plan
            self.redis_client.hset(
                f"task:{task_id}",
                "plan",
                json.dumps(plan)
            )
            
            # Execute coordination
            result = await self.coordinate_execution(task_id, plan)
            
            return {
                "task_id": task_id,
                "plan": plan,
                "execution": result
            }
        
        @self.app.get("/task/{task_id}")
        async def get_task_status(task_id: str):
            """Get task execution status"""
            return await self.monitor_task(task_id)
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "ai-agent-orchestrator",
                "version": "2.0.0",
                "capabilities": [
                    "task_analysis",
                    "multi_agent_coordination",
                    "intelligent_routing",
                    "execution_monitoring"
                ]
            }
```

### 2.2 Multi-Agent Coordinator Implementation

```python
# /opt/sutazaiapp/agents/multi-agent-coordinator/enhanced_app.py
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import networkx as nx
from dataclasses import dataclass
import json
import redis
import httpx
from datetime import datetime
import logging

@dataclass
class Agent:
    id: str
    capabilities: List[str]
    status: str
    workload: int
    performance_score: float

class MultiAgentCoordinator:
    """Intelligent multi-agent coordination with graph-based planning"""
    
    def __init__(self):
        self.app = FastAPI(title="Multi-Agent Coordinator", version="2.0.0")
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.agent_graph = nx.DiGraph()
        self.active_agents: Dict[str, Agent] = {}
        self.ollama_url = "http://ollama:10104"
        self.setup_routes()
        self.logger = logging.getLogger(__name__)
        
    def register_agent(self, agent: Agent):
        """Register an agent in the coordination network"""
        self.active_agents[agent.id] = agent
        self.agent_graph.add_node(agent.id, **agent.__dict__)
        
        # Update Redis registry
        self.redis_client.hset(
            "agents:registry",
            agent.id,
            json.dumps({
                "capabilities": agent.capabilities,
                "status": agent.status,
                "registered_at": datetime.utcnow().isoformat()
            })
        )
    
    async def create_collaboration_plan(self, task: Dict[str, Any]) -> nx.DiGraph:
        """Create an optimal collaboration graph for task execution"""
        
        prompt = f"""
        Given this task and available agents, create an optimal collaboration plan:
        
        Task: {json.dumps(task, indent=2)}
        
        Available Agents:
        {json.dumps([{
            "id": a.id,
            "capabilities": a.capabilities,
            "workload": a.workload
        } for a in self.active_agents.values()], indent=2)}
        
        Create a collaboration plan with:
        1. Agent assignments
        2. Communication paths
        3. Data flow between agents
        4. Synchronization points
        
        Respond in JSON format with:
        {{
            "nodes": [{{"agent": "id", "role": "description", "priority": 1-10}}],
            "edges": [{{"from": "agent1", "to": "agent2", "data": "type", "weight": 1-10}}],
            "synchronization_points": ["description"],
            "estimated_completion": "seconds"
        }}
        """
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "tinyllama",
                        "prompt": prompt,
                        "format": "json",
                        "stream": False
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    plan = json.loads(result.get('response', '{}'))
                    return self.build_collaboration_graph(plan)
                    
            except Exception as e:
                self.logger.error(f"AI planning failed: {e}")
                return self.fallback_collaboration_plan(task)
    
    def build_collaboration_graph(self, plan: Dict[str, Any]) -> nx.DiGraph:
        """Build NetworkX graph from collaboration plan"""
        graph = nx.DiGraph()
        
        # Add nodes
        for node in plan.get('nodes', []):
            graph.add_node(
                node['agent'],
                role=node['role'],
                priority=node['priority']
            )
        
        # Add edges
        for edge in plan.get('edges', []):
            graph.add_edge(
                edge['from'],
                edge['to'],
                data=edge['data'],
                weight=edge['weight']
            )
        
        return graph
    
    async def execute_collaboration(self, graph: nx.DiGraph, task_id: str) -> Dict[str, Any]:
        """Execute the collaboration plan"""
        
        # Topological sort for execution order
        try:
            execution_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Handle cycles
            execution_order = list(graph.nodes())
        
        results = {}
        
        for agent_id in execution_order:
            if agent_id not in self.active_agents:
                continue
                
            agent = self.active_agents[agent_id]
            node_data = graph.nodes[agent_id]
            
            # Get incoming data from predecessors
            incoming_data = {}
            for predecessor in graph.predecessors(agent_id):
                if predecessor in results:
                    edge_data = graph.edges[predecessor, agent_id]
                    incoming_data[predecessor] = {
                        "data": results[predecessor],
                        "type": edge_data.get('data', 'unknown')
                    }
            
            # Execute agent task
            result = await self.dispatch_to_agent(
                agent,
                node_data['role'],
                incoming_data,
                task_id
            )
            
            results[agent_id] = result
        
        return {
            "task_id": task_id,
            "execution_order": execution_order,
            "results": results,
            "graph_metrics": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph)
            }
        }
    
    async def dispatch_to_agent(
        self,
        agent: Agent,
        role: str,
        incoming_data: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Dispatch work to specific agent"""
        
        # Update agent workload
        agent.workload += 1
        
        # Send task via HTTP (could also use RabbitMQ)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"http://{agent.id}:8080/process",
                    json={
                        "task_id": task_id,
                        "role": role,
                        "incoming_data": incoming_data
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    agent.workload -= 1
                    return result
                    
            except Exception as e:
                self.logger.error(f"Agent {agent.id} dispatch failed: {e}")
                agent.workload -= 1
                return {"error": str(e)}
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/coordinate")
        async def coordinate_task(task: Dict[str, Any]):
            """Coordinate multi-agent collaboration"""
            
            task_id = f"collab_{datetime.utcnow().timestamp()}"
            
            # Create collaboration plan
            graph = await self.create_collaboration_plan(task)
            
            # Execute collaboration
            result = await self.execute_collaboration(graph, task_id)
            
            return result
        
        @self.app.post("/register")
        async def register_agent(agent_data: Dict[str, Any]):
            """Register new agent"""
            
            agent = Agent(
                id=agent_data['id'],
                capabilities=agent_data.get('capabilities', []),
                status='active',
                workload=0,
                performance_score=1.0
            )
            
            self.register_agent(agent)
            
            return {"status": "registered", "agent_id": agent.id}
        
        @self.app.websocket("/ws/{agent_id}")
        async def websocket_endpoint(websocket: WebSocket, agent_id: str):
            """WebSocket for real-time agent communication"""
            
            await websocket.accept()
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Broadcast to other agents if needed
                    if message.get('broadcast'):
                        await self.broadcast_to_agents(message, exclude=agent_id)
                    
                    # Send acknowledgment
                    await websocket.send_json({
                        "type": "ack",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                self.logger.info(f"Agent {agent_id} disconnected")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "service": "multi-agent-coordinator",
                "active_agents": len(self.active_agents),
                "capabilities": [
                    "agent_registration",
                    "collaboration_planning",
                    "graph_based_coordination",
                    "real_time_communication"
                ]
            }
```

## Phase 3: Deploying Missing Services

### 3.1 Docker Compose Services Addition

```yaml
# /opt/sutazaiapp/docker-compose.services.yml
version: '3.8'

services:
  # Service Mesh Components
  kong:
    image: kong:3.5
    container_name: sutazai-kong
    environment:
      KONG_DATABASE: "off"
      KONG_DECLARATIVE_CONFIG: "/kong/kong.yml"
      KONG_PROXY_ACCESS_LOG: "/dev/stdout"
      KONG_ADMIN_ACCESS_LOG: "/dev/stdout"
      KONG_PROXY_ERROR_LOG: "/dev/stderr"
      KONG_ADMIN_ERROR_LOG: "/dev/stderr"
    ports:
      - "10005:8000"  # Proxy
      - "8001:8001"   # Admin API
    volumes:
      - ./config/kong:/kong
    networks:
      - sutazai-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "kong", "health"]
      interval: 30s
      timeout: 10s
      retries: 3

  consul:
    image: hashicorp/consul:latest
    container_name: sutazai-consul
    command: "agent -server -bootstrap-expect=1 -ui -client=0.0.0.0"
    ports:
      - "10006:8500"  # HTTP API/UI
      - "8600:8600/udp"  # DNS
    environment:
      CONSUL_BIND_INTERFACE: eth0
    volumes:
      - consul_data:/consul/data
    networks:
      - sutazai-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "consul", "members"]
      interval: 30s
      timeout: 10s
      retries: 3

  rabbitmq:
    image: rabbitmq:3.12-management
    container_name: sutazai-rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD:-sutazai_rabbit}
    ports:
      - "10007:5672"   # AMQP
      - "10008:15672"  # Management UI
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - sutazai-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Enhanced AI Agents with Real Logic
  ai-agent-orchestrator:
    build:
      context: ./agents/ai-agent-orchestrator
      dockerfile: Dockerfile
    container_name: sutazai-ai-agent-orchestrator
    environment:
      PORT: 8589
      REDIS_URL: redis://redis:6379/0
      RABBITMQ_URL: amqp://admin:${RABBITMQ_PASSWORD:-sutazai_rabbit}@rabbitmq:5672
      OLLAMA_URL: http://ollama:10104
      CONSUL_URL: http://consul:8500
    ports:
      - "8589:8589"
    depends_on:
      - redis
      - rabbitmq
      - consul
      - ollama
    networks:
      - sutazai-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8589/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  multi-agent-coordinator:
    build:
      context: ./agents/multi-agent-coordinator
      dockerfile: Dockerfile
    container_name: sutazai-multi-agent-coordinator
    environment:
      PORT: 8587
      REDIS_URL: redis://redis:6379/0
      RABBITMQ_URL: amqp://admin:${RABBITMQ_PASSWORD:-sutazai_rabbit}@rabbitmq:5672
      OLLAMA_URL: http://ollama:10104
    ports:
      - "8587:8587"
    depends_on:
      - redis
      - rabbitmq
      - ollama
    networks:
      - sutazai-network
    restart: unless-stopped

  task-assignment-coordinator:
    build:
      context: ./agents/task-assignment-coordinator
      dockerfile: Dockerfile
    container_name: sutazai-task-assignment-coordinator
    environment:
      PORT: 8551
      REDIS_URL: redis://redis:6379/0
      RABBITMQ_URL: amqp://admin:${RABBITMQ_PASSWORD:-sutazai_rabbit}@rabbitmq:5672
    ports:
      - "8551:8551"
    depends_on:
      - redis
      - rabbitmq
    networks:
      - sutazai-network
    restart: unless-stopped

  resource-arbitration-agent:
    build:
      context: ./agents/resource-arbitration-agent
      dockerfile: Dockerfile
    container_name: sutazai-resource-arbitration-agent
    environment:
      PORT: 8588
      REDIS_URL: redis://redis:6379/0
      RABBITMQ_URL: amqp://admin:${RABBITMQ_PASSWORD:-sutazai_rabbit}@rabbitmq:5672
    ports:
      - "8588:8588"
    depends_on:
      - redis
      - rabbitmq
    networks:
      - sutazai-network
    restart: unless-stopped

  ollama-integration-specialist:
    build:
      context: ./agents/ollama-integration
      dockerfile: Dockerfile
    container_name: sutazai-ollama-integration
    environment:
      PORT: 11015
      OLLAMA_URL: http://ollama:10104
      REDIS_URL: redis://redis:6379/0
    ports:
      - "11015:11015"
    depends_on:
      - ollama
      - redis
    networks:
      - sutazai-network
    restart: unless-stopped

  # Additional Essential Services
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: sutazai-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "10009:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - sutazai-network
    restart: unless-stopped

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: sutazai-kibana
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    ports:
      - "10012:5601"
    depends_on:
      - elasticsearch
    networks:
      - sutazai-network
    restart: unless-stopped

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: sutazai-jaeger
    environment:
      COLLECTOR_ZIPKIN_HOST_PORT: ":9411"
    ports:
      - "10013:16686"  # UI
      - "10014:14268"  # Collector
    networks:
      - sutazai-network
    restart: unless-stopped

  vault:
    image: hashicorp/vault:latest
    container_name: sutazai-vault
    cap_add:
      - IPC_LOCK
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: ${VAULT_TOKEN:-myroot}
      VAULT_DEV_LISTEN_ADDRESS: 0.0.0.0:8200
    ports:
      - "10015:8200"
    networks:
      - sutazai-network
    restart: unless-stopped

volumes:
  consul_data:
  rabbitmq_data:
  elasticsearch_data:

networks:
  sutazai-network:
    external: true
```

## Phase 4: System Integration and Wiring

### 4.1 Backend Integration Layer

```python
# /opt/sutazaiapp/backend/app/integration/service_mesh.py
from typing import Dict, Any, Optional, List
import httpx
import consul
import pika
import redis
from fastapi import HTTPException
import json
import asyncio
from datetime import datetime
import logging

class ServiceMeshIntegration:
    """Integration layer for all service mesh components"""
    
    def __init__(self):
        self.consul_client = consul.Consul(host='consul', port=8500)
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.setup_rabbitmq()
        self.kong_admin_url = "http://kong:8001"
        self.logger = logging.getLogger(__name__)
        
    def setup_rabbitmq(self):
        """Setup RabbitMQ connection"""
        self.rabbit_params = pika.ConnectionParameters(
            host='rabbitmq',
            port=5672,
            credentials=pika.PlainCredentials('admin', 'sutazai_rabbit')
        )
        
    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Discover service via Consul"""
        
        index, services = self.consul_client.health.service(service_name, passing=True)
        
        if services:
            # Return first healthy service
            service = services[0]
            return {
                "address": service['Service']['Address'],
                "port": service['Service']['Port'],
                "tags": service['Service']['Tags'],
                "status": "healthy"
            }
        
        return None
    
    async def route_request(self, path: str, method: str = "GET", data: Optional[Dict] = None):
        """Route request through Kong Gateway"""
        
        async with httpx.AsyncClient() as client:
            url = f"http://kong:8000{path}"
            
            if method == "GET":
                response = await client.get(url)
            elif method == "POST":
                response = await client.post(url, json=data)
            elif method == "PUT":
                response = await client.put(url, json=data)
            elif method == "DELETE":
                response = await client.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return response.json() if response.status_code == 200 else None
    
    async def publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish event to RabbitMQ"""
        
        connection = pika.BlockingConnection(self.rabbit_params)
        channel = connection.channel()
        
        message = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "backend-api"
        }
        
        channel.basic_publish(
            exchange='ai.events',
            routing_key='',
            body=json.dumps(message)
        )
        
        connection.close()
    
    async def dispatch_task(self, agent: str, task: Dict[str, Any]) -> str:
        """Dispatch task to specific agent via RabbitMQ"""
        
        task_id = f"task_{datetime.utcnow().timestamp()}"
        
        connection = pika.BlockingConnection(self.rabbit_params)
        channel = connection.channel()
        
        message = {
            "task_id": task_id,
            "task": task,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        channel.basic_publish(
            exchange='ai.tasks',
            routing_key=f'task.{agent}.execute',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                priority=task.get('priority', 5)
            )
        )
        
        connection.close()
        
        # Track in Redis
        self.redis_client.hset(
            f"task:{task_id}",
            "status",
            "dispatched"
        )
        
        return task_id
    
    async def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get status of all registered agents"""
        
        agents = []
        
        # Get from Consul
        index, services = self.consul_client.health.state('any')
        
        for service in services:
            if 'agent' in service['ServiceName']:
                agents.append({
                    "name": service['ServiceName'],
                    "status": service['Status'],
                    "node": service['Node'],
                    "output": service.get('Output', '')
                })
        
        # Get additional info from Redis
        agent_registry = self.redis_client.hgetall("agents:registry")
        
        for agent_id, data in agent_registry.items():
            agent_data = json.loads(data)
            # Find matching agent in consul list
            for agent in agents:
                if agent['name'] == agent_id:
                    agent.update(agent_data)
                    break
            else:
                # Agent in Redis but not Consul
                agents.append({
                    "name": agent_id,
                    "status": "unknown",
                    **agent_data
                })
        
        return agents
    
    async def setup_kong_routes(self):
        """Configure Kong Gateway routes programmatically"""
        
        routes = [
            {
                "name": "backend-api",
                "paths": ["/api/v1"],
                "service": {
                    "name": "backend-service",
                    "url": "http://backend:8000"
                }
            },
            {
                "name": "ai-orchestrator",
                "paths": ["/ai/orchestrate"],
                "service": {
                    "name": "orchestrator-service",
                    "url": "http://ai-agent-orchestrator:8589"
                }
            },
            {
                "name": "vector-search",
                "paths": ["/vector"],
                "service": {
                    "name": "vector-service",
                    "url": "http://qdrant:6333"
                }
            }
        ]
        
        async with httpx.AsyncClient() as client:
            for route_config in routes:
                # Create service
                service_response = await client.post(
                    f"{self.kong_admin_url}/services",
                    json=route_config['service']
                )
                
                if service_response.status_code in [201, 409]:  # Created or already exists
                    # Create route
                    route_response = await client.post(
                        f"{self.kong_admin_url}/services/{route_config['service']['name']}/routes",
                        json={
                            "name": route_config['name'],
                            "paths": route_config['paths']
                        }
                    )
                    
                    self.logger.info(f"Kong route configured: {route_config['name']}")
```

### 4.2 Vector Database Integration

```python
# /opt/sutazaiapp/backend/app/integration/vector_stores.py
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from chromadb import Client as ChromaClient
import faiss
import httpx
import json
import logging

class UnifiedVectorStore:
    """Unified interface for all vector databases"""
    
    def __init__(self):
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        self.chroma = ChromaClient(host="chromadb", port=8000)
        self.faiss_url = "http://faiss:8000"
        self.logger = logging.getLogger(__name__)
        
    async def create_collection(self, name: str, dimension: int, store: str = "all"):
        """Create collection in specified vector store(s)"""
        
        results = {}
        
        if store in ["all", "qdrant"]:
            try:
                self.qdrant.create_collection(
                    collection_name=name,
                    vectors_config={
                        "size": dimension,
                        "distance": "Cosine"
                    }
                )
                results["qdrant"] = "created"
            except Exception as e:
                results["qdrant"] = f"error: {e}"
        
        if store in ["all", "chroma"]:
            try:
                collection = self.chroma.create_collection(
                    name=name,
                    metadata={"dimension": dimension}
                )
                results["chroma"] = "created"
            except Exception as e:
                results["chroma"] = f"error: {e}"
        
        if store in ["all", "faiss"]:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.faiss_url}/create_index",
                        json={"name": name, "dimension": dimension}
                    )
                    results["faiss"] = "created" if response.status_code == 200 else "failed"
            except Exception as e:
                results["faiss"] = f"error: {e}"
        
        return results
    
    async def insert_vectors(
        self,
        collection: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        store: str = "all"
    ):
        """Insert vectors into specified store(s)"""
        
        results = {}
        
        if store in ["all", "qdrant"]:
            try:
                points = [
                    {
                        "id": i,
                        "vector": vector,
                        "payload": meta
                    }
                    for i, (vector, meta) in enumerate(zip(vectors, metadata))
                ]
                
                self.qdrant.upsert(
                    collection_name=collection,
                    points=points
                )
                results["qdrant"] = f"inserted {len(vectors)} vectors"
            except Exception as e:
                results["qdrant"] = f"error: {e}"
        
        if store in ["all", "chroma"]:
            try:
                collection_obj = self.chroma.get_collection(collection)
                collection_obj.add(
                    embeddings=vectors,
                    metadatas=metadata,
                    ids=[str(i) for i in range(len(vectors))]
                )
                results["chroma"] = f"inserted {len(vectors)} vectors"
            except Exception as e:
                results["chroma"] = f"error: {e}"
        
        if store in ["all", "faiss"]:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.faiss_url}/add_vectors",
                        json={
                            "index": collection,
                            "vectors": vectors,
                            "metadata": metadata
                        }
                    )
                    results["faiss"] = "inserted" if response.status_code == 200 else "failed"
            except Exception as e:
                results["faiss"] = f"error: {e}"
        
        return results
    
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        store: str = "qdrant"
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        
        if store == "qdrant":
            results = self.qdrant.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k
            )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": hit.payload
                }
                for hit in results
            ]
        
        elif store == "chroma":
            collection_obj = self.chroma.get_collection(collection)
            results = collection_obj.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            return [
                {
                    "id": results['ids'][0][i],
                    "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "metadata": results['metadatas'][0][i]
                }
                for i in range(len(results['ids'][0]))
            ]
        
        elif store == "faiss":
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.faiss_url}/search",
                    json={
                        "index": collection,
                        "vector": query_vector,
                        "k": top_k
                    }
                )
                if response.status_code == 200:
                    return response.json()['results']
        
        return []
```

## Phase 5: Monitoring and Observability Enhancement

### 5.1 Enhanced Prometheus Configuration

```yaml
# /opt/sutazaiapp/monitoring/prometheus/prometheus-enhanced.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - "rules/*.yml"

scrape_configs:
  # Service Mesh Monitoring
  - job_name: 'kong'
    static_configs:
      - targets: ['kong:8001']
    metrics_path: /metrics

  - job_name: 'consul'
    static_configs:
      - targets: ['consul:8500']
    metrics_path: /v1/agent/metrics
    params:
      format: ['prometheus']

  - job_name: 'rabbitmq'
    static_configs:
      - targets: ['rabbitmq:15692']
    metrics_path: /metrics

  # AI Agents Monitoring
  - job_name: 'ai-agents'
    consul_sd_configs:
      - server: 'consul:8500'
        services: ['ai-agent-orchestrator', 'multi-agent-coordinator', 'task-assignment-coordinator']
    relabel_configs:
      - source_labels: [__meta_consul_service]
        target_label: agent_name
      - source_labels: [__meta_consul_node]
        target_label: node

  # Vector Databases
  - job_name: 'vector-stores'
    static_configs:
      - targets: 
          - 'qdrant:6333'
          - 'chromadb:8000'
          - 'faiss:8000'
    relabel_configs:
      - source_labels: [__address__]
        regex: '([^:]+):.*'
        target_label: vector_store
        replacement: '${1}'

  # Application Metrics
  - job_name: 'backend-api'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /metrics

  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

### 5.2 Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "SutazAI System Overview",
    "panels": [
      {
        "title": "Service Mesh Health",
        "type": "graph",
        "targets": [
          {
            "expr": "up{job=~'kong|consul|rabbitmq'}"
          }
        ]
      },
      {
        "title": "AI Agent Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(agent_tasks_processed_total[5m])"
          }
        ]
      },
      {
        "title": "Vector Store Operations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(vector_operations_total[5m])"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100"
          }
        ]
      }
    ]
  }
}
```

## Implementation Roadmap

### Week 1: Service Mesh Setup
- [ ] Deploy Kong, Consul, RabbitMQ
- [ ] Configure service discovery
- [ ] Setup message queues
- [ ] Test inter-service communication

### Week 2: AI Agent Enhancement
- [ ] Implement real logic in orchestrator
- [ ] Add multi-agent coordination
- [ ] Connect agents to Ollama
- [ ] Setup agent communication via RabbitMQ

### Week 3: Integration Layer
- [ ] Connect backend to service mesh
- [ ] Implement vector store unified interface
- [ ] Setup monitoring for all services
- [ ] Create integration tests

### Week 4: Advanced Features
- [ ] Deploy additional services (Elasticsearch, Vault, etc.)
- [ ] Implement advanced agent behaviors
- [ ] Setup distributed tracing with Jaeger
- [ ] Performance optimization

## Testing Strategy

### 1. Service Mesh Tests
```bash
# Test Kong routing
curl -X GET http://localhost:10005/api/v1/health

# Test Consul service discovery
curl http://localhost:10006/v1/catalog/services

# Test RabbitMQ
curl -u admin:sutazai_rabbit http://localhost:10008/api/overview
```

### 2. Agent Integration Tests
```python
# Test orchestrator
import httpx

async def test_orchestrator():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8589/orchestrate",
            json={
                "task_type": "code_generation",
                "payload": {"language": "python", "description": "Hello world"},
                "priority": 5
            }
        )
        assert response.status_code == 200
        assert "task_id" in response.json()
```

### 3. End-to-End Tests
```python
# Full system test
async def test_full_pipeline():
    # 1. Submit task via Kong
    response = await client.post(
        "http://localhost:10005/ai/orchestrate",
        json={"task": "complex_analysis"}
    )
    
    # 2. Check task status
    task_id = response.json()["task_id"]
    status = await client.get(f"http://localhost:10005/task/{task_id}")
    
    # 3. Verify completion
    assert status.json()["status"] == "completed"
```

## Performance Optimization

### 1. Connection Pooling
- Implement connection pools for all services
- Reuse RabbitMQ connections
- Cache Consul service discoveries

### 2. Caching Strategy
- Redis for frequently accessed data
- In-memory caching for agent states
- Vector search result caching

### 3. Load Balancing
- Kong for HTTP load balancing
- RabbitMQ for task distribution
- Consul for service failover

## Security Enhancements

### 1. Authentication & Authorization
- Kong JWT plugin for API security
- Vault for secrets management
- Service-to-service mTLS

### 2. Network Segmentation
- Separate networks for different service tiers
- Firewall rules between services
- Encrypted communication channels

### 3. Audit Logging
- All API calls logged via Kong
- Agent actions tracked in Elasticsearch
- Security events to dedicated log stream

## Conclusion

This comprehensive enhancement plan builds upon all existing SutazAI components without removing anything. It transforms stub services into intelligent agents, properly configures the service mesh, and creates a fully integrated AI platform. The modular approach allows for incremental implementation while maintaining system stability throughout the enhancement process.

The architecture now supports:
- Scalable multi-agent AI processing
- Distributed task orchestration
- Unified vector search across multiple databases
- Complete observability and monitoring
- Production-ready service mesh
- Secure inter-service communication

All enhancements preserve backward compatibility and existing functionality while adding significant new capabilities to the system.
