# SutazAI Technical Architecture Documentation
## Complete Implementation Blueprint

**Version:** 1.0  
**Date:** August 5, 2025  
**Classification:** TECHNICAL SPECIFICATION  
**Purpose:** Detailed technical architecture for implementing the SutazAI distributed AI platform

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│                      API GATEWAY (Kong)                          │
│                         Port: 10005                              │
├─────────────────────────────────────────────────────────────────┤
│                     SERVICE MESH (Consul)                        │
│                         Port: 10006                              │
├─────────────────────────────────────────────────────────────────┤
│                      MESSAGE QUEUE (RabbitMQ)                    │
│                    Ports: 10041, 10042                          │
├─────────────────────────────────────────────────────────────────┤
│                         AGENT LAYER                              │
│              69 Specialized AI Agents (10300-10599)             │
├─────────────────────────────────────────────────────────────────┤
│                      LLM RUNTIME (Ollama)                        │
│                         Port: 10104                              │
├─────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                │
│   PostgreSQL (10000) | Redis (10001) | Neo4j (10002-10003)     │
│   ChromaDB (10100) | Qdrant (10101-10102) | FAISS (10103)      │
├─────────────────────────────────────────────────────────────────┤
│                    MONITORING & OBSERVABILITY                    │
│   Prometheus (10200) | Grafana (10201) | Loki | AlertManager   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Network Architecture

```yaml
networks:
  sutazai_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
          
  agent_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/16
          
  data_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.22.0.0/16
```

### 1.3 Communication Patterns

#### Request Flow
```
1. Client → Kong Gateway (Authentication/Rate Limiting)
2. Kong → Consul (Service Discovery)
3. Consul → Target Agent (Direct routing)
4. Agent → RabbitMQ (Task distribution)
5. Agent → Ollama (LLM inference)
6. Agent → Data Layer (Persistence)
7. Response → Client (via reverse path)
```

#### Inter-Agent Communication
```
Agent A → RabbitMQ Exchange → Topic → Agent B Queue → Agent B
                    ↓
              Priority Queue
                    ↓
              Dead Letter Exchange (for failures)
```

---

## 2. CORE INFRASTRUCTURE COMPONENTS

### 2.1 Database Architecture

#### PostgreSQL Configuration
```yaml
service: postgres
image: postgres:15-alpine
port: 10000
memory_limit: 2GB
environment:
  POSTGRES_DB: sutazai
  POSTGRES_USER: sutazai_user
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
  POSTGRES_MAX_CONNECTIONS: 200
  POSTGRES_SHARED_BUFFERS: 512MB
  POSTGRES_EFFECTIVE_CACHE_SIZE: 1GB
  POSTGRES_WORK_MEM: 4MB
volumes:
  - postgres_data:/var/lib/postgresql/data
  - ./init.sql:/docker-entrypoint-initdb.d/init.sql
```

#### Database Schema
```sql
-- Core tables
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL,
    port INTEGER UNIQUE,
    status VARCHAR(20) DEFAULT 'inactive',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    type VARCHAR(100),
    payload JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE agent_metrics (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES agents(id),
    metric_name VARCHAR(100),
    metric_value NUMERIC,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_metrics_agent_timestamp ON agent_metrics(agent_id, timestamp DESC);
```

### 2.2 Redis Configuration

```yaml
service: redis
image: redis:7-alpine
port: 10001
memory_limit: 1GB
command:
  - redis-server
  - --maxmemory 1gb
  - --maxmemory-policy allkeys-lru
  - --save 60 1000
  - --appendonly yes
  - --appendfsync everysec
volumes:
  - redis_data:/data
```

#### Redis Data Structures
```python
# Cache keys
cache:agent:{agent_id}:response:{hash}  # TTL: 3600s
cache:ollama:inference:{model}:{prompt_hash}  # TTL: 7200s

# Session management
session:{session_id}:data  # Hash
session:{session_id}:agents  # Set

# Rate limiting
ratelimit:{agent_id}:{window}  # Counter

# Pub/Sub channels
agent:broadcast  # System-wide announcements
agent:{agent_id}:commands  # Agent-specific commands
```

### 2.3 Neo4j Graph Database

```yaml
service: neo4j
image: neo4j:5-community
ports:
  - "10002:7474"  # HTTP
  - "10003:7687"  # Bolt
memory_limit: 4GB
environment:
  NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
  NEO4J_dbms_memory_heap_max__size: 2G
  NEO4J_dbms_memory_pagecache_size: 1G
volumes:
  - neo4j_data:/data
  - neo4j_logs:/logs
```

#### Graph Schema
```cypher
// Agent relationships
CREATE (a:Agent {id: $agent_id, name: $name, type: $type})

// Task dependencies
CREATE (t1:Task)-[:DEPENDS_ON]->(t2:Task)

// Agent collaboration
CREATE (a1:Agent)-[:COLLABORATES_WITH {frequency: $freq}]->(a2:Agent)

// Knowledge graph
CREATE (c:Concept {name: $concept})
CREATE (a:Agent)-[:KNOWS]->(c:Concept)
```

---

## 3. VECTOR DATABASE ARCHITECTURE

### 3.1 ChromaDB Configuration

```python
# chromadb_config.py
import chromadb
from chromadb.config import Settings

settings = Settings(
    chroma_server_host="0.0.0.0",
    chroma_server_http_port=10100,
    chroma_server_grpc_port=10100,
    anonymized_telemetry=False,
    persist_directory="/data/chroma",
    allow_reset=False
)

client = chromadb.HttpClient(
    host="localhost",
    port=10100,
    settings=settings
)

# Create collections
collections = {
    "agent_knowledge": {
        "metadata": {"hnsw:space": "cosine"},
        "embedding_function": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "code_embeddings": {
        "metadata": {"hnsw:space": "l2"},
        "embedding_function": "microsoft/codebert-base"
    },
    "documentation": {
        "metadata": {"hnsw:space": "ip"},
        "embedding_function": "sentence-transformers/all-mpnet-base-v2"
    }
}
```

### 3.2 Qdrant Configuration

```yaml
service: qdrant
image: qdrant/qdrant:latest
ports:
  - "10101:6333"  # HTTP API
  - "10102:6334"  # gRPC
memory_limit: 2GB
environment:
  QDRANT__SERVICE__HTTP_PORT: 6333
  QDRANT__SERVICE__GRPC_PORT: 6334
  QDRANT__STORAGE__STORAGE_PATH: /qdrant/storage
  QDRANT__STORAGE__SNAPSHOTS_PATH: /qdrant/snapshots
volumes:
  - qdrant_storage:/qdrant/storage
  - qdrant_snapshots:/qdrant/snapshots
```

### 3.3 FAISS Integration

```python
# faiss_service.py
import faiss
import numpy as np
from fastapi import FastAPI
import pickle

app = FastAPI()

class FAISSIndex:
    def __init__(self, dimension=768, index_type="IVF1024,Flat"):
        self.dimension = dimension
        self.index = faiss.index_factory(dimension, index_type)
        self.index.train(np.random.randn(10000, dimension).astype('float32'))
        
    def add_vectors(self, vectors, ids):
        self.index.add_with_ids(vectors, ids)
        
    def search(self, query_vector, k=10):
        distances, indices = self.index.search(query_vector, k)
        return indices, distances

@app.post("/index/{collection}/add")
async def add_vectors(collection: str, vectors: list, ids: list):
    # Implementation
    pass

@app.post("/index/{collection}/search")
async def search_vectors(collection: str, query: list, k: int = 10):
    # Implementation
    pass
```

---

## 4. OLLAMA LLM RUNTIME ARCHITECTURE

### 4.1 Ollama Service Configuration

```yaml
service: ollama
image: ollama/ollama:latest
port: 10104
memory_limit: 4GB
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
environment:
  OLLAMA_HOST: 0.0.0.0:11434
  OLLAMA_NUM_PARALLEL: 1
  OLLAMA_NUM_THREADS: 4
  OLLAMA_MAX_LOADED_MODELS: 1
  OLLAMA_KEEP_ALIVE: 30s
  OLLAMA_DEBUG: false
volumes:
  - ollama_models:/root/.ollama
  - ./modelfile:/modelfile
```

### 4.2 Model Management Strategy

```python
# ollama_manager.py
import httpx
import asyncio
from typing import Optional, Dict, Any

class OllamaManager:
    def __init__(self, base_url="http://localhost:10104"):
        self.base_url = base_url
        self.models = {
            "default": "tinyllama:latest",
            "reasoning": "mistral:7b-instruct-q4_K_M",
            "coding": "deepseek-coder:6.7b-instruct-q4_K_M",
            "analysis": "qwen2:7b-instruct-q4_K_M"
        }
        self.current_model = None
        self.lock = asyncio.Lock()
        
    async def ensure_model_loaded(self, model_type="default"):
        async with self.lock:
            model = self.models.get(model_type, self.models["default"])
            
            if self.current_model != model:
                # Unload current model
                if self.current_model:
                    await self._unload_model(self.current_model)
                
                # Load new model
                await self._load_model(model)
                self.current_model = model
                
    async def _load_model(self, model: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/pull",
                json={"name": model}
            )
            return response.json()
            
    async def _unload_model(self, model: str):
        # Ollama automatically unloads based on KEEP_ALIVE
        pass
        
    async def generate(self, prompt: str, model_type="default", **kwargs):
        await self.ensure_model_loaded(model_type)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.models[model_type],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 0.9),
                        "max_tokens": kwargs.get("max_tokens", 512),
                        "num_predict": kwargs.get("num_predict", 128)
                    }
                }
            )
            return response.json()
```

### 4.3 Request Queue Management

```python
# ollama_queue.py
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class InferenceRequest:
    id: str
    agent_id: str
    prompt: str
    model_type: str
    priority: int
    timestamp: float
    callback: Optional[callable] = None

class OllamaQueue:
    def __init__(self, max_concurrent=1):
        self.queues = {
            1: deque(),  # Critical
            2: deque(),  # High
            3: deque(),  # Normal
            4: deque(),  # Low
        }
        self.max_concurrent = max_concurrent
        self.active_requests = 0
        self.processing = False
        
    async def add_request(self, request: InferenceRequest):
        priority = min(max(request.priority, 1), 4)
        self.queues[priority].append(request)
        
        if not self.processing:
            asyncio.create_task(self._process_queue())
            
    async def _process_queue(self):
        self.processing = True
        ollama_manager = OllamaManager()
        
        while any(self.queues.values()):
            if self.active_requests >= self.max_concurrent:
                await asyncio.sleep(0.1)
                continue
                
            # Get highest priority request
            request = None
            for priority in sorted(self.queues.keys()):
                if self.queues[priority]:
                    request = self.queues[priority].popleft()
                    break
                    
            if request:
                self.active_requests += 1
                asyncio.create_task(self._process_request(request, ollama_manager))
                
        self.processing = False
        
    async def _process_request(self, request: InferenceRequest, manager: OllamaManager):
        try:
            result = await manager.generate(
                prompt=request.prompt,
                model_type=request.model_type
            )
            
            if request.callback:
                await request.callback(request.id, result)
                
        except Exception as e:
            print(f"Error processing request {request.id}: {e}")
            
        finally:
            self.active_requests -= 1
```

---

## 5. SERVICE MESH ARCHITECTURE

### 5.1 Consul Configuration

```hcl
# consul_config.hcl
datacenter = "sutazai-dc1"
data_dir = "/consul/data"
log_level = "INFO"
node_name = "consul-server"
server = true
bootstrap_expect = 1
encrypt = "CONSUL_ENCRYPT_KEY"
ui_config {
  enabled = true
}

connect {
  enabled = true
}

ports {
  grpc = 8502
  http = 8500
  serf_lan = 8301
  serf_wan = 8302
  server = 8300
}

services {
  name = "agent-registry"
  port = 8500
  check {
    http = "http://localhost:8500/v1/status/leader"
    interval = "10s"
  }
}
```

### 5.2 Kong API Gateway Configuration

```yaml
# kong.yml
_format_version: "3.0"

services:
  - name: agent-api
    url: http://consul:8500
    routes:
      - name: agent-route
        paths:
          - /api/v1/agents
        strip_path: false
        
plugins:
  - name: jwt
    config:
      key_claim_name: kid
      secret_is_base64: false
      
  - name: rate-limiting
    config:
      minute: 60
      hour: 1000
      policy: local
      
  - name: cors
    config:
      origins:
        - "*"
      methods:
        - GET
        - POST
        - PUT
        - DELETE
      headers:
        - Accept
        - Content-Type
        - Authorization
        
  - name: request-transformer
    config:
      add:
        headers:
          - "X-Request-ID:$(uuid)"
          
  - name: prometheus
    config:
      status_code_metrics: true
      latency_metrics: true
      bandwidth_metrics: true
      upstream_health_metrics: true
```

### 5.3 RabbitMQ Configuration

```yaml
# rabbitmq.conf
listeners.tcp.default = 5672
management.tcp.port = 15672
vm_memory_high_watermark.relative = 0.6
disk_free_limit.absolute = 2GB

# Clustering
cluster_formation.peer_discovery_backend = rabbit_peer_discovery_consul
cluster_formation.consul.host = consul
cluster_formation.consul.port = 8500
cluster_formation.consul.svc = rabbitmq
cluster_formation.consul.svc_addr_auto = true

# Queue defaults
queue_master_locator = min-masters
lazy_queue_explicit_gc_run_operation_threshold = 1000
```

#### Exchange and Queue Setup
```python
# rabbitmq_setup.py
import pika
import json

def setup_rabbitmq():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost', 10041)
    )
    channel = connection.channel()
    
    # Declare exchanges
    channel.exchange_declare(
        exchange='agent.tasks',
        exchange_type='topic',
        durable=True
    )
    
    channel.exchange_declare(
        exchange='agent.events',
        exchange_type='fanout',
        durable=True
    )
    
    channel.exchange_declare(
        exchange='agent.dlx',
        exchange_type='direct',
        durable=True
    )
    
    # Declare queues for each agent
    agents = load_agent_config()
    
    for agent in agents:
        # Task queue
        channel.queue_declare(
            queue=f'agent.{agent["id"]}.tasks',
            durable=True,
            arguments={
                'x-max-priority': 10,
                'x-message-ttl': 3600000,  # 1 hour
                'x-dead-letter-exchange': 'agent.dlx'
            }
        )
        
        # Bind to exchange
        channel.queue_bind(
            exchange='agent.tasks',
            queue=f'agent.{agent["id"]}.tasks',
            routing_key=f'agent.{agent["type"]}.#'
        )
    
    connection.close()
```

---

## 6. AGENT ARCHITECTURE

### 6.1 Base Agent Framework

```python
# base_agent.py
from abc import ABC, abstractmethod
import asyncio
import httpx
import json
from typing import Dict, Any, Optional
import consul
import pika
import redis

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.id = config['id']
        self.name = config['name']
        self.type = config['type']
        self.port = config['port']
        self.memory_limit = config.get('memory_limit', '256MB')
        
        # Service connections
        self.consul = consul.Consul(host='consul', port=8500)
        self.redis = redis.Redis(host='redis', port=10001)
        self.rabbit_connection = None
        self.rabbit_channel = None
        
        # Ollama client
        self.ollama_url = "http://ollama:10104"
        
        # Health status
        self.healthy = True
        self.metrics = {}
        
    async def initialize(self):
        """Initialize agent connections and register with service mesh"""
        # Register with Consul
        self.register_service()
        
        # Setup RabbitMQ
        self.setup_rabbitmq()
        
        # Start health check loop
        asyncio.create_task(self.health_check_loop())
        
        # Start metrics collection
        asyncio.create_task(self.metrics_loop())
        
    def register_service(self):
        """Register agent with Consul"""
        self.consul.agent.service.register(
            name=self.name,
            service_id=self.id,
            address='localhost',
            port=self.port,
            tags=[self.type, 'agent'],
            check=consul.Check.http(
                f"http://localhost:{self.port}/health",
                interval="10s",
                timeout="5s"
            )
        )
        
    def setup_rabbitmq(self):
        """Setup RabbitMQ connection and consumer"""
        self.rabbit_connection = pika.BlockingConnection(
            pika.ConnectionParameters('rabbitmq', 10041)
        )
        self.rabbit_channel = self.rabbit_connection.channel()
        
        # Declare agent's queue
        queue_name = f'agent.{self.id}.tasks'
        self.rabbit_channel.queue_declare(queue=queue_name, durable=True)
        
        # Set up consumer
        self.rabbit_channel.basic_consume(
            queue=queue_name,
            on_message_callback=self.handle_message,
            auto_ack=False
        )
        
    def handle_message(self, channel, method, properties, body):
        """Handle incoming RabbitMQ message"""
        try:
            message = json.loads(body)
            asyncio.create_task(self.process_task(message))
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Error handling message: {e}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task - must be implemented by each agent"""
        pass
        
    async def call_ollama(self, prompt: str, model_type: str = "default") -> str:
        """Call Ollama for LLM inference"""
        # Check cache first
        cache_key = f"cache:ollama:{model_type}:{hash(prompt)}"
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)['response']
            
        # Make request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.get_model_for_type(model_type),
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 512
                    }
                }
            )
            
        result = response.json()
        
        # Cache result
        self.redis.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(result)
        )
        
        return result['response']
        
    def get_model_for_type(self, model_type: str) -> str:
        """Get appropriate model for task type"""
        models = {
            "default": "tinyllama:latest",
            "reasoning": "mistral:7b-instruct-q4_K_M",
            "coding": "deepseek-coder:6.7b-instruct-q4_K_M",
            "analysis": "qwen2:7b-instruct-q4_K_M"
        }
        return models.get(model_type, models["default"])
        
    async def health_check_loop(self):
        """Periodic health check"""
        while True:
            try:
                self.healthy = await self.check_health()
                await asyncio.sleep(10)
            except Exception as e:
                print(f"Health check error: {e}")
                self.healthy = False
                
    @abstractmethod
    async def check_health(self) -> bool:
        """Check agent health - override for custom checks"""
        return True
        
    async def metrics_loop(self):
        """Collect and report metrics"""
        while True:
            try:
                self.metrics = await self.collect_metrics()
                # Report to monitoring system
                await self.report_metrics()
                await asyncio.sleep(15)
            except Exception as e:
                print(f"Metrics error: {e}")
                
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect agent metrics - override for custom metrics"""
        return {
            "requests_processed": 0,
            "errors": 0,
            "latency_ms": 0
        }
        
    async def report_metrics(self):
        """Report metrics to Prometheus"""
        # Implementation for Prometheus pushgateway
        pass
```

### 6.2 Specialized Agent Example

```python
# senior_backend_developer_agent.py
from base_agent import BaseAgent
from typing import Dict, Any
import ast
import black

class SeniorBackendDeveloperAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.specializations = [
            "api_design",
            "database_optimization",
            "microservices",
            "performance_tuning",
            "security"
        ]
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process backend development tasks"""
        task_type = task.get('type')
        
        if task_type == 'api_design':
            return await self.design_api(task)
        elif task_type == 'code_review':
            return await self.review_code(task)
        elif task_type == 'optimization':
            return await self.optimize_code(task)
        elif task_type == 'architecture':
            return await self.design_architecture(task)
        else:
            return await self.general_backend_task(task)
            
    async def design_api(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Design REST API endpoints"""
        requirements = task.get('requirements', '')
        
        prompt = f"""
        As a senior backend developer, design a REST API for the following requirements:
        {requirements}
        
        Include:
        1. Endpoint definitions with HTTP methods
        2. Request/response schemas
        3. Authentication requirements
        4. Rate limiting recommendations
        5. Error handling patterns
        
        Format as OpenAPI 3.0 specification.
        """
        
        response = await self.call_ollama(prompt, model_type="reasoning")
        
        return {
            "task_id": task['id'],
            "result": response,
            "type": "api_design",
            "agent": self.name
        }
        
    async def review_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review backend code for quality and best practices"""
        code = task.get('code', '')
        language = task.get('language', 'python')
        
        # Parse code for basic validation
        if language == 'python':
            try:
                ast.parse(code)
                # Format with black
                formatted = black.format_str(code, mode=black.Mode())
            except SyntaxError as e:
                return {
                    "task_id": task['id'],
                    "error": f"Syntax error: {e}",
                    "type": "code_review",
                    "agent": self.name
                }
                
        prompt = f"""
        Review the following {language} code for:
        1. Security vulnerabilities
        2. Performance issues
        3. Best practices
        4. Error handling
        5. Code organization
        
        Code:
        ```{language}
        {code}
        ```
        
        Provide specific recommendations with line numbers.
        """
        
        response = await self.call_ollama(prompt, model_type="coding")
        
        return {
            "task_id": task['id'],
            "result": response,
            "formatted_code": formatted if language == 'python' else code,
            "type": "code_review",
            "agent": self.name
        }
        
    async def check_health(self) -> bool:
        """Check agent-specific health"""
        # Check database connection
        try:
            # Test Redis
            self.redis.ping()
            
            # Test RabbitMQ
            if self.rabbit_connection and not self.rabbit_connection.is_closed:
                return True
                
        except Exception:
            return False
            
        return True
        
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect backend-specific metrics"""
        return {
            "requests_processed": self.redis.get(f"agent:{self.id}:requests") or 0,
            "api_designs": self.redis.get(f"agent:{self.id}:api_designs") or 0,
            "code_reviews": self.redis.get(f"agent:{self.id}:reviews") or 0,
            "avg_response_time": self.redis.get(f"agent:{self.id}:avg_time") or 0
        }
```

---

## 7. MONITORING ARCHITECTURE

### 7.1 Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'sutazai-production'
    
rule_files:
  - '/etc/prometheus/rules/*.yml'
  
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
        
scrape_configs:
  # Agent discovery via Consul
  - job_name: 'agents'
    consul_sd_configs:
      - server: 'consul:8500'
        services: []
    relabel_configs:
      - source_labels: [__meta_consul_service]
        target_label: agent_name
      - source_labels: [__meta_consul_service_id]
        target_label: agent_id
      - source_labels: [__meta_consul_tags]
        target_label: agent_type
        
  # Infrastructure metrics
  - job_name: 'infrastructure'
    static_configs:
      - targets:
          - 'postgres-exporter:9187'
          - 'redis-exporter:9121'
          - 'rabbitmq-exporter:9419'
          - 'node-exporter:9100'
          
  # Ollama metrics
  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    metrics_path: '/metrics'
```

### 7.2 Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "SutazAI System Overview",
    "panels": [
      {
        "title": "Agent Status Grid",
        "type": "stat",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "up{job='agents'}",
            "legendFormat": "{{agent_name}}"
          }
        ]
      },
      {
        "title": "Resource Utilization",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "container_memory_usage_bytes{name=~'sutazai.*'}/1024/1024/1024",
            "legendFormat": "{{name}} - Memory GB"
          },
          {
            "expr": "rate(container_cpu_usage_seconds_total{name=~'sutazai.*'}[5m])*100",
            "legendFormat": "{{name}} - CPU %"
          }
        ]
      },
      {
        "title": "Ollama Inference Latency",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ollama_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95 Latency"
          }
        ]
      },
      {
        "title": "Task Queue Depth",
        "type": "graph",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
        "targets": [
          {
            "expr": "rabbitmq_queue_messages{queue=~'agent.*'}",
            "legendFormat": "{{queue}}"
          }
        ]
      }
    ]
  }
}
```

### 7.3 Alerting Rules

```yaml
# alert_rules.yml
groups:
  - name: agent_alerts
    interval: 30s
    rules:
      - alert: AgentDown
        expr: up{job="agents"} == 0
        for: 1m
        labels:
          severity: critical
          component: agent
        annotations:
          summary: "Agent {{ $labels.agent_name }} is down"
          description: "Agent {{ $labels.agent_name }} has been down for more than 1 minute"
          
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{name=~"sutazai.*"} / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "High memory usage for {{ $labels.name }}"
          description: "Container {{ $labels.name }} is using >90% of memory limit"
          
      - alert: OllamaOverload
        expr: rate(ollama_request_duration_seconds_count[5m]) > 2
        for: 2m
        labels:
          severity: critical
          component: ollama
        annotations:
          summary: "Ollama is overloaded"
          description: "Ollama is processing more than 2 requests per second"
          
      - alert: QueueBacklog
        expr: rabbitmq_queue_messages{queue=~"agent.*"} > 100
        for: 5m
        labels:
          severity: warning
          component: rabbitmq
        annotations:
          summary: "Queue backlog for {{ $labels.queue }}"
          description: "Queue {{ $labels.queue }} has more than 100 pending messages"
```

---

## 8. DEPLOYMENT ARCHITECTURE

### 8.1 Docker Compose Structure

```yaml
# docker-compose.yml
version: '3.9'

x-common-variables: &common-variables
  CONSUL_HOST: consul
  REDIS_HOST: redis
  POSTGRES_HOST: postgres
  RABBITMQ_HOST: rabbitmq
  OLLAMA_HOST: ollama
  LOG_LEVEL: INFO

x-resource-limits: &resource-limits-small
  deploy:
    resources:
      limits:
        cpus: '0.5'
        memory: 256M
      reservations:
        cpus: '0.25'
        memory: 128M

x-resource-limits: &resource-limits-medium
  deploy:
    resources:
      limits:
        cpus: '1'
        memory: 512M
      reservations:
        cpus: '0.5'
        memory: 256M

services:
  # Infrastructure services
  postgres:
    image: postgres:15-alpine
    container_name: sutazai-postgres
    <<: *resource-limits-medium
    environment:
      <<: *common-variables
      POSTGRES_DB: sutazai
      POSTGRES_USER: sutazai
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "10000:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - data_network
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "sutazai"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: sutazai-redis
    <<: *resource-limits-small
    command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
    ports:
      - "10001:6379"
    volumes:
      - redis_data:/data
    networks:
      - data_network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Agent services (example for one agent)
  agent-senior-backend:
    build:
      context: ./agents/senior-backend-developer
      dockerfile: Dockerfile
    container_name: sutazai-senior-backend
    <<: *resource-limits-medium
    environment:
      <<: *common-variables
      AGENT_ID: ag-007
      AGENT_NAME: senior-backend-developer
      AGENT_PORT: 10306
    ports:
      - "10306:8080"
    depends_on:
      - consul
      - redis
      - rabbitmq
      - ollama
    networks:
      - agent_network
      - data_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
  ollama_models:
  rabbitmq_data:
  consul_data:

networks:
  sutazai_network:
    driver: bridge
  agent_network:
    driver: bridge
  data_network:
    driver: bridge
```

### 8.2 Kubernetes Deployment (Alternative)

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sutazai-agent-deployment
  namespace: sutazai
spec:
  replicas: 69
  selector:
    matchLabels:
      app: sutazai-agent
  template:
    metadata:
      labels:
        app: sutazai-agent
    spec:
      containers:
      - name: agent
        image: sutazai/agent:latest
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
          requests:
            memory: "256Mi"
            cpu: "250m"
        env:
        - name: AGENT_TYPE
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['agent-type']
        - name: CONSUL_HOST
          value: consul-service.sutazai.svc.cluster.local
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sutazai-agent-service
  namespace: sutazai
spec:
  selector:
    app: sutazai-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

---

## 9. CONFIGURATION MANAGEMENT

### 9.1 Environment Configuration

```bash
# .env.production
# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=sutazai
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=SECURE_PASSWORD_HERE

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=SECURE_REDIS_PASSWORD

# Neo4j Configuration
NEO4J_HOST=neo4j
NEO4J_BOLT_PORT=7687
NEO4J_HTTP_PORT=7474
NEO4J_PASSWORD=SECURE_NEO4J_PASSWORD

# Ollama Configuration
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
OLLAMA_NUM_PARALLEL=1
OLLAMA_NUM_THREADS=4
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=30s

# RabbitMQ Configuration
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_MANAGEMENT_PORT=15672
RABBITMQ_DEFAULT_USER=sutazai
RABBITMQ_DEFAULT_PASS=SECURE_RABBIT_PASSWORD

# Consul Configuration
CONSUL_HOST=consul
CONSUL_PORT=8500
CONSUL_ENCRYPT_KEY=SECURE_CONSUL_KEY

# Kong Configuration
KONG_HOST=kong
KONG_PROXY_PORT=8000
KONG_ADMIN_PORT=8001

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=SECURE_GRAFANA_PASSWORD

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=production
DEBUG=false
```

### 9.2 Agent Configuration

```yaml
# agents/config.yaml
agents:
  - id: ag-001
    name: agentzero-coordinator
    type: orchestration
    port: 10300
    memory_limit: 512MB
    cpu_limit: 1.0
    capabilities:
      - task_routing
      - agent_coordination
      - workflow_management
    dependencies:
      - consul
      - rabbitmq
      - redis
    environment:
      MODEL_TYPE: reasoning
      MAX_CONCURRENT_TASKS: 10
      TIMEOUT_SECONDS: 300
      
  - id: ag-007
    name: senior-backend-developer
    type: development
    port: 10306
    memory_limit: 512MB
    cpu_limit: 1.0
    capabilities:
      - api_design
      - code_review
      - database_optimization
      - microservices
    dependencies:
      - postgres
      - redis
      - ollama
    environment:
      MODEL_TYPE: coding
      CODE_LANGUAGES: python,javascript,go,rust
      MAX_CODE_LENGTH: 10000
```

---

## 10. PRODUCTION READINESS CHECKLIST

### 10.1 Pre-Deployment

- [ ] All environment variables configured
- [ ] Secrets stored securely (HashiCorp Vault / K8s Secrets)
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan documented

### 10.2 Infrastructure

- [ ] PostgreSQL replication configured
- [ ] Redis persistence enabled
- [ ] Neo4j backup scheduled
- [ ] Ollama models pre-downloaded
- [ ] RabbitMQ clustering setup
- [ ] Consul consensus established

### 10.3 Monitoring

- [ ] Prometheus scraping all targets
- [ ] Grafana dashboards imported
- [ ] Alert rules configured
- [ ] Log aggregation working
- [ ] Distributed tracing enabled
- [ ] Health checks passing

### 10.4 Security

- [ ] JWT authentication enabled
- [ ] TLS/SSL configured
- [ ] Network policies applied
- [ ] Secrets encrypted
- [ ] RBAC implemented
- [ ] Security scanning passed

### 10.5 Performance

- [ ] Load testing completed
- [ ] Resource limits verified
- [ ] Cache hit rates optimized
- [ ] Database indexes created
- [ ] Connection pools sized
- [ ] Rate limiting configured

---

**END OF TECHNICAL ARCHITECTURE DOCUMENTATION**

This document provides the complete technical blueprint for implementing the SutazAI distributed AI platform. Follow each section sequentially for successful deployment.