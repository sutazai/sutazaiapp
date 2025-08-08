# Perfect Jarvis System - Developer Training Guide

**Version:** 1.0  
**Last Updated:** August 8, 2025  
**Target Audience:** Software Developers, AI Engineers

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [API Integration Patterns](#api-integration-patterns)
3. [WebSocket Streaming Implementation](#websocket-streaming-implementation)
4. [Agent Development Guidelines](#agent-development-guidelines)
5. [Testing Strategies](#testing-strategies)
6. [Debugging Techniques](#debugging-techniques)
7. [Performance Profiling](#performance-profiling)
8. [Code Examples and Best Practices](#code-examples-and-best-practices)
9. [Contributing Guidelines](#contributing-guidelines)

## Development Environment Setup

### Prerequisites
- Docker 24.0+ and Docker Compose v2.0+
- Python 3.11+
- Node.js 18+ (for frontend components)
- Git

### Local Development Setup

#### 1. Clone and Setup Repository
```bash
git clone [repository-url]
cd sutazaiapp
```

#### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
# Key variables:
# - POSTGRES_PASSWORD=your_secure_password
# - REDIS_PASSWORD=your_redis_password
# - API_SECRET_KEY=your_api_secret
```

#### 3. Start Development Environment
```bash
# Create Docker network
docker network create sutazai-network 2>/dev/null

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

#### 4. Verify Installation
```bash
# Backend health check
curl http://localhost:10010/health

# Frontend access
open http://localhost:10011

# Ollama model check
curl http://localhost:10104/api/tags
```

### Development Tools Setup

#### IDE Configuration
**VS Code Extensions:**
- Python
- Docker
- REST Client
- Jupyter

**PyCharm Configuration:**
- Enable Docker integration
- Configure Python interpreter to use container
- Set up debugging with Docker

#### Database Tools
```bash
# PostgreSQL client
docker exec -it sutazai-postgres psql -U sutazai -d sutazai

# GUI tools (optional)
# - pgAdmin 4
# - DBeaver
# - Redis Desktop Manager
```

## API Integration Patterns

### REST API Architecture

#### Base Configuration
```python
# app/core/config.py
class Settings:
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Perfect Jarvis"
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    DATABASE_URL: str = "postgresql://sutazai:password@postgres:5432/sutazai"
```

#### FastAPI Application Structure
```python
# app/main.py
from fastapi import FastAPI, HTTPException
from app.api.v1 import api_router
from app.core.config import settings

app = FastAPI(title="Perfect Jarvis API", version="1.0.0")
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }
```

### Endpoint Development Patterns

#### 1. Chat Endpoint Implementation
```python
# app/api/v1/endpoints/chat.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.services.model_manager import ModelManager

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "tinyllama"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: datetime

@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    try:
        response = await model_manager.generate_response(
            prompt=request.message,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(
            response=response,
            model=request.model,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 2. Document Processing Endpoint
```python
# app/api/v1/endpoints/documents.py
from fastapi import APIRouter, UploadFile, File, Depends
from app.services.document_processor import DocumentProcessor

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_document_processor)
):
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(400, "Unsupported file type")
    
    content = await file.read()
    processed = await processor.process_document(content, file.filename)
    
    return {
        "filename": file.filename,
        "processed_chunks": len(processed.chunks),
        "document_id": processed.id
    }
```

#### 3. Agent Management Endpoints
```python
# app/api/v1/endpoints/agents.py
from fastapi import APIRouter, Depends
from app.services.agent_registry import AgentRegistry

router = APIRouter()

@router.get("/agents")
async def list_agents(registry: AgentRegistry = Depends(get_agent_registry)):
    return await registry.get_all_agents()

@router.post("/agents/{agent_id}/invoke")
async def invoke_agent(
    agent_id: str,
    task: AgentTask,
    registry: AgentRegistry = Depends(get_agent_registry)
):
    agent = await registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    
    result = await agent.execute(task)
    return {"result": result, "agent_id": agent_id}
```

## WebSocket Streaming Implementation

### Real-time Chat Streaming
```python
# app/api/v1/streaming.py
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message through LLM
            async for token in stream_llm_response(message_data["message"]):
                await manager.send_personal_message(
                    json.dumps({"token": token, "type": "stream"}),
                    websocket
                )
            
            # Send completion signal
            await manager.send_personal_message(
                json.dumps({"type": "complete"}),
                websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### Frontend WebSocket Integration
```javascript
// frontend/components/chat_streaming.js
class ChatStreaming {
    constructor(wsUrl) {
        this.wsUrl = wsUrl;
        this.ws = null;
        this.messageContainer = null;
    }
    
    connect() {
        this.ws = new WebSocket(this.wsUrl);
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'stream') {
                this.appendToken(data.token);
            } else if (data.type === 'complete') {
                this.finalizeMessage();
            }
        };
    }
    
    sendMessage(message) {
        const data = {
            message: message,
            timestamp: new Date().toISOString()
        };
        this.ws.send(JSON.stringify(data));
    }
    
    appendToken(token) {
        if (!this.messageContainer) {
            this.messageContainer = this.createMessageContainer();
        }
        this.messageContainer.textContent += token;
    }
}
```

## Agent Development Guidelines

### Agent Base Class
```python
# app/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

class BaseAgent(ABC):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"agent.{agent_id}")
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary function"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the agent is healthy and ready"""
        pass
    
    async def initialize(self):
        """Initialize agent resources"""
        self.logger.info(f"Initializing agent {self.agent_id}")
    
    async def cleanup(self):
        """Clean up agent resources"""
        self.logger.info(f"Cleaning up agent {self.agent_id}")
```

### Custom Agent Implementation
```python
# agents/text_processor/app.py
from app.agents.base_agent import BaseAgent
import asyncio

class TextProcessorAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("text_processor", config)
        self.model_client = None
    
    async def initialize(self):
        await super().initialize()
        # Initialize Ollama client
        self.model_client = OllamaClient(
            base_url=self.config.get("ollama_url", "http://ollama:11434")
        )
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        text = task.get("text", "")
        operation = task.get("operation", "summarize")
        
        if operation == "summarize":
            return await self._summarize_text(text)
        elif operation == "extract_entities":
            return await self._extract_entities(text)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _summarize_text(self, text: str) -> Dict[str, Any]:
        prompt = f"Summarize the following text:\n\n{text}"
        response = await self.model_client.generate(
            model="tinyllama",
            prompt=prompt,
            stream=False
        )
        
        return {
            "operation": "summarize",
            "summary": response["response"],
            "original_length": len(text),
            "summary_length": len(response["response"])
        }
    
    async def health_check(self) -> bool:
        try:
            if not self.model_client:
                return False
            
            # Test with simple prompt
            response = await self.model_client.generate(
                model="tinyllama",
                prompt="Test",
                stream=False
            )
            return bool(response.get("response"))
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
```

### Agent Flask Wrapper (for containerized agents)
```python
# agents/text_processor/flask_app.py
from flask import Flask, request, jsonify
import asyncio
from text_processor_agent import TextProcessorAgent

app = Flask(__name__)
agent = TextProcessorAgent(config={})

@app.route("/health", methods=["GET"])
async def health():
    healthy = await agent.health_check()
    return jsonify({
        "status": "healthy" if healthy else "unhealthy",
        "agent_id": agent.agent_id
    })

@app.route("/process", methods=["POST"])
async def process():
    try:
        task = request.json
        result = await agent.execute(task)
        return jsonify({
            "status": "success",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    asyncio.run(agent.initialize())
    app.run(host="0.0.0.0", port=8000)
```

## Testing Strategies

### Unit Testing with pytest
```python
# tests/test_chat_endpoint.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/api/v1/chat", json={
        "message": "Hello, how are you?",
        "model": "tinyllama"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "model" in data
    assert data["model"] == "tinyllama"

@pytest.mark.asyncio
async def test_agent_execution():
    from app.agents.text_processor import TextProcessorAgent
    
    agent = TextProcessorAgent(config={})
    await agent.initialize()
    
    result = await agent.execute({
        "text": "This is a test document.",
        "operation": "summarize"
    })
    
    assert "summary" in result
    assert result["operation"] == "summarize"
```

### Integration Testing
```python
# tests/test_integration.py
import pytest
import asyncio
from app.services.model_manager import ModelManager

@pytest.mark.integration
async def test_full_chat_flow():
    """Test complete chat flow from API to LLM"""
    
    # Test with real Ollama instance
    model_manager = ModelManager()
    
    response = await model_manager.generate_response(
        prompt="What is 2+2?",
        model="tinyllama"
    )
    
    assert response is not None
    assert len(response) > 0

@pytest.mark.integration  
def test_database_connection():
    """Test database connectivity"""
    from app.core.database import get_db
    
    db = next(get_db())
    result = db.execute("SELECT 1").fetchone()
    assert result[0] == 1
```

### Load Testing with pytest-benchmark
```python
# tests/test_performance.py
import pytest
from app.services.model_manager import ModelManager

@pytest.mark.benchmark
def test_llm_response_time(benchmark):
    """Benchmark LLM response times"""
    
    async def generate_response():
        manager = ModelManager()
        return await manager.generate_response(
            prompt="Hello world",
            model="tinyllama"
        )
    
    result = benchmark(asyncio.run, generate_response())
    assert result is not None
```

## Debugging Techniques

### Logging Configuration
```python
# app/core/logging.py
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("/app/logs/app.log")
        ]
    )

# Usage in modules
logger = logging.getLogger(__name__)
```

### Debug Endpoints
```python
# app/api/v1/debug.py (only in development)
from fastapi import APIRouter
import psutil
import asyncio

router = APIRouter()

@router.get("/debug/system")
async def system_info():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage('/')._asdict()
    }

@router.get("/debug/agents")
async def agent_status():
    # Check all agent health
    from app.services.agent_registry import get_agent_registry
    registry = get_agent_registry()
    
    status = {}
    for agent_id, agent in registry.agents.items():
        status[agent_id] = await agent.health_check()
    
    return status
```

### Database Query Debugging
```python
# Enable SQL logging in development
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### WebSocket Debugging
```python
# app/api/v1/streaming.py
import logging

logger = logging.getLogger(__name__)

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    client_ip = websocket.client.host
    logger.info(f"WebSocket connection from {client_ip}")
    
    try:
        await manager.connect(websocket)
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received message: {data[:100]}...")
            # Process message...
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        raise
```

## Performance Profiling

### Memory Profiling
```python
# app/utils/profiling.py
import tracemalloc
from functools import wraps

def memory_profile(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = await func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"{func.__name__}: Current memory: {current / 1024 / 1024:.1f} MB")
        print(f"{func.__name__}: Peak memory: {peak / 1024 / 1024:.1f} MB")
        tracemalloc.stop()
        return result
    return wrapper

# Usage
@memory_profile
async def process_large_document(document):
    # Processing logic...
    return result
```

### Response Time Monitoring
```python
# app/middleware/timing.py
import time
from fastapi import Request, Response

async def timing_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Database Query Optimization
```python
# app/services/analytics.py
from sqlalchemy import text
import time

async def analyze_slow_queries():
    """Analyze and log slow database queries"""
    
    query = text("""
        SELECT query, mean_time, calls, total_time
        FROM pg_stat_statements 
        WHERE mean_time > 100  -- queries slower than 100ms
        ORDER BY mean_time DESC 
        LIMIT 10
    """)
    
    result = await db.execute(query)
    for row in result:
        logger.warning(f"Slow query: {row.query[:100]}... "
                      f"Mean time: {row.mean_time:.2f}ms")
```

## Code Examples and Best Practices

### Error Handling Pattern
```python
# app/utils/exceptions.py
from fastapi import HTTPException
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def handle_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except ConnectionError as e:
            logger.error(f"Connection error in {func.__name__}: {e}")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    return wrapper
```

### Database Connection Management
```python
# app/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from contextlib import asynccontextmanager

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, pool_pre_ping=True)
    
    @asynccontextmanager
    async def get_session(self):
        async with AsyncSession(self.engine) as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
```

### Caching Pattern
```python
# app/services/cache.py
import asyncio
import json
from typing import Any, Optional
import redis.asyncio as redis

class CacheService:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def get(self, key: str) -> Optional[Any]:
        data = await self.redis.get(key)
        return json.loads(data) if data else None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        await self.redis.set(key, json.dumps(value), ex=ttl)
    
    async def delete(self, key: str):
        await self.redis.delete(key)

# Usage with decorator
def cache_result(ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            cached = await cache_service.get(cache_key)
            if cached:
                return cached
            
            result = await func(*args, **kwargs)
            await cache_service.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
```

## Contributing Guidelines

### Development Workflow
1. **Create feature branch:** `git checkout -b feature/your-feature-name`
2. **Implement changes** following code standards
3. **Write tests** for new functionality
4. **Run test suite:** `pytest`
5. **Create pull request** with description
6. **Code review** and iterate
7. **Merge to main** after approval

### Code Standards
- Follow PEP 8 for Python code
- Use type hints for all functions
- Write docstrings for all public methods
- Maximum line length: 88 characters
- Use async/await for I/O operations

### Commit Message Format
```
type(scope): short description

Longer description if needed

- Bullet points for changes
- Reference issue numbers: #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Hooks include:
# - Black (code formatting)
# - isort (import sorting)
# - flake8 (linting)
# - mypy (type checking)
```

### Documentation Standards
- Update docstrings when changing functions
- Add API documentation for new endpoints
- Include usage examples in docstrings
- Update training materials for major changes

---

**Note:** This guide reflects the current system implementation using TinyLlama via Ollama. Agent services are currently stub implementations ready for enhancement with real AI capabilities.