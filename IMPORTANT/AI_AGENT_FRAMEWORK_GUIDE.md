# AI Agent Framework Implementation Guide

## Executive Summary

This guide documents the AI agent frameworks available in the SutazAI system, their current deployment status, and implementation patterns. The system uses a hybrid approach combining deployed agent containers with available framework repositories.

## Deployment Status Overview

### Currently Deployed (Running in Docker)
- ✅ **AI Agent Orchestrator** - Port 8589
- ✅ **Multi-Agent Coordinator** - Port 8587
- ✅ **Hardware Resource Optimizer** - Port 8002
- ✅ **Resource Arbitration Agent** - Port 8588
- ✅ **Task Assignment Coordinator** - Port 8551
- ⚠️ **AgentGPT** - Available in compose but needs activation
- ⚠️ **AgentZero** - Available in compose but needs activation
- ⚠️ **Aider** - Available in compose but needs activation

### Available Framework Repositories (Ready for Integration)

#### Tier 1: Production-Ready Frameworks
1. **LangChain** - https://github.com/langchain-ai/langchain
2. **AutoGen** - https://github.com/ag2ai/ag2
3. **CrewAI** - https://github.com/crewAIInc/crewAI
4. **LlamaIndex** - https://github.com/run-llama/llama_index

#### Tier 2: Advanced Agent Systems
1. **AutoGPT** - https://github.com/Significant-Gravitas/AutoGPT
2. **Letta (MemGPT)** - https://github.com/mysuperai/letta
3. **LocalAGI** - https://github.com/mudler/LocalAGI
4. **BigAGI** - https://github.com/enricoros/big-agi
5. **PrivateGPT** - https://github.com/zylon-ai/private-gpt

#### Tier 3: Workflow & UI Tools
1. **Langflow** - https://github.com/langflow-ai/langflow
2. **FlowiseAI** - https://github.com/FlowiseAI/Flowise
3. **Dify** - https://github.com/langgenius/dify

## Core Agent Architecture

### Base Agent Pattern
All agents follow a standardized pattern for consistency:

```python
# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio
import logging

class BaseAgent(ABC):
    """Base class for all AI agents in the system"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.status = "initialized"
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return results"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return health status"""
        pass
    
    async def initialize(self) -> None:
        """Initialize agent resources"""
        self.status = "ready"
        
    async def shutdown(self) -> None:
        """Clean shutdown"""
        self.status = "stopped"
```

### Agent Service Template
```python
# agents/[agent-name]/app.py
from flask import Flask, jsonify, request
from base_agent import BaseAgent
import os

app = Flask(__name__)

class SpecificAgent(BaseAgent):
    async def process(self, input_data):
        # Agent-specific logic here
        result = await self.execute_task(input_data)
        return {"status": "success", "result": result}
    
    async def health_check(self):
        return {
            "status": "healthy",
            "agent": self.name,
            "uptime": self.get_uptime()
        }

agent = SpecificAgent(
    name=os.environ.get("AGENT_NAME", "specific-agent"),
    config={
        "model": "tinyllama",
        "max_tokens": 1000
    }
)

@app.route('/health')
def health():
    return jsonify(asyncio.run(agent.health_check()))

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    result = asyncio.run(agent.process(data))
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Framework Implementation Patterns

### 1. LangChain Integration

```python
# agents/langchain_agent.py
from langchain.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class LangChainAgent(BaseAgent):
    def __init__(self, name, config):
        super().__init__(name, config)
        
        # Initialize Ollama model
        self.llm = ChatOllama(
            base_url="http://ollama:11434",
            model="tinyllama",
            temperature=0.7
        )
        
        # Setup memory
        self.memory = ConversationBufferMemory()
        
        # Create chain
        self.prompt = PromptTemplate(
            input_variables=["input", "history"],
            template="""
            History: {history}
            Human: {input}
            Assistant: """
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )
    
    async def process(self, input_data):
        response = await self.chain.arun(input_data["message"])
        return {"response": response}
```

### 2. AutoGen Multi-Agent System

```python
# agents/autogen_coordinator.py
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

class AutoGenCoordinator(BaseAgent):
    def __init__(self, name, config):
        super().__init__(name, config)
        
        # Configure Ollama endpoint
        llm_config = {
            "base_url": "http://ollama:11434/v1",
            "model": "tinyllama",
            "api_key": "sk-111111"  # Dummy key for local
        }
        
        # Create agents
        self.assistant = AssistantAgent(
            name="assistant",
            llm_config=llm_config,
            system_message="You are a helpful AI assistant."
        )
        
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False}
        )
        
        # Setup group chat
        self.groupchat = GroupChat(
            agents=[self.assistant, self.user_proxy],
            messages=[],
            max_round=10
        )
        
        self.manager = GroupChatManager(
            groupchat=self.groupchat,
            llm_config=llm_config
        )
    
    async def process(self, input_data):
        # Initiate chat
        await self.user_proxy.initiate_chat(
            self.manager,
            message=input_data["task"]
        )
        
        # Get conversation history
        messages = self.groupchat.messages
        return {"conversation": messages}
```

### 3. CrewAI Team Implementation

```python
# agents/crewai_team.py
from crewai import Agent, Task, Crew, Process

class CrewAITeam(BaseAgent):
    def __init__(self, name, config):
        super().__init__(name, config)
        
        # Define agents with roles
        self.researcher = Agent(
            role='Research Analyst',
            goal='Gather and analyze information',
            backstory='Expert in research and data analysis',
            verbose=True,
            allow_delegation=False,
            llm="ollama/tinyllama"
        )
        
        self.writer = Agent(
            role='Content Writer',
            goal='Create comprehensive content',
            backstory='Skilled technical writer',
            verbose=True,
            allow_delegation=False,
            llm="ollama/tinyllama"
        )
        
        self.reviewer = Agent(
            role='Quality Reviewer',
            goal='Ensure content quality',
            backstory='Expert in quality assurance',
            verbose=True,
            allow_delegation=False,
            llm="ollama/tinyllama"
        )
    
    async def process(self, input_data):
        # Create tasks
        research_task = Task(
            description=f"Research: {input_data['topic']}",
            agent=self.researcher,
            expected_output="Research findings"
        )
        
        write_task = Task(
            description=f"Write content based on research",
            agent=self.writer,
            expected_output="Written content"
        )
        
        review_task = Task(
            description=f"Review and improve content",
            agent=self.reviewer,
            expected_output="Final reviewed content"
        )
        
        # Create crew
        crew = Crew(
            agents=[self.researcher, self.writer, self.reviewer],
            tasks=[research_task, write_task, review_task],
            process=Process.sequential
        )
        
        # Execute
        result = await crew.kickoff()
        return {"result": result}
```

### 4. LlamaIndex RAG Agent

```python
# agents/llamaindex_rag.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
import qdrant_client

class LlamaIndexRAG(BaseAgent):
    def __init__(self, name, config):
        super().__init__(name, config)
        
        # Connect to Qdrant
        self.qdrant_client = qdrant_client.QdrantClient(
            host="localhost",
            port=10101
        )
        
        # Setup vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="documents"
        )
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Initialize index
        self.index = None
    
    async def load_documents(self, directory):
        """Load documents into the index"""
        documents = SimpleDirectoryReader(directory).load_data()
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context
        )
    
    async def process(self, input_data):
        if not self.index:
            return {"error": "No documents loaded"}
        
        # Create query engine
        query_engine = self.index.as_query_engine()
        
        # Execute query
        response = await query_engine.aquery(input_data["query"])
        
        return {
            "answer": str(response),
            "sources": response.source_nodes
        }
```

## Agent Communication Patterns

### 1. Direct Communication
```python
# Direct agent-to-agent communication
async def direct_communication(agent1, agent2, message):
    # Agent1 processes and sends to Agent2
    result1 = await agent1.process(message)
    result2 = await agent2.process(result1)
    return result2
```

### 2. Message Queue Pattern (RabbitMQ)
```python
import aio_pika

async def queue_communication():
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(
        "amqp://guest:guest@rabbitmq:5672/"
    )
    
    channel = await connection.channel()
    
    # Declare queue
    queue = await channel.declare_queue('agent_tasks')
    
    # Publish message
    await channel.default_exchange.publish(
        aio_pika.Message(body=b"task_data"),
        routing_key='agent_tasks'
    )
```

### 3. Event-Driven Pattern
```python
from typing import Dict, List, Callable
import asyncio

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, data: Any):
        if event_type in self.subscribers:
            tasks = [handler(data) for handler in self.subscribers[event_type]]
            await asyncio.gather(*tasks)

# Usage
event_bus = EventBus()
event_bus.subscribe("task_completed", agent2.handle_completion)
await event_bus.publish("task_completed", result_data)
```

## Specialized Agent Implementations

### 1. Code Generation Agent (GPT-Engineer Style)
```python
class CodeGenerationAgent(BaseAgent):
    async def process(self, input_data):
        prompt = f"""
        Generate Python code for: {input_data['requirement']}
        
        Requirements:
        - Include proper error handling
        - Add comprehensive comments
        - Follow PEP 8 style guide
        
        Code:
        """
        
        response = await self.llm.generate(prompt)
        
        # Parse and validate code
        code = self.extract_code(response)
        validation = self.validate_code(code)
        
        return {
            "code": code,
            "validation": validation,
            "explanation": response
        }
```

### 2. Security Analysis Agent (Semgrep Integration)
```python
import subprocess
import json

class SecurityAnalysisAgent(BaseAgent):
    async def process(self, input_data):
        # Save code to temporary file
        with open('/tmp/code.py', 'w') as f:
            f.write(input_data['code'])
        
        # Run Semgrep
        result = subprocess.run(
            ['semgrep', '--config=auto', '--json', '/tmp/code.py'],
            capture_output=True,
            text=True
        )
        
        findings = json.loads(result.stdout)
        
        # Analyze with LLM
        analysis = await self.llm.generate(f"""
        Analyze these security findings: {findings}
        Provide recommendations for fixes.
        """)
        
        return {
            "findings": findings,
            "analysis": analysis,
            "risk_level": self.calculate_risk(findings)
        }
```

### 3. Document Processing Agent (Documind Style)
```python
from typing import List
import PyPDF2
import docx

class DocumentProcessingAgent(BaseAgent):
    async def process(self, input_data):
        file_path = input_data['file_path']
        file_type = input_data['file_type']
        
        # Extract text based on file type
        if file_type == 'pdf':
            text = self.extract_pdf(file_path)
        elif file_type == 'docx':
            text = self.extract_docx(file_path)
        else:
            text = self.extract_text(file_path)
        
        # Process with LLM
        summary = await self.llm.generate(f"Summarize: {text[:3000]}")
        
        # Extract entities
        entities = await self.extract_entities(text)
        
        # Generate embeddings
        embeddings = await self.generate_embeddings(text)
        
        return {
            "summary": summary,
            "entities": entities,
            "embeddings": embeddings,
            "word_count": len(text.split())
        }
```

## Agent Orchestration Patterns

### 1. Sequential Processing
```python
class SequentialOrchestrator:
    async def execute(self, agents: List[BaseAgent], initial_input):
        result = initial_input
        for agent in agents:
            result = await agent.process(result)
        return result
```

### 2. Parallel Processing
```python
class ParallelOrchestrator:
    async def execute(self, agents: List[BaseAgent], input_data):
        tasks = [agent.process(input_data) for agent in agents]
        results = await asyncio.gather(*tasks)
        return self.merge_results(results)
```

### 3. Conditional Routing
```python
class ConditionalOrchestrator:
    def __init__(self):
        self.routes = {}
    
    def add_route(self, condition: Callable, agent: BaseAgent):
        self.routes[condition] = agent
    
    async def execute(self, input_data):
        for condition, agent in self.routes.items():
            if condition(input_data):
                return await agent.process(input_data)
        return {"error": "No matching route"}
```

## Deployment Configuration

### Docker Compose Service Definition
```yaml
# docker-compose.agents.yml
services:
  langchain-agent:
    build:
      context: ./agents/langchain-agent
      dockerfile: Dockerfile
    container_name: sutazai-langchain-agent
    ports:
      - "11050:8080"
    environment:
      - OLLAMA_HOST=ollama:11434
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
    depends_on:
      - ollama
      - redis
      - postgres
    restart: unless-stopped
```

### Agent Configuration File
```yaml
# agents/config/agent-config.yml
agents:
  langchain_researcher:
    type: langchain
    model: tinyllama
    temperature: 0.7
    max_tokens: 2000
    tools:
      - web_search
      - calculator
      - python_repl
    
  autogen_team:
    type: autogen
    agents:
      - name: coder
        role: "Python Developer"
      - name: reviewer
        role: "Code Reviewer"
      - name: tester
        role: "QA Tester"
    max_rounds: 10
    
  crewai_research:
    type: crewai
    process: sequential
    agents:
      - researcher
      - analyst
      - writer
```

## Performance Optimization

### 1. Connection Pooling
```python
class AgentPool:
    def __init__(self, agent_class, size=10):
        self.pool = [agent_class() for _ in range(size)]
        self.available = asyncio.Queue()
        for agent in self.pool:
            self.available.put_nowait(agent)
    
    async def acquire(self):
        return await self.available.get()
    
    async def release(self, agent):
        await self.available.put(agent)
```

### 2. Result Caching
```python
from functools import lru_cache
import hashlib

class CachedAgent(BaseAgent):
    @lru_cache(maxsize=100)
    async def process_cached(self, input_hash):
        # Expensive processing
        return await self.process_internal(input_hash)
    
    async def process(self, input_data):
        # Create hash of input
        input_hash = hashlib.md5(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()
        
        return await self.process_cached(input_hash)
```

### 3. Batch Processing
```python
class BatchAgent(BaseAgent):
    async def process_batch(self, items: List[Dict]):
        # Process items in batches
        batch_size = 10
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            batch_results = await asyncio.gather(
                *[self.process(item) for item in batch]
            )
            results.extend(batch_results)
        
        return results
```

## Monitoring and Observability

### Agent Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
agent_requests = Counter(
    'agent_requests_total',
    'Total agent requests',
    ['agent_name', 'status']
)

agent_latency = Histogram(
    'agent_processing_seconds',
    'Agent processing latency',
    ['agent_name']
)

agent_active = Gauge(
    'agent_active_tasks',
    'Currently active agent tasks',
    ['agent_name']
)

# Use in agent
class MonitoredAgent(BaseAgent):
    async def process(self, input_data):
        agent_active.labels(agent_name=self.name).inc()
        
        with agent_latency.labels(agent_name=self.name).time():
            try:
                result = await self.process_internal(input_data)
                agent_requests.labels(
                    agent_name=self.name,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                agent_requests.labels(
                    agent_name=self.name,
                    status='error'
                ).inc()
                raise
            finally:
                agent_active.labels(agent_name=self.name).dec()
```

## Testing Framework

### Unit Testing
```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_agent_process():
    agent = TestAgent("test", {})
    agent.llm = AsyncMock()
    agent.llm.generate.return_value = "Test response"
    
    result = await agent.process({"input": "test"})
    
    assert result["status"] == "success"
    agent.llm.generate.assert_called_once()
```

### Integration Testing
```python
@pytest.mark.integration
async def test_agent_integration():
    # Start test containers
    async with TestEnvironment() as env:
        agent = LangChainAgent("test", {"model": "tinyllama"})
        
        result = await agent.process({
            "message": "Hello, world!"
        })
        
        assert "response" in result
        assert len(result["response"]) > 0
```

## Deployment Checklist

### Prerequisites
- [ ] Docker and Docker Compose installed
- [ ] Ollama service running with models loaded
- [ ] Core services (PostgreSQL, Redis, Neo4j) healthy
- [ ] RabbitMQ for message passing configured
- [ ] Network (`sutazai-network`) created

### Deployment Steps
1. Build agent images: `docker-compose -f docker-compose.agents.yml build`
2. Start agents: `docker-compose -f docker-compose.agents.yml up -d`
3. Verify health: `./scripts/check-agent-health.sh`
4. Test endpoints: `./scripts/test-agents.sh`
5. Monitor logs: `docker-compose logs -f [agent-name]`

### Production Considerations
- Enable authentication on all agent endpoints
- Configure resource limits in Docker Compose
- Setup monitoring and alerting
- Implement circuit breakers for resilience
- Configure horizontal scaling for high-load agents