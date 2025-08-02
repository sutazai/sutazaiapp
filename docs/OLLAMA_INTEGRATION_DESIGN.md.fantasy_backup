# Ollama Integration Design for All AI Agents

## Overview

This document outlines the comprehensive design for integrating all 40+ AI agents with Ollama to ensure 100% local functionality without external API dependencies.

## Core Integration Architecture

### 1. Ollama Service Configuration
```yaml
# Already running at sutazai-ollama:11434
ollama:
  container_name: sutazai-ollama
  image: ollama/ollama:latest
  command: /bin/ollama serve
  ports:
    - "11434:11434"
  volumes:
    - ollama_data:/root/.ollama
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### 2. LiteLLM Proxy Layer
```yaml
# NEW - OpenAI API Compatibility
litellm:
  container_name: sutazai-litellm
  image: ghcr.io/berriai/litellm:main-latest
  ports:
    - "4000:4000"
  environment:
    LITELLM_MASTER_KEY: sk-local
    LITELLM_LOG: DEBUG
  volumes:
    - ./config/litellm_config.yaml:/app/config.yaml
  command: --config /app/config.yaml --detailed_debug
  depends_on:
    - ollama
```

### 3. LiteLLM Configuration
```yaml
# config/litellm_config.yaml
model_list:
  - model_name: tinyllama-8b
    litellm_params:
      model: ollama/tinyllama
      api_base: http://ollama:11434
      
  - model_name: qwen3-8b
    litellm_params:
      model: ollama/qwen3:8b
      api_base: http://ollama:11434
      
  - model_name: codellama-7b
    litellm_params:
      model: ollama/codellama:7b
      api_base: http://ollama:11434
      
  - model_name: llama3-2-3b
    litellm_params:
      model: ollama/llama3.2:3b
      api_base: http://ollama:11434

general_settings:
  master_key: sk-local
  database_url: postgresql://sutazai:sutazai_password@postgres:5432/sutazai
```

## Agent Integration Patterns

### Pattern 1: Direct Ollama Integration
For agents that can directly use Ollama API:

```python
# Base Ollama Agent Class
import os
import httpx
from typing import Optional, Dict, Any

class OllamaAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.default_model = os.getenv("MODEL_NAME", "llama3.2:3b")
        self.available_models = self._get_available_models()
    
    async def _get_available_models(self) -> list:
        """Get list of available models from Ollama"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error getting models: {e}")
        return []
    
    async def generate_response(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate response using Ollama"""
        model = model or self.default_model
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 40
                        }
                    }
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "No response generated")
                else:
                    return f"Error: HTTP {response.status_code}"
                    
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def chat_completion(self, messages: list, model: Optional[str] = None) -> Dict[str, Any]:
        """OpenAI-style chat completion using Ollama"""
        model = model or self.default_model
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        response = await self.generate_response(prompt, model)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "index": 0,
                "finish_reason": "stop"
            }],
            "model": model,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }
        }
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI messages format to prompt"""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
```

### Pattern 2: OpenAI API Compatibility via LiteLLM
For agents expecting OpenAI API format:

```python
# OpenAI-Compatible Agent Class
from openai import AsyncOpenAI

class OpenAICompatibleAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.client = AsyncOpenAI(
            api_key="sk-local",
            base_url=os.getenv("LITELLM_BASE_URL", "http://litellm:4000/v1")
        )
    
    async def chat_completion(self, messages: list, model: str = "tinyllama-8b") -> Dict[str, Any]:
        """Standard OpenAI chat completion"""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            return response.model_dump()
        except Exception as e:
            return {"error": str(e)}
```

### Pattern 3: Legacy API Integration
For agents with custom API requirements:

```python
# Legacy API Adapter
class LegacyAPIAdapter:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.ollama_agent = OllamaAgent(agent_name)
    
    async def legacy_api_call(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt legacy API calls to Ollama"""
        
        # Extract prompt from various legacy formats
        prompt = self._extract_prompt_from_payload(payload)
        
        # Generate response using Ollama
        response = await self.ollama_agent.generate_response(prompt)
        
        # Format response in expected legacy format
        return self._format_legacy_response(response, payload)
    
    def _extract_prompt_from_payload(self, payload: Dict[str, Any]) -> str:
        """Extract prompt from various legacy API formats"""
        # Handle different legacy formats
        if "prompt" in payload:
            return payload["prompt"]
        elif "input" in payload:
            return payload["input"]
        elif "query" in payload:
            return payload["query"]
        elif "message" in payload:
            return payload["message"]
        else:
            return str(payload)
    
    def _format_legacy_response(self, response: str, original_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Format response in expected legacy format"""
        return {
            "response": response,
            "status": "success",
            "model": self.ollama_agent.default_model,
            "agent": self.agent_name
        }
```

## Agent-Specific Configurations

### 1. AutoGPT Integration
```dockerfile
# docker/autogpt/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY autogpt_ollama_service.py .
COPY config.yaml .

EXPOSE 8080

CMD ["python", "autogpt_ollama_service.py"]
```

```python
# docker/autogpt/autogpt_ollama_service.py
from fastapi import FastAPI
from ollama_agent import OllamaAgent

app = FastAPI()
agent = OllamaAgent("autogpt")

@app.post("/agent/task")
async def execute_task(task: dict):
    prompt = f"""
    You are AutoGPT, an autonomous AI agent. Execute this task:
    
    Task: {task.get('description', '')}
    Goals: {task.get('goals', [])}
    
    Break down the task into steps and execute them autonomously.
    Provide a detailed execution plan and results.
    """
    
    response = await agent.generate_response(prompt)
    return {
        "task_id": task.get("id"),
        "status": "completed",
        "result": response,
        "agent": "autogpt"
    }
```

### 2. CrewAI Integration
```python
# docker/crewai/crewai_ollama_service.py
from fastapi import FastAPI
from ollama_agent import OllamaAgent
import json

app = FastAPI()
agent = OllamaAgent("crewai")

@app.post("/crew/execute")
async def execute_crew_task(crew_task: dict):
    agents = crew_task.get("agents", [])
    task = crew_task.get("task", "")
    
    results = []
    for crew_agent in agents:
        prompt = f"""
        You are {crew_agent.get('role', 'AI Agent')} in a CrewAI team.
        Your expertise: {crew_agent.get('expertise', 'general')}
        
        Team Task: {task}
        Your specific responsibility: {crew_agent.get('responsibility', 'contribute to the task')}
        
        Collaborate with the team to complete this task.
        Provide your specialized contribution.
        """
        
        response = await agent.generate_response(prompt)
        results.append({
            "agent": crew_agent.get('role'),
            "contribution": response
        })
    
    # Synthesize final result
    synthesis_prompt = f"""
    As the CrewAI coordinator, synthesize these team contributions into a final result:
    
    Task: {task}
    Team Contributions:
    {json.dumps(results, indent=2)}
    
    Provide a comprehensive final deliverable.
    """
    
    final_result = await agent.generate_response(synthesis_prompt)
    
    return {
        "task": task,
        "team_contributions": results,
        "final_result": final_result,
        "status": "completed"
    }
```

### 3. Letta (MemGPT) Integration
```python
# docker/letta/letta_ollama_service.py
from fastapi import FastAPI
from ollama_agent import OllamaAgent
import json

app = FastAPI()
agent = OllamaAgent("letta")

class LettaMemoryManager:
    def __init__(self):
        self.core_memory = {}
        self.archival_memory = []
        self.recall_memory = []
    
    def update_core_memory(self, key: str, value: str):
        self.core_memory[key] = value
    
    def add_to_archival(self, content: str):
        self.archival_memory.append(content)
    
    def recall_similar(self, query: str, limit: int = 5):
        # Simple similarity search (can be enhanced with embeddings)
        relevant = [mem for mem in self.archival_memory if query.lower() in mem.lower()]
        return relevant[:limit]

memory_manager = LettaMemoryManager()

@app.post("/letta/chat")
async def letta_chat(chat_request: dict):
    message = chat_request.get("message", "")
    user_id = chat_request.get("user_id", "default")
    
    # Recall relevant memories
    relevant_memories = memory_manager.recall_similar(message)
    
    prompt = f"""
    You are Letta (MemGPT), an AI with persistent memory and self-awareness.
    
    Core Memory: {json.dumps(memory_manager.core_memory, indent=2)}
    
    Relevant Past Conversations:
    {chr(10).join(relevant_memories)}
    
    Current Message: {message}
    
    Respond naturally while:
    1. Using your persistent memory
    2. Updating core memory if needed
    3. Storing important information for future recall
    4. Maintaining conversation continuity
    
    Format your response as:
    RESPONSE: [your response]
    MEMORY_UPDATE: [any memory updates needed]
    ARCHIVAL_ADD: [information to store for later]
    """
    
    response = await agent.generate_response(prompt)
    
    # Process memory updates (simplified)
    if "MEMORY_UPDATE:" in response:
        memory_update = response.split("MEMORY_UPDATE:")[1].split("ARCHIVAL_ADD:")[0].strip()
        # Parse and update memory (implementation needed)
    
    if "ARCHIVAL_ADD:" in response:
        archival_content = response.split("ARCHIVAL_ADD:")[1].strip()
        memory_manager.add_to_archival(archival_content)
    
    # Extract main response
    main_response = response.split("RESPONSE:")[1].split("MEMORY_UPDATE:")[0].strip()
    
    return {
        "response": main_response,
        "memory_updated": "MEMORY_UPDATE:" in response,
        "new_memories": "ARCHIVAL_ADD:" in response,
        "agent": "letta"
    }
```

## Environment Configuration Template

### Standard Agent Environment Variables
```yaml
x-ollama-agent-config: &ollama-agent-config
  # Ollama Configuration
  OLLAMA_BASE_URL: http://ollama:11434
  OLLAMA_API_KEY: local
  MODEL_NAME: ${DEFAULT_MODEL:-llama3.2:3b}
  
  # LiteLLM Configuration (for OpenAI compatibility)
  LITELLM_BASE_URL: http://litellm:4000/v1
  OPENAI_API_BASE: http://litellm:4000/v1
  OPENAI_API_KEY: sk-local
  
  # Model Selection
  REASONING_MODEL: tinyllama
  CODE_MODEL: codellama:7b
  CHAT_MODEL: llama3.2:3b
  MULTIMODAL_MODEL: qwen3:8b
  
  # Agent Configuration
  AGENT_NAME: ${AGENT_NAME}
  AGENT_TYPE: ${AGENT_TYPE}
  WORKSPACE_PATH: /workspace
  
  # Vector Database Integration
  CHROMADB_URL: http://chromadb:8000
  QDRANT_URL: http://qdrant:6333
  
  # Knowledge Graph
  NEO4J_URI: bolt://neo4j:7687
  NEO4J_USER: neo4j
  NEO4J_PASSWORD: ${NEO4J_PASSWORD}
  
  # Disable External APIs
  OPENAI_API_KEY: ""
  ANTHROPIC_API_KEY: ""
  GOOGLE_API_KEY: ""
  TOGETHER_API_KEY: ""
  REPLICATE_API_TOKEN: ""
```

## Model Distribution Strategy

### Primary Models by Use Case
1. **Reasoning Tasks**: tinyllama
2. **Code Generation**: codellama:7b
3. **General Chat**: llama3.2:3b
4. **Multimodal**: qwen3:8b
5. **Fast Responses**: llama3.2:1b

### Agent-Model Mapping
- **AutoGPT**: tinyllama (complex reasoning)
- **CrewAI**: llama3.2:3b (team coordination)
- **Aider**: codellama:7b (code assistance)
- **GPT-Engineer**: codellama:7b (software engineering)
- **Letta**: llama3.2:3b (conversational memory)
- **Research Agents**: tinyllama (analysis)
- **Code Completion**: codellama:7b (fast completion)

## Monitoring & Health Checks

### Agent Health Check Template
```python
# Common health check endpoint
@app.get("/health")
async def health_check():
    # Check Ollama connectivity
    ollama_status = await agent.check_ollama_connection()
    
    # Check model availability
    available_models = await agent._get_available_models()
    
    return {
        "status": "healthy" if ollama_status else "unhealthy",
        "agent": agent.agent_name,
        "ollama_connected": ollama_status,
        "available_models": available_models,
        "default_model": agent.default_model,
        "timestamp": datetime.utcnow().isoformat()
    }
```

## Success Metrics

1. **Connectivity**: 100% of agents connected to Ollama
2. **API Compatibility**: All OpenAI-expecting agents work via LiteLLM
3. **Model Availability**: All agents can access appropriate models
4. **Response Quality**: Comparable quality to external APIs
5. **Performance**: Response times under 30 seconds for most queries
6. **Reliability**: 99% uptime for all agent-Ollama connections