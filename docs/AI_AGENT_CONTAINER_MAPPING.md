# AI Agent Container Mapping & Configuration

## Container Status Overview

### ✅ Already Running
1. **Ollama** - Model management server
2. **TabbyML** - Code completion
3. **LangFlow** - Visual workflow builder
4. **FlowiseAI** - Drag & drop AI flows
5. **Dify** - AI application platform

### ⚠️ Configured but Not Running
1. **AutoGPT** - Needs Dockerfile fix
2. **CrewAI** - Needs Dockerfile fix
3. **Letta** - Needs Dockerfile fix
4. **Aider** - Needs Dockerfile fix
5. **GPT-Engineer** - Needs Dockerfile fix
6. **LocalAGI** - Container exists
7. **N8N** - Configuration issue
8. **Backend-AGI** - Missing Dockerfile
9. **Frontend-AGI** - Missing Dockerfile

### 🔧 Dockerfiles Exist, Need Activation
1. **AgentGPT** - `/docker/agentgpt/`
2. **AutoGen** - `/docker/autogen/`
3. **BrowserUse** - `/docker/browser-use/`
4. **OpenDevin** - `/docker/opendevin/`
5. **PrivateGPT** - `/docker/privategpt/`
6. **LlamaIndex** - `/docker/llamaindex/`
7. **LangChain Agents** - `/docker/langchain-agents/`
8. **BigAGI** - `/docker/bigagi/`

### 📦 Need Full Integration

## Detailed Container Configurations

### 1. Deepseek-R1 Integration
```yaml
deepseek-r1:
  container_name: sutazai-deepseek-r1
  build:
    context: ./docker/deepseek-r1
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    MODEL_NAME: deepseek-r1:8b
    AGENT_TYPE: reasoning
  ports:
    - "8120:8080"
  depends_on:
    - ollama
  networks:
    - sutazai-network
```

### 2. Qwen3 Integration
```yaml
qwen3:
  container_name: sutazai-qwen3
  build:
    context: ./docker/qwen3
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    MODEL_NAME: qwen3:8b
    AGENT_TYPE: multimodal
  ports:
    - "8121:8080"
  networks:
    - sutazai-network
```

### 3. LiteLLM Proxy
```yaml
litellm:
  container_name: sutazai-litellm
  image: ghcr.io/berriai/litellm:main-latest
  environment:
    <<: *common-variables
    LITELLM_MASTER_KEY: ${LITELLM_KEY:-sk-local}
    LITELLM_SALT_KEY: ${LITELLM_SALT:-secret-salt}
    DATABASE_URL: ${DATABASE_URL}
  volumes:
    - ./config/litellm_config.yaml:/app/config.yaml
  ports:
    - "4000:4000"
  command: --config /app/config.yaml
  networks:
    - sutazai-network
```

### 4. LocalAI Integration
```yaml
localai:
  container_name: sutazai-localai
  image: localai/localai:latest-aio-cpu
  environment:
    <<: *common-variables
    THREADS: 4
    CONTEXT_SIZE: 2048
  volumes:
    - models_data:/models
  ports:
    - "8122:8080"
  networks:
    - sutazai-network
```

### 5. SWE-Agent
```yaml
swe-agent:
  container_name: sutazai-swe-agent
  build:
    context: ./docker/swe-agent
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: software_engineering
    MODEL_BACKEND: ollama
  volumes:
    - agent_workspaces:/workspace
    - ./backend:/app/target:rw
  ports:
    - "8123:8080"
  networks:
    - sutazai-network
```

### 6. Devika
```yaml
devika:
  container_name: sutazai-devika
  build:
    context: ./docker/devika
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: software_development
  ports:
    - "8124:8080"
  networks:
    - sutazai-network
```

### 7. GPT-Pilot
```yaml
gpt-pilot:
  container_name: sutazai-gpt-pilot
  build:
    context: ./docker/gpt-pilot
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: project_management
  ports:
    - "8125:8080"
  networks:
    - sutazai-network
```

### 8. MetaGPT
```yaml
metagpt:
  container_name: sutazai-metagpt
  build:
    context: ./docker/metagpt
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: software_company
  ports:
    - "8126:8080"
  networks:
    - sutazai-network
```

### 9. ChatDev
```yaml
chatdev:
  container_name: sutazai-chatdev
  build:
    context: ./docker/chatdev
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: collaborative_development
  ports:
    - "8127:8080"
  networks:
    - sutazai-network
```

### 10. BabyAGI
```yaml
babyagi:
  container_name: sutazai-babyagi
  build:
    context: ./docker/babyagi
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config, *vector-config]
    AGENT_TYPE: task_management
  ports:
    - "8128:8080"
  networks:
    - sutazai-network
```

### 11. SuperAGI
```yaml
superagi:
  container_name: sutazai-superagi
  build:
    context: ./docker/superagi
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config, *database-config]
    AGENT_TYPE: autonomous_agent
  ports:
    - "8129:8080"
  networks:
    - sutazai-network
```

### 12. AgentVerse
```yaml
agentverse:
  container_name: sutazai-agentverse
  build:
    context: ./docker/agentverse
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: multi_agent_simulation
  ports:
    - "8130:8080"
  networks:
    - sutazai-network
```

### 13. JARVIS
```yaml
jarvis:
  container_name: sutazai-jarvis
  build:
    context: ./docker/jarvis
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: personal_assistant
  ports:
    - "8131:8080"
  networks:
    - sutazai-network
```

### 14. Open-Assistant
```yaml
open-assistant:
  container_name: sutazai-open-assistant
  build:
    context: ./docker/open-assistant
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config, *database-config]
    AGENT_TYPE: conversational_ai
  ports:
    - "8132:8080"
  networks:
    - sutazai-network
```

### 15. Continue.dev
```yaml
continue-dev:
  container_name: sutazai-continue-dev
  build:
    context: ./docker/continue-dev
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: code_assistant
  ports:
    - "8133:8080"
  networks:
    - sutazai-network
```

### 16. Codeium
```yaml
codeium:
  container_name: sutazai-codeium
  build:
    context: ./docker/codeium
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config]
    AGENT_TYPE: code_completion
  ports:
    - "8134:8080"
  networks:
    - sutazai-network
```

### 17. AutoGPT-Forge
```yaml
autogpt-forge:
  container_name: sutazai-autogpt-forge
  build:
    context: ./docker/autogpt-forge
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config, *vector-config]
    AGENT_TYPE: autonomous_forge
  ports:
    - "8135:8080"
  networks:
    - sutazai-network
```

### 18. H2O.ai
```yaml
h2o-ai:
  container_name: sutazai-h2o-ai
  image: h2oai/h2o-3:latest
  environment:
    <<: *common-variables
  ports:
    - "8136:54321"
  networks:
    - sutazai-network
```

### 19. MindsDB
```yaml
mindsdb:
  container_name: sutazai-mindsdb
  image: mindsdb/mindsdb:latest
  environment:
    <<: [*common-variables, *database-config]
  ports:
    - "8137:47334"
    - "8138:47335"
  networks:
    - sutazai-network
```

### 20. Haystack
```yaml
haystack:
  container_name: sutazai-haystack
  build:
    context: ./docker/haystack
    dockerfile: Dockerfile
  environment:
    <<: [*common-variables, *ollama-config, *vector-config]
    AGENT_TYPE: document_qa
  ports:
    - "8139:8080"
  networks:
    - sutazai-network
```

## Integration Priority

### Phase 1 - Critical (Week 1)
1. Fix existing container issues
2. LiteLLM proxy for API compatibility
3. Deepseek-R1 & Qwen3 models
4. Activate existing Dockerfiles

### Phase 2 - Important (Week 2)
1. SWE-agent, Devika, GPT-Pilot
2. MetaGPT, ChatDev
3. BabyAGI, SuperAGI

### Phase 3 - Enhancement (Week 3)
1. AgentVerse, JARVIS
2. Continue.dev, Codeium
3. H2O.ai, MindsDB, Haystack

### Phase 4 - Advanced (Week 4)
1. Remaining specialized agents
2. Custom integrations
3. Performance optimization

## Common Integration Pattern

Each agent follows this structure:
```python
# agent_service.py
import os
from fastapi import FastAPI
import httpx

app = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": os.getenv("AGENT_TYPE")}

@app.post("/process")
async def process(request: dict):
    prompt = request.get("prompt", "")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
        )
    
    return {"response": response.json()["response"]}
```

## Monitoring & Management

All agents will be monitored through:
1. Prometheus metrics collection
2. Grafana dashboards
3. Health check endpoints
4. Centralized logging via Loki
5. Agent registry in backend-agi