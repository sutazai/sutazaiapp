from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, List, Any, Optional
import httpx
import json
import os
import asyncio

app = FastAPI(title="LangChain Orchestrator")

# Custom LLM that uses Ollama through HTTP
class OllamaLLM(BaseChatModel):
    """Custom LLM that interfaces with Ollama"""
    
    base_url: str = "http://ollama:11434"
    model: str = "llama2:7b"
    
    async def _agenerate(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Async generation method"""
        # Convert messages to Ollama format
        prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return AIMessage(content=result.get("response", ""))
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
    
    def _generate(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Sync generation method"""
        return asyncio.run(self._agenerate(messages, **kwargs))

class AgentRequest(BaseModel):
    task: str
    context: Dict[str, Any] = {}
    tools: Optional[List[str]] = None
    memory: bool = True

class ChainRequest(BaseModel):
    prompt: str
    variables: Dict[str, Any] = {}
    chain_type: str = "simple"

class AgentResponse(BaseModel):
    result: str
    metadata: Dict[str, Any] = {}
    tools_used: List[str] = []

# Initialize tools
def create_tools():
    """Create available tools for agents"""
    
    # Code analysis tool
    async def analyze_code(code: str) -> str:
        """Analyze code using semgrep service"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://semgrep-service:8087/analyze/code",
                    json={"code": code, "language": "python"},
                    timeout=60.0
                )
                if response.status_code == 200:
                    result = response.json()
                    findings = result.get("findings", [])
                    if findings:
                        return f"Found {len(findings)} issues: " + "; ".join([f["message"] for f in findings[:3]])
                    else:
                        return "No issues found in the code"
                else:
                    return "Code analysis failed"
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    # Vector search tool
    async def search_knowledge(query: str) -> str:
        """Search knowledge base using vector database"""
        try:
            async with httpx.AsyncClient() as client:
                # First, get embedding (simplified - in production use proper embedding service)
                # For now, return mock result
                return f"Found relevant information about: {query}"
        except Exception as e:
            return f"Search error: {str(e)}"
    
    # Task planning tool
    def plan_task(description: str) -> str:
        """Create a plan for complex tasks"""
        steps = [
            "1. Analyze the requirements",
            "2. Break down into subtasks",
            "3. Identify dependencies",
            "4. Create execution order",
            "5. Define success criteria"
        ]
        return f"Task plan for '{description}':\n" + "\n".join(steps)
    
    # Model selection tool
    def select_model(task_type: str) -> str:
        """Select the best model for a given task"""
        model_mapping = {
            "code": "codellama:7b",
            "chat": "llama2:7b",
            "analysis": "deepseek-r1:8b",
            "complex": "qwen3:8b"
        }
        selected = model_mapping.get(task_type, "llama2:7b")
        return f"Recommended model for {task_type}: {selected}"
    
    tools = [
        Tool(
            name="Code_Analysis",
            func=lambda x: asyncio.run(analyze_code(x)),
            description="Analyze code for security vulnerabilities and quality issues"
        ),
        Tool(
            name="Knowledge_Search",
            func=lambda x: asyncio.run(search_knowledge(x)),
            description="Search the knowledge base for relevant information"
        ),
        Tool(
            name="Task_Planning",
            func=plan_task,
            description="Create an execution plan for complex tasks"
        ),
        Tool(
            name="Model_Selection",
            func=select_model,
            description="Choose the optimal model for a given task type"
        ),
    ]
    
    return tools

# Global tools
AVAILABLE_TOOLS = create_tools()

@app.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute an agent with tools and memory"""
    try:
        # Initialize LLM
        llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            model="llama2:7b"
        )
        
        # Select tools
        if request.tools:
            tools = [t for t in AVAILABLE_TOOLS if t.name in request.tools]
        else:
            tools = AVAILABLE_TOOLS
        
        # Initialize memory if requested
        memory = ConversationBufferMemory() if request.memory else None
        
        # Initialize agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
        
        # Execute task
        result = agent.run(request.task)
        
        # Track which tools were used (simplified)
        tools_used = [t.name for t in tools]
        
        return AgentResponse(
            result=result,
            metadata={
                "task": request.task,
                "context": request.context,
                "has_memory": request.memory
            },
            tools_used=tools_used
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chain/execute")
async def execute_chain(request: ChainRequest):
    """Execute a LangChain chain"""
    try:
        # Initialize LLM
        llm = OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            model="llama2:7b"
        )
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=list(request.variables.keys()),
            template=request.prompt
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Execute chain
        result = chain.run(**request.variables)
        
        return {
            "result": result,
            "chain_type": request.chain_type,
            "variables_used": list(request.variables.keys())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in AVAILABLE_TOOLS
        ]
    }

@app.post("/memory/store")
async def store_memory(conversation_id: str, message: str, role: str = "human"):
    """Store conversation in memory (simplified)"""
    # In production, this would interface with a proper memory store
    return {
        "status": "stored",
        "conversation_id": conversation_id,
        "message": message,
        "role": role
    }

@app.get("/memory/retrieve/{conversation_id}")
async def retrieve_memory(conversation_id: str, limit: int = 10):
    """Retrieve conversation history"""
    # In production, this would retrieve from proper memory store
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "total": 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama connection
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')}/api/tags",
                timeout=5.0
            )
            ollama_healthy = response.status_code == 200
    except:
        ollama_healthy = False
    
    return {
        "status": "healthy" if ollama_healthy else "degraded",
        "service": "langchain-orchestrator",
        "dependencies": {
            "ollama": ollama_healthy
        },
        "tools_available": len(AVAILABLE_TOOLS)
    }