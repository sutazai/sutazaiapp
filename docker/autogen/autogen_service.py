from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import autogen
from autogen import AssistantAgent, UserProxyAgent
import asyncio
import json

app = FastAPI(title="AutoGen Multi-Agent Service")

class TaskRequest(BaseModel):
    task: str
    agents: List[str] = ["assistant", "user_proxy"]
    max_rounds: int = 10
    require_human_input: bool = False
    code_execution: bool = True

class AgentConfig(BaseModel):
    name: str
    system_message: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    status: str
    result: Any
    chat_history: List[Dict[str, Any]]
    execution_log: List[str]

# Global agent registry
agent_registry = {}

def get_llm_config():
    """Get LLM configuration for LiteLLM proxy"""
    return {
        "config_list": [{
            "model": "gpt-4",
            "api_base": "http://litellm:4000/v1",
            "api_key": "sk-local"
        }],
        "temperature": 0.7,
        "cache_seed": 42
    }

def create_agents(agent_configs: List[str]):
    """Create agents based on configuration"""
    agents = []
    
    for agent_name in agent_configs:
        if agent_name == "assistant":
            agent = AssistantAgent(
                name="assistant",
                llm_config=get_llm_config(),
                system_message="You are a helpful AI assistant. Help solve tasks step by step."
            )
            agents.append(agent)
        
        elif agent_name == "user_proxy":
            agent = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=10,
                is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                code_execution_config={
                    "work_dir": "/tmp/coding",
                    "use_docker": False
                }
            )
            agents.append(agent)
        
        elif agent_name == "coder":
            agent = AssistantAgent(
                name="coder",
                llm_config=get_llm_config(),
                system_message="You are an expert programmer. Write clean, efficient code."
            )
            agents.append(agent)
        
        elif agent_name == "critic":
            agent = AssistantAgent(
                name="critic",
                llm_config=get_llm_config(),
                system_message="You are a code reviewer. Analyze code for bugs, security issues, and improvements."
            )
            agents.append(agent)
        
        elif agent_name == "planner":
            agent = AssistantAgent(
                name="planner",
                llm_config=get_llm_config(),
                system_message="You are a project planner. Break down complex tasks into manageable steps."
            )
            agents.append(agent)
    
    return agents

@app.post("/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task using AutoGen agents"""
    try:
        # Create agents
        agents = create_agents(request.agents)
        if len(agents) < 2:
            raise HTTPException(status_code=400, detail="At least 2 agents required")
        
        # Initialize chat
        groupchat = autogen.GroupChat(
            agents=agents,
            messages=[],
            max_round=request.max_rounds,
            speaker_selection_method="round_robin"
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=get_llm_config()
        )
        
        # Execute task
        result = agents[0].initiate_chat(
            manager,
            message=request.task
        )
        
        # Extract chat history
        chat_history = []
        for msg in groupchat.messages:
            chat_history.append({
                "sender": msg.get("name", "unknown"),
                "content": msg.get("content", ""),
                "role": msg.get("role", "assistant")
            })
        
        return TaskResponse(
            status="completed",
            result=result.summary if hasattr(result, 'summary') else str(result),
            chat_history=chat_history,
            execution_log=[]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_agent")
async def create_custom_agent(config: AgentConfig):
    """Create a custom agent with specific configuration"""
    try:
        llm_config = config.llm_config or get_llm_config()
        
        agent = AssistantAgent(
            name=config.name,
            llm_config=llm_config,
            system_message=config.system_message or f"You are {config.name}"
        )
        
        agent_registry[config.name] = agent
        
        return {"status": "created", "agent": config.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi_agent_collaboration")
async def multi_agent_collaboration(task: str, agent_names: List[str]):
    """Run a collaborative task with multiple agents"""
    try:
        # Get or create agents
        agents = []
        for name in agent_names:
            if name in agent_registry:
                agents.append(agent_registry[name])
            else:
                # Create default agent
                agent = AssistantAgent(
                    name=name,
                    llm_config=get_llm_config(),
                    system_message=f"You are {name}, a helpful assistant."
                )
                agents.append(agent)
        
        # Create group chat
        groupchat = autogen.GroupChat(
            agents=agents,
            messages=[],
            max_round=20,
            speaker_selection_method="auto"
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=get_llm_config()
        )
        
        # Start collaboration
        result = agents[0].initiate_chat(
            manager,
            message=task
        )
        
        return {
            "status": "completed",
            "participants": agent_names,
            "result": str(result),
            "message_count": len(groupchat.messages)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    return {
        "default_agents": ["assistant", "user_proxy", "coder", "critic", "planner"],
        "custom_agents": list(agent_registry.keys())
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "autogen", "version": "2.0"}

@app.get("/")
async def root():
    return {
        "service": "AutoGen Multi-Agent Service",
        "endpoints": [
            "/execute - Execute task with agents",
            "/create_agent - Create custom agent",
            "/multi_agent_collaboration - Multi-agent collaboration",
            "/agents - List available agents",
            "/health - Health check"
        ]
    }