#!/usr/bin/env python3

import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from autogen import ConversableAgent, UserProxyAgent, AssistantAgent
    from autogen.agentchat.groupchat import GroupChat
    from autogen.agentchat.group_chat_manager import GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    logger.warning("AutoGen not available, running in Mock mode")
    AUTOGEN_AVAILABLE = False

app = FastAPI(title="AutoGen Agent Server", version="1.0.0")

class TaskRequest(BaseModel):
    task: str
    agent_type: Optional[str] = "assistant"
    max_rounds: Optional[int] = 10

class AutoGenAgentServer:
    def __init__(self):
        self.ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:10104")
        self.agents = {}
        self.group_chat = None
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize AutoGen agents"""
        try:
            if not AUTOGEN_AVAILABLE:
                logger.info("AutoGen not available, using Mock agents")
                self.agents = {
                    "assistant": "Mock Assistant Agent",
                    "user_proxy": "Mock User Proxy Agent",
                    "code_executor": "Mock Code Executor Agent",
                    "planner": "Mock Planner Agent"
                }
                return
            
            # Configure LLM settings for Ollama
            llm_config = {
                "config_list": [
                    {
                        "model": "tinyllama.2:1b",
                        "base_url": self.ollama_base,
                        "api_key": "dummy"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # Create different types of agents
            self.agents = {
                "assistant": AssistantAgent(
                    name="assistant",
                    llm_config=llm_config,
                    system_message="You are a helpful AI assistant."
                ),
                "user_proxy": UserProxyAgent(
                    name="user_proxy",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=10,
                    llm_config=llm_config,
                    system_message="You are a user proxy agent."
                ),
                "code_executor": AssistantAgent(
                    name="code_executor",
                    llm_config=llm_config,
                    system_message="You are a code execution specialist. You can write and execute code."
                ),
                "planner": AssistantAgent(
                    name="planner",
                    llm_config=llm_config,
                    system_message="You are a task planning specialist. You break down complex tasks into steps."
                )
            }
            
            # Create group chat for multi-agent conversations
            self.group_chat = GroupChat(
                agents=list(self.agents.values()),
                messages=[],
                max_round=10
            )
            
            logger.info(f"Initialized {len(self.agents)} AutoGen agents")
            
        except Exception as e:
            logger.error(f"Failed to setup agents: {e}")
            # Fallback to Mock agents
            self.agents = {
                "assistant": "Mock Assistant Agent (Error)",
                "user_proxy": "Mock User Proxy Agent (Error)",
                "code_executor": "Mock Code Executor Agent (Error)",
                "planner": "Mock Planner Agent (Error)"
            }
    
    async def execute_task(self, task: str, agent_type: str = "assistant", max_rounds: int = 10) -> str:
        """Execute a task using AutoGen agents"""
        try:
            if not AUTOGEN_AVAILABLE or not isinstance(self.agents.get(agent_type), (AssistantAgent, UserProxyAgent)):
                # Mock response for testing
                return f"AutoGen Agent ({agent_type}) processed task: {task[:100]}... [Mock Response]"
            
            agent = self.agents[agent_type]
            
            # Execute single agent task
            if agent_type == "user_proxy":
                result = await asyncio.to_thread(
                    agent.initiate_chat,
                    self.agents["assistant"],
                    message=task
                )
            else:
                result = await asyncio.to_thread(
                    agent.generate_reply,
                    messages=[{"content": task, "role": "user"}]
                )
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return f"AutoGen Agent ({agent_type}) encountered error: {str(e)}"
    
    async def execute_group_task(self, task: str, max_rounds: int = 10) -> str:
        """Execute a task using group chat"""
        try:
            if not AUTOGEN_AVAILABLE or not self.group_chat:
                return f"AutoGen Group Chat processed task: {task[:100]}... [Mock Response]"
            
            # Create group chat manager
            manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config=self.agents["assistant"].llm_config
            )
            
            # Execute group conversation
            result = await asyncio.to_thread(
                self.agents["user_proxy"].initiate_chat,
                manager,
                message=task
            )
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Group task execution failed: {e}")
            return f"AutoGen Group Chat encountered error: {str(e)}"

# Global agent server instance
agent_server = AutoGenAgentServer()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "agents": len(agent_server.agents),
        "autogen_available": AUTOGEN_AVAILABLE
    }

@app.get("/agents")
async def list_agents():
    return {
        "available_agents": list(agent_server.agents.keys()),
        "total_agents": len(agent_server.agents),
        "autogen_available": AUTOGEN_AVAILABLE
    }

@app.post("/execute")
async def execute_task(request: TaskRequest):
    try:
        result = await agent_server.execute_task(
            task=request.task,
            agent_type=request.agent_type,
            max_rounds=request.max_rounds
        )
        return {
            "task": request.task,
            "agent_type": request.agent_type,
            "result": result,
            "status": "completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group_execute")
async def execute_group_task(request: TaskRequest):
    try:
        result = await agent_server.execute_group_task(
            task=request.task,
            max_rounds=request.max_rounds
        )
        return {
            "task": request.task,
            "execution_type": "group_chat",
            "result": result,
            "status": "completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {
        "service": "AutoGen Agent Server",
        "status": "running",
        "agents": len(agent_server.agents),
        "autogen_available": AUTOGEN_AVAILABLE,
        "ollama_base": agent_server.ollama_base
    }

if __name__ == "__main__":
    port = int(os.getenv("AUTOGEN_PORT", 8085))
    uvicorn.run(app, host="0.0.0.0", port=port)