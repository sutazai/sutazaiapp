#!/usr/bin/env python3

import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain Agent Server", version="1.0.0")

class TaskRequest(BaseModel):
    task: str
    agent_type: Optional[str] = "general"
    model: Optional[str] = "llama3.2:1b"

class LangChainAgentServer:
    def __init__(self):
        self.ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        self.agents = {}
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize different LangChain agents"""
        try:
            # Initialize Ollama LLM
            llm = Ollama(
                base_url=self.ollama_base,
                model="llama3.2:1b"
            )
            
            # Create tools
            search_tool = DuckDuckGoSearchRun()
            
            tools = [
                Tool(
                    name="search",
                    description="Search for information on the internet",
                    func=search_tool.run,
                ),
                Tool(
                    name="calculator",
                    description="Calculate mathematical expressions",
                    func=lambda x: str(eval(x)) if x.replace(".", "").replace("-", "").isdigit() else "Invalid expression",
                ),
            ]
            
            # Create different agent types
            self.agents = {
                "general": self.create_general_agent(llm, tools),
                "researcher": self.create_researcher_agent(llm, tools),
                "code_helper": self.create_code_helper_agent(llm, tools),
                "task_planner": self.create_task_planner_agent(llm, tools)
            }
            
            logger.info(f"Initialized {len(self.agents)} LangChain agents")
            
        except Exception as e:
            logger.error(f"Failed to setup agents: {e}")
            self.agents = {}
    
    def create_general_agent(self, llm, tools):
        """Create a general-purpose agent"""
        prompt = PromptTemplate.from_template("""
        You are a helpful AI assistant. Use the available tools to help answer questions and complete tasks.
        
        Tools available: {tools}
        
        Question: {input}
        
        Think step by step and use tools when needed.
        """)
        
        return LLMChain(llm=llm, prompt=prompt)
    
    def create_researcher_agent(self, llm, tools):
        """Create a research-focused agent"""
        prompt = PromptTemplate.from_template("""
        You are a research assistant. Use search tools to find accurate information.
        
        Tools available: {tools}
        
        Research query: {input}
        
        Provide well-researched and accurate information.
        """)
        
        return LLMChain(llm=llm, prompt=prompt)
    
    def create_code_helper_agent(self, llm, tools):
        """Create a code assistance agent"""
        prompt = PromptTemplate.from_template("""
        You are a code assistant. Help with programming questions, debugging, and code generation.
        
        Tools available: {tools}
        
        Code request: {input}
        
        Provide clear, working code examples with explanations.
        """)
        
        return LLMChain(llm=llm, prompt=prompt)
    
    def create_task_planner_agent(self, llm, tools):
        """Create a task planning agent"""
        prompt = PromptTemplate.from_template("""
        You are a task planning assistant. Break down complex tasks into manageable steps.
        
        Tools available: {tools}
        
        Task to plan: {input}
        
        Create a detailed, step-by-step plan.
        """)
        
        return LLMChain(llm=llm, prompt=prompt)
    
    async def execute_task(self, task: str, agent_type: str = "general") -> str:
        """Execute a task using the specified agent"""
        try:
            if agent_type not in self.agents:
                agent_type = "general"
            
            agent = self.agents[agent_type]
            
            # Execute the task
            result = await asyncio.to_thread(
                agent.run,
                input=task,
                tools=list(self.agents.keys())
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return f"Error executing task: {str(e)}"

# Global agent server instance
agent_server = LangChainAgentServer()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agents": len(agent_server.agents)}

@app.get("/agents")
async def list_agents():
    return {
        "available_agents": list(agent_server.agents.keys()),
        "total_agents": len(agent_server.agents)
    }

@app.post("/execute")
async def execute_task(request: TaskRequest):
    try:
        result = await agent_server.execute_task(
            task=request.task,
            agent_type=request.agent_type
        )
        return {
            "task": request.task,
            "agent_type": request.agent_type,
            "result": result,
            "status": "completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {
        "service": "LangChain Agent Server",
        "status": "running",
        "agents": len(agent_server.agents),
        "ollama_base": agent_server.ollama_base
    }

if __name__ == "__main__":
    port = int(os.getenv("LANGCHAIN_PORT", 8084))
    uvicorn.run(app, host="0.0.0.0", port=port)