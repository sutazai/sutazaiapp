#!/usr/bin/env python3
"""
AutoGen Agent Implementation for SutazAI Brain
Provides multi-agent coordination and task decomposition
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import autogen
from autogen.agentchat import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoGen Agent", version="1.0.0")


class AgentRequest(BaseModel):
    """Request model for agent execution"""
    input: str
    task_plan: List[Dict[str, Any]]
    context: Dict[str, Any]


class AgentResponse(BaseModel):
    """Response model for agent execution"""
    output: Any
    quality_score: float = 0.85
    metadata: Dict[str, Any] = {}


class AutoGenAgent:
    """AutoGen agent implementation"""
    
    def __init__(self):
        # Configure LLM
        self.llm_config = {
            "model": "ollama/tinyllama",
            "api_base": "http://sutazai-ollama:11434/v1",
            "api_type": "open_ai",
            "api_key": "ollama"  # Ollama doesn't need a real key
        }
        
        # Initialize agents
        self._init_agents()
        
    def _init_agents(self):
        """Initialize AutoGen agents"""
        # Planner agent
        self.planner = AssistantAgent(
            name="Planner",
            system_message="""You are a task planner. Break down complex tasks into steps.
            Analyze requirements and create actionable plans.""",
            llm_config=self.llm_config
        )
        
        # Executor agent
        self.executor = AssistantAgent(
            name="Executor",
            system_message="""You execute tasks according to plans.
            Implement solutions and handle technical details.""",
            llm_config=self.llm_config
        )
        
        # Reviewer agent
        self.reviewer = AssistantAgent(
            name="Reviewer",
            system_message="""You review and improve solutions.
            Check quality, suggest improvements, and ensure correctness.""",
            llm_config=self.llm_config
        )
        
        # User proxy (non-human)
        self.user_proxy = UserProxyAgent(
            name="TaskProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "use_docker": True,
                "timeout": 60,
                "work_dir": "/tmp/autogen"
            }
        )
        
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute task using AutoGen agents"""
        try:
            # Create group chat
            groupchat = GroupChat(
                agents=[self.user_proxy, self.planner, self.executor, self.reviewer],
                messages=[],
                max_round=10,
                speaker_selection_method="round_robin"
            )
            
            manager = GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)
            
            # Format the task
            task_message = f"""Task: {request.input}

Task Plan:
{self._format_task_plan(request.task_plan)}

Context:
- Available memories: {len(request.context.get('memories', []))}
- Resource constraints: {request.context.get('resources', {})}

Please complete this task step by step."""
            
            # Execute the task
            self.user_proxy.initiate_chat(
                manager,
                message=task_message
            )
            
            # Extract results
            chat_history = groupchat.messages
            final_output = self._extract_final_output(chat_history)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(chat_history, request)
            
            return AgentResponse(
                output=final_output,
                quality_score=quality_score,
                metadata={
                    "rounds": len(chat_history),
                    "agents_involved": list(set(msg.get("name", "unknown") for msg in chat_history))
                }
            )
            
        except Exception as e:
            logger.error(f"AutoGen execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _format_task_plan(self, task_plan: List[Dict[str, Any]]) -> str:
        """Format task plan for agents"""
        formatted = []
        for i, task in enumerate(task_plan, 1):
            formatted.append(f"{i}. {task.get('description', 'Unknown task')}")
            if 'complexity' in task:
                formatted.append(f"   Complexity: {task['complexity']}")
        return "\n".join(formatted)
    
    def _extract_final_output(self, chat_history: List[Dict[str, Any]]) -> str:
        """Extract final output from chat history"""
        # Look for the last substantial message from executor or reviewer
        for msg in reversed(chat_history):
            if msg.get("name") in ["Executor", "Reviewer"] and len(msg.get("content", "")) > 50:
                return msg["content"]
        
        # Fallback to last message
        if chat_history:
            return chat_history[-1].get("content", "No output generated")
        
        return "No output generated"
    
    def _calculate_quality_score(self, chat_history: List[Dict[str, Any]], request: AgentRequest) -> float:
        """Calculate quality score based on execution"""
        base_score = 0.7
        
        # Bonus for successful completion
        if any("completed" in msg.get("content", "").lower() for msg in chat_history):
            base_score += 0.1
        
        # Bonus for reviewer approval
        if any(msg.get("name") == "Reviewer" and "good" in msg.get("content", "").lower() for msg in chat_history):
            base_score += 0.1
        
        # Penalty for errors
        if any("error" in msg.get("content", "").lower() for msg in chat_history):
            base_score -= 0.1
        
        # Bonus for addressing all task plan items
        task_count = len(request.task_plan)
        if task_count > 0 and len(chat_history) >= task_count * 2:
            base_score += 0.05
        
        return max(0.0, min(1.0, base_score))


# Initialize agent
agent = AutoGenAgent()


@app.post("/execute")
async def execute(request: AgentRequest) -> AgentResponse:
    """Execute agent task"""
    return await agent.execute(request)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "autogen",
        "capabilities": [
            "multi-agent-coordination",
            "task-decomposition",
            "conversation"
        ]
    }


@app.get("/")
async def info():
    """Agent information"""
    return {
        "name": "AutoGen Agent",
        "version": "1.0.0",
        "description": "Multi-agent coordination and task decomposition",
        "llm_backend": "ollama"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)