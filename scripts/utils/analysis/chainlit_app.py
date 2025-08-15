#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Chainlit App for SutazAI - Conversational AI Interface
"""

import chainlit as cl
import httpx
import os

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://sutazai-backend:8000")

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    await cl.Message(
        content="🤖 Welcome to SutazAI Multi-Agent System!\n\nI can help you with:\n- Code analysis and review\n- Task automation\n- Agent orchestration\n- Workflow design\n\nWhat would you like to do today?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages"""
    
    # Show typing indicator
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Route to appropriate agent based on message content
        response = await route_to_agent(message.content)
        
        # Update message with response
        msg.content = response
        await msg.update()
        
    except Exception as e:
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()

async def route_to_agent(message: str) -> str:
    """Route message to appropriate agent"""
    
    # Analyze message intent
    message_lower = message.lower()
    
    if "code" in message_lower or "review" in message_lower:
        agent = "code-generation-improver"
    elif "deploy" in message_lower:
        agent = "deployment-automation-master"
    elif "test" in message_lower:
        agent = "testing-qa-validator"
    elif "security" in message_lower:
        agent = "security-pentesting-specialist"
    else:
        agent = "senior-ai-engineer"
    
    # Call backend API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/agents/{agent}/execute",
                json={"task": message},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return f"**Agent:** {agent}\n\n**Result:**\n{result.get('result', 'Task completed')}"
            else:
                return f"Agent returned status {response.status_code}"
                
    except Exception as e:
        return f"Failed to contact agent: {str(e)}"

@cl.on_settings_update
async def setup_agent_settings(settings):
    """Handle settings updates"""
    logger.info(f"Settings updated: {settings}")

# Add custom actions
@cl.action_callback("analyze_code")
async def on_action(action):
    await cl.Message(content=f"Analyzing code...").send()
    # Implement code analysis

@cl.action_callback("create_workflow") 
async def on_workflow_action(action):
    await cl.Message(content=f"Creating workflow...").send()
    # Implement workflow creation