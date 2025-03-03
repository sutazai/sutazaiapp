#!/usr/bin/env python3.11
"""Core Router Module

This module provides the core API endpoints for the SutazAI backend.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from loguru import logger

from ai_agents.agent_factory import AgentFactory

# Create router
core_router = APIRouter()

# Initialize agent factory
agent_factory = AgentFactory()


@core_router.get("/agents")
async def list_agents() -> Dict[str, List[str]]:
    """List available agent types.
    
    Returns:
        Dict containing list of available agent types
    """
    try:
        available_agents = agent_factory.get_available_agents()
        return {"agents": list(available_agents.keys())}
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list available agents",
        )


@core_router.post("/process-document")
async def process_document(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a document using the document processor agent.
    
    Args:
        request: Dictionary containing:
            - file_path: Path to the document to process
            - output_format: Desired output format
            
    Returns:
        Dict containing processing results
    """
    try:
        # Create document processor agent
        processor = agent_factory.create_agent(
            "document_processor",
            config={
                "input_dir": "data/input",
                "output_dir": "data/output",
                "temp_dir": "data/temp",
            },
        )
        
        # Initialize agent
        if not processor.initialize():
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize document processor",
            )
            
        # Process document
        result = processor.execute(request)
        
        # Check for errors
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Unknown error"),
            )
            
        return result
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@core_router.post("/execute-task")
async def execute_task(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a task using the specified agent.
    
    Args:
        request: Dictionary containing:
            - agent_type: Type of agent to use
            - task: Task details
            - config: Optional agent configuration
            
    Returns:
        Dict containing task execution results
    """
    try:
        agent_type = request.get("agent_type")
        if not agent_type:
            raise HTTPException(
                status_code=400,
                detail="agent_type is required",
            )
            
        task = request.get("task")
        if not task:
            raise HTTPException(
                status_code=400,
                detail="task is required",
            )
            
        # Create agent
        agent = agent_factory.create_agent(
            agent_type,
            config=request.get("config"),
        )
        
        # Initialize agent
        if not agent.initialize():
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize {agent_type} agent",
            )
            
        # Execute task
        result = agent.execute(task)
        
        # Check for errors
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Unknown error"),
            )
            
        return result
        
    except ValueError as e:
        # Agent type not found
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

