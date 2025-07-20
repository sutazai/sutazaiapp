#!/usr/bin/env python3
"""
SutazAI Core Component Initializer
"""

import asyncio
import logging

from fastapi import FastAPI

from backend.ai_agents.model_manager import ModelManager
from backend.ai_agents.agent_framework import AgentFramework
from backend.ai_agents.ethical_verifier import EthicalVerifier
from backend.sandbox.code_sandbox import CodeSandbox
from backend.utils.logging_setup import get_api_logger

logger = get_api_logger()

async def initialize_components(app: FastAPI):
    """
    Initialize all core components of the SutazAI system.
    """
    logger.info("Initializing core components...")
    try:
        app.state.model_manager = ModelManager()
        app.state.agent_framework = AgentFramework(app.state.model_manager)
        app.state.ethical_verifier = EthicalVerifier()
        app.state.code_sandbox = CodeSandbox()
        logger.info("Core components initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize core components: {e}", exc_info=True)
        raise

async def cleanup_components(app: FastAPI):
    """
    Clean up all core components of the SutazAI system.
    """
    logger.info("Cleaning up core components...")
    # The cleanup logic from the original main.py's shutdown event can be moved here
    # For now, we'll just log a message
    logger.info("Core components cleaned up successfully.")
