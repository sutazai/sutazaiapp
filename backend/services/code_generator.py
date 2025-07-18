#!/usr/bin/env python3
"""
SutazAI Code Generator
Handles code generation and programming assistance
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generates and analyzes code"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.gpt_engineer_url = settings.GPT_ENGINEER_URL
        self.aider_url = settings.AIDER_URL
        self.tabby_url = settings.TABBYML_URL
        self._initialized = False
    
    async def initialize(self):
        """Initialize code generator"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Code Generator...")
            
            # Test connections
            services = [
                ("GPT Engineer", self.gpt_engineer_url),
                ("Aider", self.aider_url),
                ("TabbyML", self.tabby_url)
            ]
            
            for name, url in services:
                try:
                    response = await self.http_client.get(f"{url}/health")
                    if response.status_code == 200:
                        logger.info(f"{name} service connected")
                    else:
                        logger.warning(f"{name} service unhealthy: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Cannot reach {name}: {e}")
            
            self._initialized = True
            logger.info("Code Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Code Generator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown code generator"""
        self._initialized = False
        logger.info("Code Generator shutdown")
    
    async def generate_code(self, prompt: str, language: str = "python", framework: str = None) -> Dict[str, Any]:
        """Generate code from prompt"""
        try:
            request_data = {
                "prompt": prompt,
                "language": language,
                "framework": framework
            }
            
            response = await self.http_client.post(
                f"{self.gpt_engineer_url}/generate",
                json=request_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Code generation failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            raise
    
    async def complete_code(self, code_context: str, cursor_position: int = None) -> Dict[str, Any]:
        """Complete code at cursor position"""
        try:
            request_data = {
                "code": code_context,
                "position": cursor_position
            }
            
            response = await self.http_client.post(
                f"{self.tabby_url}/complete",
                json=request_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Code completion failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to complete code: {e}")
            raise
    
    async def improve_code(self, code: str, improvement_request: str) -> Dict[str, Any]:
        """Improve existing code"""
        try:
            request_data = {
                "code": code,
                "improvement_request": improvement_request
            }
            
            response = await self.http_client.post(
                f"{self.aider_url}/improve",
                json=request_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Code improvement failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to improve code: {e}")
            raise