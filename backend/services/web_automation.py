#!/usr/bin/env python3
"""
SutazAI Web Automation Manager
Handles web automation and browser interactions
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


class WebAutomationManager:
    """Manages web automation services"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.browser_use_url = settings.BROWSER_USE_URL
        self.skyvern_url = settings.SKYVERN_URL
        self._initialized = False
    
    async def initialize(self):
        """Initialize web automation manager"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Web Automation Manager...")
            
            # Test connections
            services = [
                ("Browser Use", self.browser_use_url),
                ("Skyvern", self.skyvern_url)
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
            logger.info("Web Automation Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Web Automation Manager: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown web automation manager"""
        self._initialized = False
        logger.info("Web Automation Manager shutdown")