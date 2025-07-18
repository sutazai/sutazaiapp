#!/usr/bin/env python3
"""
SutazAI Financial Analyzer
Handles financial analysis and reporting
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """Analyzes financial data and generates reports"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.service_url = settings.FINROBOT_URL
        self._initialized = False
    
    async def initialize(self):
        """Initialize financial analyzer"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Financial Analyzer...")
            
            # Test connection
            response = await self.http_client.get(f"{self.service_url}/health")
            if response.status_code == 200:
                logger.info("Financial Analyzer service connected")
            else:
                logger.warning(f"Financial Analyzer service unhealthy: {response.status_code}")
            
            self._initialized = True
            logger.info("Financial Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Financial Analyzer: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown financial analyzer"""
        self._initialized = False
        logger.info("Financial Analyzer shutdown")