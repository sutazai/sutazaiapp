#!/usr/bin/env python3
"""
SutazAI Document Processor
Handles document processing and analysis
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes and analyzes documents"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.service_url = settings.DOCUMIND_URL
        self._initialized = False
    
    async def initialize(self):
        """Initialize document processor"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Document Processor...")
            
            # Test connection
            response = await self.http_client.get(f"{self.service_url}/health")
            if response.status_code == 200:
                logger.info("Document Processor service connected")
            else:
                logger.warning(f"Document Processor service unhealthy: {response.status_code}")
            
            self._initialized = True
            logger.info("Document Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Document Processor: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown document processor"""
        self._initialized = False
        logger.info("Document Processor shutdown")
    
    async def process_document(self, document_path: str, document_type: str = "auto") -> Dict[str, Any]:
        """Process a document"""
        try:
            request_data = {
                "document_path": document_path,
                "document_type": document_type,
                "extract_text": True,
                "extract_metadata": True,
                "generate_embeddings": True
            }
            
            response = await self.http_client.post(
                f"{self.service_url}/process",
                json=request_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Document processing failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to process document {document_path}: {e}")
            raise
    
    async def extract_text(self, document_path: str) -> str:
        """Extract text from document"""
        try:
            request_data = {"document_path": document_path}
            
            response = await self.http_client.post(
                f"{self.service_url}/extract_text",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("text", "")
            else:
                raise Exception(f"Text extraction failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to extract text from {document_path}: {e}")
            raise
    
    async def analyze_document(self, document_path: str) -> Dict[str, Any]:
        """Analyze document content"""
        try:
            request_data = {"document_path": document_path}
            
            response = await self.http_client.post(
                f"{self.service_url}/analyze",
                json=request_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Document analysis failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to analyze document {document_path}: {e}")
            raise