#!/usr/bin/env python3
"""
SutazAI AGI/ASI Complete Backend System
Production-ready FastAPI backend with all AI services integrated
"""

import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Core imports
from pydantic import BaseModel, Field
import uvicorn
import aioredis
import asyncpg
import httpx
import json
from motor.motor_asyncio import AsyncIOMotorClient

# Application imports
from core.config import settings
from core.database import DatabaseManager
from core.cache import CacheManager
from core.security import SecurityManager
from core.monitoring import MetricsCollector, HealthChecker
from core.logging_config import setup_logging

# Service imports
from services.agent_orchestrator import AgentOrchestrator
from services.model_manager import ModelManager
from services.vector_store import VectorStoreManager
from services.document_processor import DocumentProcessor
from services.code_generator import CodeGenerator
from services.web_automation import WebAutomationManager
from services.financial_analyzer import FinancialAnalyzer
from services.workflow_engine import WorkflowEngine
from services.backup_manager import BackupManager

# API routes
from api.v1 import agents, models, documents, chat, workflows, admin, health

# Initialize logging
logger = setup_logging()

class SutazAIApp:
    """Main SutazAI Application Class"""
    
    def __init__(self):
        self.app = None
        self.db_manager = None
        self.cache_manager = None
        self.security_manager = None
        self.metrics_collector = None
        self.health_checker = None
        self.agent_orchestrator = None
        self.model_manager = None
        self.vector_store_manager = None
        self.document_processor = None
        self.code_generator = None
        self.web_automation_manager = None
        self.financial_analyzer = None
        self.workflow_engine = None
        self.backup_manager = None
        self.http_client = None
        
    async def initialize_services(self):
        """Initialize all core services"""
        try:
            logger.info("Initializing SutazAI services...")
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )
            
            # Initialize core services
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            self.cache_manager = CacheManager()
            await self.cache_manager.initialize()
            
            self.security_manager = SecurityManager()
            self.metrics_collector = MetricsCollector()
            self.health_checker = HealthChecker()
            
            # Initialize AI services
            self.agent_orchestrator = AgentOrchestrator(self.http_client)
            await self.agent_orchestrator.initialize()
            
            self.model_manager = ModelManager(self.http_client)
            await self.model_manager.initialize()
            
            self.vector_store_manager = VectorStoreManager(self.http_client)
            await self.vector_store_manager.initialize()
            
            self.document_processor = DocumentProcessor(self.http_client)
            await self.document_processor.initialize()
            
            self.code_generator = CodeGenerator(self.http_client)
            await self.code_generator.initialize()
            
            self.web_automation_manager = WebAutomationManager(self.http_client)
            await self.web_automation_manager.initialize()
            
            self.financial_analyzer = FinancialAnalyzer(self.http_client)
            await self.financial_analyzer.initialize()
            
            self.workflow_engine = WorkflowEngine(
                self.agent_orchestrator,
                self.model_manager,
                self.vector_store_manager
            )
            await self.workflow_engine.initialize()
            
            self.backup_manager = BackupManager()
            await self.backup_manager.initialize()
            
            # Start background tasks
            asyncio.create_task(self.periodic_health_check())
            asyncio.create_task(self.periodic_metrics_collection())
            asyncio.create_task(self.periodic_model_optimization())
            
            logger.info("All SutazAI services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    async def shutdown_services(self):
        """Shutdown all services gracefully"""
        try:
            logger.info("Shutting down SutazAI services...")
            
            # Shutdown services in reverse order
            if self.backup_manager:
                await self.backup_manager.shutdown()
            
            if self.workflow_engine:
                await self.workflow_engine.shutdown()
            
            if self.financial_analyzer:
                await self.financial_analyzer.shutdown()
            
            if self.web_automation_manager:
                await self.web_automation_manager.shutdown()
            
            if self.code_generator:
                await self.code_generator.shutdown()
            
            if self.document_processor:
                await self.document_processor.shutdown()
            
            if self.vector_store_manager:
                await self.vector_store_manager.shutdown()
            
            if self.model_manager:
                await self.model_manager.shutdown()
            
            if self.agent_orchestrator:
                await self.agent_orchestrator.shutdown()
            
            if self.cache_manager:
                await self.cache_manager.shutdown()
            
            if self.db_manager:
                await self.db_manager.shutdown()
            
            if self.http_client:
                await self.http_client.aclose()
            
            logger.info("All SutazAI services shutdown successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def periodic_health_check(self):
        """Periodic health check for all services"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.health_checker:
                    health_status = await self.health_checker.check_all_services()
                    
                    # Log any unhealthy services
                    for service, status in health_status.items():
                        if status.get("status") != "healthy":
                            logger.warning(f"Service {service} is unhealthy: {status}")
                    
                    # Store health status in cache
                    if self.cache_manager:
                        await self.cache_manager.set(
                            "system:health_status",
                            health_status,
                            ttl=120
                        )
                        
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def periodic_metrics_collection(self):
        """Periodic metrics collection"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                if self.metrics_collector:
                    await self.metrics_collector.collect_system_metrics()
                    
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def periodic_model_optimization(self):
        """Periodic model optimization"""
        while True:
            try:
                await asyncio.sleep(3600)  # Optimize every hour
                
                if self.model_manager:
                    await self.model_manager.optimize_model_performance()
                    
            except Exception as e:
                logger.error(f"Model optimization error: {e}")

# Global app instance
sutazai_app = SutazAIApp()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await sutazai_app.initialize_services()
    yield
    # Shutdown
    await sutazai_app.shutdown_services()

# Create FastAPI app
app = FastAPI(
    title="SutazAI AGI/ASI System",
    description="Complete autonomous AI system with integrated agents and services",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    if not credentials:
        return None
    
    try:
        user = await sutazai_app.security_manager.verify_token(credentials.credentials)
        return user
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        return None

# Dependency injection
async def get_db():
    """Get database connection"""
    return sutazai_app.db_manager

async def get_cache():
    """Get cache connection"""
    return sutazai_app.cache_manager

async def get_agent_orchestrator():
    """Get agent orchestrator"""
    return sutazai_app.agent_orchestrator

async def get_model_manager():
    """Get model manager"""
    return sutazai_app.model_manager

async def get_vector_store_manager():
    """Get vector store manager"""
    return sutazai_app.vector_store_manager

async def get_document_processor():
    """Get document processor"""
    return sutazai_app.document_processor

async def get_code_generator():
    """Get code generator"""
    return sutazai_app.code_generator

async def get_web_automation_manager():
    """Get web automation manager"""
    return sutazai_app.web_automation_manager

async def get_financial_analyzer():
    """Get financial analyzer"""
    return sutazai_app.financial_analyzer

async def get_workflow_engine():
    """Get workflow engine"""
    return sutazai_app.workflow_engine

# Include API routes
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SutazAI AGI/ASI System",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "agents": "/api/v1/agents",
            "models": "/api/v1/models",
            "documents": "/api/v1/documents",
            "chat": "/api/v1/chat",
            "workflows": "/api/v1/workflows",
            "admin": "/api/v1/admin",
            "health": "/health",
            "docs": "/docs"
        }
    }

# System status endpoint
@app.get("/status")
async def system_status():
    """Get comprehensive system status"""
    try:
        # Get health status from cache
        health_status = await sutazai_app.cache_manager.get("system:health_status")
        
        # Get system metrics
        metrics = await sutazai_app.metrics_collector.get_current_metrics()
        
        # Get active agents
        active_agents = await sutazai_app.agent_orchestrator.get_active_agents_count()
        
        # Get loaded models
        loaded_models = await sutazai_app.model_manager.get_loaded_models_count()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "health": health_status,
            "metrics": metrics,
            "active_agents": active_agents,
            "loaded_models": loaded_models,
            "uptime": sutazai_app.metrics_collector.get_uptime()
        }
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    # Record metrics
    if sutazai_app.metrics_collector:
        await sutazai_app.metrics_collector.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=process_time
        )
    
    return response

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        access_log=True,
        reload=False
    )