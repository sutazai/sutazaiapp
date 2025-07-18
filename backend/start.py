#!/usr/bin/env python3
"""
SutazAI Backend Startup Script
Simple startup script to test the backend
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    """Main startup function"""
    try:
        print("Starting SutazAI Backend...")
        
        # Test imports
        from core.config import settings
        from core.database import DatabaseManager
        from core.cache import CacheManager
        from core.security import SecurityManager
        from core.monitoring import MetricsCollector, HealthChecker
        from core.logging_config import logger
        
        print("✓ Core modules imported successfully")
        
        # Test service imports
        from services.agent_orchestrator import AgentOrchestrator
        from services.model_manager import ModelManager
        from services.vector_store import VectorStoreManager
        from services.document_processor import DocumentProcessor
        from services.code_generator import CodeGenerator
        from services.web_automation import WebAutomationManager
        from services.financial_analyzer import FinancialAnalyzer
        from services.workflow_engine import WorkflowEngine
        from services.backup_manager import BackupManager
        
        print("✓ Service modules imported successfully")
        
        # Test API imports
        from api.v1 import health, agents, models, documents, chat, workflows, admin
        
        print("✓ API modules imported successfully")
        
        # Test settings
        print(f"✓ Settings loaded: {settings.PROJECT_NAME}")
        
        # Test logger
        logger.info("Backend startup test completed successfully")
        
        print("\n🎉 SutazAI Backend is ready!")
        print(f"Configuration: {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
        print(f"Database: {settings.DATABASE_URL}")
        print(f"Redis: {settings.REDIS_URL}")
        print(f"Ollama: {settings.OLLAMA_URL}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)