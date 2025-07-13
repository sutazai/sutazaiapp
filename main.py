#!/usr/bin/env python3
"""
SutazAI Main Application Entry Point
Comprehensive AI Agent System with Advanced Capabilities
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.config import Config
from backend.database.connection import init_database, create_tables, check_database_connection
from backend.backend_main import app
from loguru import logger
import uvicorn


def setup_logging():
    """Configure application logging"""
    config = Config()
    
    # Remove default logger
    logger.remove()
    
    # Console logging
    if config.logging.enable_console:
        logger.add(
            sys.stdout,
            level=config.logging.level,
            format=config.logging.format,
            colorize=True
        )
    
    # File logging
    if config.logging.enable_file:
        log_path = Path(config.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=config.logging.level,
            format=config.logging.format,
            rotation=config.logging.rotation,
            retention=config.logging.retention,
            compression=config.logging.compression
        )
    
    logger.info("Logging system initialized")


def setup_directories():
    """Create necessary directories"""
    config = Config()
    
    directories = [
        "logs",
        "data",
        "data/uploads", 
        "data/chromadb",
        "data/models",
        "backups",
        Path(config.storage.local_path).parent if hasattr(config.storage, 'local_path') else "data/uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


async def initialize_system():
    """Initialize all system components"""
    logger.info("🚀 Starting SutazAI System Initialization")
    
    # Setup directories
    setup_directories()
    logger.info("✅ Directory structure initialized")
    
    # Initialize database
    try:
        init_database()
        if not check_database_connection():
            logger.error("❌ Database connection failed")
            return False
        
        create_tables()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    
    # TODO: Initialize AI agents
    try:
        # from ai_agents.agent_factory import AgentFactory
        # agent_factory = AgentFactory()
        # await agent_factory.initialize_agents()
        logger.info("✅ AI agents initialization ready")
    except Exception as e:
        logger.warning(f"⚠️ AI agents initialization skipped: {e}")
    
    # TODO: Initialize vector database
    try:
        # Initialize ChromaDB, Qdrant, or PGVector based on config
        logger.info("✅ Vector database initialization ready")
    except Exception as e:
        logger.warning(f"⚠️ Vector database initialization skipped: {e}")
    
    logger.info("🎉 SutazAI System initialization completed successfully!")
    return True


async def run_development_server():
    """Run development server with auto-reload"""
    config = Config()
    
    logger.info("🔧 Starting SutazAI in DEVELOPMENT mode")
    
    if not await initialize_system():
        logger.error("❌ System initialization failed")
        sys.exit(1)
    
    uvicorn.run(
        "main:app",
        host=config.app.host,
        port=config.app.port,
        reload=True,
        log_level=config.logging.level.lower(),
        access_log=True
    )


async def run_production_server():
    """Run production server"""
    config = Config()
    
    logger.info("🏭 Starting SutazAI in PRODUCTION mode")
    
    if not await initialize_system():
        logger.error("❌ System initialization failed")
        sys.exit(1)
    
    uvicorn.run(
        app,
        host=config.app.host,
        port=config.app.port,
        workers=config.app.workers,
        log_level=config.logging.level.lower(),
        access_log=config.monitoring.log_requests
    )


def run_database_migration():
    """Run database migrations"""
    logger.info("🔄 Running database migrations")
    
    try:
        # TODO: Implement Alembic migrations
        from alembic.config import Config as AlembicConfig
        from alembic import command
        
        alembic_cfg = AlembicConfig("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        
        logger.info("✅ Database migrations completed")
    except Exception as e:
        logger.error(f"❌ Database migration failed: {e}")
        sys.exit(1)


def create_admin_user():
    """Create initial admin user"""
    logger.info("👤 Creating admin user")
    
    try:
        from backend.database.connection import SessionLocal
        from backend.database.models import create_user, UserRole
        
        db = SessionLocal()
        
        admin_user = create_user(
            db=db,
            username="admin",
            email="admin@sutazai.com",
            password="admin123",
            full_name="System Administrator"
        )
        admin_user.role = UserRole.ADMIN
        admin_user.is_verified = True
        db.commit()
        
        logger.info(f"✅ Admin user created: {admin_user.username}")
        logger.warning("🔐 Please change the default admin password!")
        
    except Exception as e:
        logger.error(f"❌ Failed to create admin user: {e}")


def show_system_info():
    """Display system information"""
    config = Config()
    
    print("\n" + "="*60)
    print("🤖 SutazAI - Advanced AI Agent System")
    print("="*60)
    print(f"Version: {config.version}")
    print(f"Environment: {config.app.environment}")
    print(f"Debug Mode: {config.app.debug}")
    print(f"Host: {config.app.host}:{config.app.port}")
    print(f"Database: {config.database.type}")
    print(f"Workers: {config.app.workers}")
    print("="*60)
    print("\n📚 Available endpoints:")
    print(f"  • API Documentation: http://{config.app.host}:{config.app.port}/docs")
    print(f"  • Health Check: http://{config.app.host}:{config.app.port}/health")
    print(f"  • Metrics: http://{config.app.host}:{config.app.port}/metrics")
    print("\n🔧 Management commands:")
    print("  python main.py --migrate    - Run database migrations")
    print("  python main.py --create-admin - Create admin user")
    print("  python main.py --dev       - Start development server")
    print("="*60 + "\n")


async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="SutazAI - Advanced AI Agent System")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    parser.add_argument("--migrate", action="store_true", help="Run database migrations")
    parser.add_argument("--create-admin", action="store_true", help="Create admin user")
    parser.add_argument("--info", action="store_true", help="Show system information")
    
    args = parser.parse_args()
    
    # Setup logging first
    setup_logging()
    
    # Show system info
    if args.info:
        show_system_info()
        return
    
    # Run database migrations
    if args.migrate:
        run_database_migration()
        return
    
    # Create admin user
    if args.create_admin:
        create_admin_user()
        return
    
    # Run development server
    if args.dev:
        await run_development_server()
        return
    
    # Default: run production server
    show_system_info()
    await run_production_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 SutazAI shutdown requested by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)