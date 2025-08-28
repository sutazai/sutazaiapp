"""
Create database tables for SutazAI Platform
Run this script to initialize the database schema
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
from app.core.database import Base
from app.models.user import User  # Import to register model

async def create_tables():
    """Create all database tables"""
    print(f"Connecting to database: {settings.DATABASE_URL}")
    
    # Create engine
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=True  # Show SQL statements
    )
    
    async with engine.begin() as conn:
        # Drop all tables (optional - comment out in production)
        # await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created successfully!")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(create_tables())