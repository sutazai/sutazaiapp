#!/usr/bin/env python3
"""
Database Setup and Migration Script for SutazAI
Creates and initializes database tables and configuration
"""

import sys
import os
import sqlite3
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_sqlite_database():
    """Create SQLite database with basic tables"""
    db_path = project_root / "data" / "sutazai.db"
    db_path.parent.mkdir(exist_ok=True)
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat_sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR(255) UNIQUE NOT NULL,
                user_id INTEGER,
                model_name VARCHAR(255) DEFAULT 'llama3-chatqa',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Create chat_messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR(255) NOT NULL,
                message_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                model_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
            )
        """)
        
        # Create system_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_level VARCHAR(50) NOT NULL,
                module VARCHAR(255),
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create configurations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key VARCHAR(255) UNIQUE NOT NULL,
                config_value TEXT,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert default configurations
        default_configs = [
            ('ollama_host', 'http://localhost:11434', 'Ollama service host URL'),
            ('default_model', 'llama3-chatqa', 'Default AI model for chat'),
            ('max_chat_history', '100', 'Maximum chat messages to keep in memory'),
            ('log_level', 'INFO', 'Application logging level'),
            ('enable_chat_logging', 'true', 'Enable chat message logging to database')
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO configurations (config_key, config_value, description)
            VALUES (?, ?, ?)
        """, default_configs)
        
        conn.commit()
        logger.info(f"✅ SQLite database created successfully at {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database creation failed: {e}")
        return False
    finally:
        if conn:
            conn.close()

def create_cache_directories():
    """Create necessary cache and data directories"""
    directories = [
        project_root / "data",
        project_root / "logs", 
        project_root / "cache",
        project_root / "models" / "ollama",
        project_root / "temp",
        project_root / "run"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

def setup_environment():
    """Setup environment variables and configuration"""
    env_file = project_root / ".env"
    
    if not env_file.exists():
        env_content = """# SutazAI Environment Configuration
DATABASE_URL=sqlite:///data/sutazai.db
REDIS_URL=redis://localhost:6379/0
OLLAMA_HOST=http://localhost:11434
LOG_LEVEL=INFO
DEBUG=false
SECRET_KEY=your-secret-key-here-change-in-production
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        logger.info(f"📄 Created environment file: {env_file}")

def main():
    """Main setup function"""
    logger.info("🚀 Starting SutazAI database setup...")
    
    # Create directories
    create_cache_directories()
    
    # Setup environment
    setup_environment()
    
    # Create database
    if create_sqlite_database():
        logger.info("✅ Database setup completed successfully")
        return True
    else:
        logger.error("❌ Database setup failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)