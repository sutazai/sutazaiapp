#!/usr/bin/env python3
"""Database initialization script"""
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def init_database():
    """Initialize SQLite database for development"""
    db_path = Path("/opt/sutazaiapp/data/sutazai.db")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create basic tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

if __name__ == "__main__":
    init_database()
