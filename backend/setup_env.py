#!/usr/bin/env python3
"""
SutazAI Backend Environment Setup
Creates a .env file with default settings for development
"""

import os
from pathlib import Path

def create_env_file():
    """Create .env file with default settings"""
    env_path = Path(__file__).parent / ".env"
    
    env_content = """# SutazAI Backend Environment Configuration
# Development Settings
DEBUG_MODE=true
SECRET_KEY=sutazai_dev_secret_key_change_in_production_32_chars_minimum
JWT_SECRET=sutazai_dev_jwt_secret_change_in_production_32_chars_minimum

# Database Configuration
DATABASE_URL=postgresql://sutazai:sutazai_secure_password@localhost:5432/sutazai
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://sutazai:sutazai_mongo_password@localhost:27017/sutazai

# Vector Database URLs
CHROMADB_URL=http://localhost:8001
QDRANT_URL=http://localhost:6333
FAISS_URL=http://localhost:8088

# AI Model Services
OLLAMA_URL=http://localhost:11434

# AI Agent Services
AUTOGPT_URL=http://localhost:8010
LOCALAGI_URL=http://localhost:8011
TABBYML_URL=http://localhost:8012
AGENTZERO_URL=http://localhost:8013
BIGAGI_URL=http://localhost:8014

# Web Automation Services
BROWSER_USE_URL=http://localhost:8015
SKYVERN_URL=http://localhost:8016

# Document Processing
DOCUMIND_URL=http://localhost:8017

# Financial Analysis
FINROBOT_URL=http://localhost:8018

# Code Generation Services
GPT_ENGINEER_URL=http://localhost:8019
AIDER_URL=http://localhost:8020

# Framework Services
LANGFLOW_URL=http://localhost:8021
DIFY_URL=http://localhost:8022
PYTORCH_URL=http://localhost:8023
TENSORFLOW_URL=http://localhost:8024
JAX_URL=http://localhost:8025

# Specialized Services
AWESOME_CODE_AI_URL=http://localhost:8027

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=text

# Performance
MAX_CONCURRENT_REQUESTS=100
WORKER_COUNT=4
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"✓ Created .env file at {env_path}")
    print("✓ Environment configuration ready for development")

if __name__ == "__main__":
    create_env_file()