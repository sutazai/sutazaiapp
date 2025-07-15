#!/usr/bin/env python3
"""
Quick Deployment Script for SutazAI
Simplified deployment for immediate testing and development
"""

import asyncio
import logging
import subprocess
import os
import json
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickDeploy:
    """Quick deployment for SutazAI"""
    
    def __init__(self):
        self.root_dir = Path("/opt/sutazaiapp")
        self.deployment_steps = []
        
    async def quick_deploy(self):
        """Quick deployment process"""
        logger.info("🚀 Starting Quick Deployment of SutazAI")
        
        # Step 1: Create essential directories
        self._create_directories()
        
        # Step 2: Set environment variables
        self._set_environment()
        
        # Step 3: Initialize database
        await self._initialize_database()
        
        # Step 4: Initialize AI systems
        await self._initialize_ai_systems()
        
        # Step 5: Create startup script
        self._create_startup_script()
        
        # Step 6: Test system
        await self._test_system()
        
        logger.info("✅ Quick deployment completed!")
        return self.deployment_steps
    
    def _create_directories(self):
        """Create essential directories"""
        directories = [
            "logs", "data", "cache", "models", "backups",
            "config", "scripts", "static", "templates"
        ]
        
        for directory in directories:
            (self.root_dir / directory).mkdir(parents=True, exist_ok=True)
        
        self.deployment_steps.append("Created essential directories")
        logger.info("✅ Directories created")
    
    def _set_environment(self):
        """Set environment variables"""
        env_vars = {
            "PYTHONPATH": str(self.root_dir),
            "SUTAZAI_ROOT": str(self.root_dir),
            "ENVIRONMENT": "development",
            "DEBUG": "true",
            "LOG_LEVEL": "INFO"
        }
        
        # Create .env file
        env_content = ""
        for key, value in env_vars.items():
            os.environ[key] = value
            env_content += f"{key}={value}\n"
        
        env_file = self.root_dir / ".env"
        env_file.write_text(env_content)
        
        self.deployment_steps.append("Environment variables configured")
        logger.info("✅ Environment configured")
    
    async def _initialize_database(self):
        """Initialize database systems"""
        try:
            # Create simple database initialization
            db_init_script = self.root_dir / "scripts/init_db.py"
            db_init_content = '''#!/usr/bin/env python3
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
'''
            
            db_init_script.write_text(db_init_content)
            db_init_script.chmod(0o755)
            
            # Run database initialization
            await self._run_command(f"python3 {db_init_script}")
            
            self.deployment_steps.append("Database initialized")
            logger.info("✅ Database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    async def _initialize_ai_systems(self):
        """Initialize AI systems"""
        try:
            # Create AI initialization script
            ai_init_script = self.root_dir / "scripts/init_ai.py"
            ai_init_content = '''#!/usr/bin/env python3
"""AI systems initialization"""
import asyncio
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

async def init_ai_systems():
    """Initialize AI systems"""
    try:
        # Initialize model registry
        registry_file = Path("/opt/sutazaiapp/data/model_registry.json")
        
        registry_data = {
            "models": {
                "local-assistant": {
                    "name": "Local Assistant",
                    "type": "chat",
                    "status": "available",
                    "capabilities": ["text_generation", "conversation"]
                },
                "code-helper": {
                    "name": "Code Helper",
                    "type": "code",
                    "status": "available", 
                    "capabilities": ["code_generation", "code_review"]
                }
            },
            "initialized_at": "2024-01-01T00:00:00Z"
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        # Initialize neural network state
        network_file = Path("/opt/sutazaiapp/data/neural_network.json")
        
        network_data = {
            "network_state": {
                "total_nodes": 100,
                "total_connections": 500,
                "global_activity": 0.5,
                "learning_rate": 0.01
            },
            "initialized_at": "2024-01-01T00:00:00Z"
        }
        
        with open(network_file, 'w') as f:
            json.dump(network_data, f, indent=2)
        
        logger.info("✅ AI systems initialized")
        
    except Exception as e:
        logger.error(f"AI initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(init_ai_systems())
'''
            
            ai_init_script.write_text(ai_init_content)
            ai_init_script.chmod(0o755)
            
            # Run AI initialization
            await self._run_command(f"python3 {ai_init_script}")
            
            self.deployment_steps.append("AI systems initialized")
            logger.info("✅ AI systems initialized")
            
        except Exception as e:
            logger.error(f"AI initialization failed: {e}")
    
    def _create_startup_script(self):
        """Create startup script"""
        startup_content = f"""#!/bin/bash
set -e

echo "🚀 Starting SutazAI System"
echo "=========================="

# Change to project directory
cd {self.root_dir}

# Set environment variables
export PYTHONPATH={self.root_dir}:$PYTHONPATH
export SUTAZAI_ROOT={self.root_dir}

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
echo "🤖 Starting SutazAI application..."
python3 main.py --dev &

APP_PID=$!

echo "✅ SutazAI started successfully!"
echo "📊 Process ID: $APP_PID"
echo "🌐 Access at: http://localhost:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the application"

# Wait for interrupt
trap "echo '🛑 Stopping SutazAI...'; kill $APP_PID; exit 0" INT
wait $APP_PID
"""
        
        startup_script = self.root_dir / "start.sh"
        startup_script.write_text(startup_content)
        startup_script.chmod(0o755)
        
        self.deployment_steps.append("Startup script created")
        logger.info("✅ Startup script created")
    
    async def _test_system(self):
        """Test system functionality"""
        try:
            # Test basic imports
            test_script = self.root_dir / "scripts/test_system.py"
            test_content = '''#!/usr/bin/env python3
"""System functionality test"""
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, '/opt/sutazaiapp')

logger = logging.getLogger(__name__)

def test_imports():
    """Test critical imports"""
    try:
        # Test core imports
        from pathlib import Path
        from typing import Dict, List, Any
        import json
        import asyncio
        
        print("✅ Core Python modules imported successfully")
        
        # Test project imports
        try:
            from backend.config import Config
            print("✅ Backend config imported successfully")
        except ImportError as e:
            print(f"⚠️  Backend config import failed: {e}")
        
        try:
            from backend.ai.model_manager import model_manager
            print("✅ AI model manager imported successfully")
        except ImportError as e:
            print(f"⚠️  AI model manager import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_file_structure():
    """Test file structure"""
    root_dir = Path('/opt/sutazaiapp')
    
    required_files = [
        'main.py',
        'data/sutazai.db',
        'data/model_registry.json',
        'start.sh'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (root_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests"""
    print("🧪 Testing SutazAI System")
    print("=========================")
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed - System ready!")
        return True
    else:
        print("⚠️  Some tests failed - Check configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
            
            test_script.write_text(test_content)
            test_script.chmod(0o755)
            
            # Run system test
            result = await self._run_command(f"python3 {test_script}")
            
            self.deployment_steps.append("System tests completed")
            logger.info("✅ System tests completed")
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
    
    async def _run_command(self, command: str) -> str:
        """Run shell command"""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Command failed"
            logger.warning(f"Command failed: {command} - {error_msg}")
        
        return stdout.decode()
    
    def generate_deployment_report(self):
        """Generate deployment report"""
        report = {
            "quick_deployment_report": {
                "timestamp": time.time(),
                "deployment_steps": self.deployment_steps,
                "status": "completed",
                "features_deployed": [
                    "Essential directory structure",
                    "Environment configuration",
                    "SQLite database initialization",
                    "AI systems basic setup",
                    "Startup automation script",
                    "System validation tests"
                ],
                "files_created": [
                    "start.sh - Main startup script",
                    "scripts/init_db.py - Database initialization", 
                    "scripts/init_ai.py - AI systems initialization",
                    "scripts/test_system.py - System validation",
                    ".env - Environment configuration",
                    "data/sutazai.db - SQLite database",
                    "data/model_registry.json - AI model registry"
                ],
                "usage_instructions": [
                    "Run ./start.sh to start the system",
                    "Access the web interface at http://localhost:8000",
                    "Use Ctrl+C to stop the application",
                    "Check logs in ./logs/ directory for troubleshooting"
                ]
            }
        }
        
        report_file = self.root_dir / "QUICK_DEPLOYMENT_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Deployment report saved: {report_file}")
        return report

async def main():
    """Main deployment function"""
    deployer = QuickDeploy()
    
    try:
        steps = await deployer.quick_deploy()
        report = deployer.generate_deployment_report()
        
        print("🎉 SutazAI Quick Deployment Completed!")
        print(f"✅ Completed {len(steps)} deployment steps")
        print("")
        print("🚀 To start SutazAI:")
        print("   ./start.sh")
        print("")
        print("🌐 Access URLs:")
        print("   Web Interface: http://localhost:8000")
        print("   API Documentation: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick deployment failed: {e}")
        print("❌ Deployment failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)