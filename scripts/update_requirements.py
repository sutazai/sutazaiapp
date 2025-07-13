#!/usr/bin/env python3
"""
Update requirements to latest stable versions
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/opt/sutazaiapp")

def get_latest_versions():
    """Get latest versions of all packages"""
    # Core packages for SutazAI
    packages = [
        'fastapi>=0.100.0',
        'uvicorn[standard]>=0.23.0',
        'pydantic>=2.0.0',
        'aiohttp>=3.8.5',
        'aiohttp-cors>=0.8.0',
        'redis>=4.6.0',
        'sqlalchemy>=2.0.0',
        'alembic>=1.11.0',
        'psycopg2-binary>=2.9.7',
        'python-multipart>=0.0.6',
        'python-jose[cryptography]>=3.3.0',
        'passlib[bcrypt]>=1.7.4',
        'python-dotenv>=1.0.0',
        'loguru>=0.7.0',
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.0',
        'pytest-cov>=4.1.0',
        'httpx>=0.24.0',
        'requests>=2.31.0',
        'jinja2>=3.1.2',
        'schedule>=1.2.0',
        'watchdog>=3.0.0'
    ]
    
    return packages

def create_updated_requirements():
    """Create updated requirements file"""
    packages = get_latest_versions()
    
    requirements_content = """# SutazAI Enterprise Requirements
# Updated to latest stable versions for production deployment

# Core Framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# HTTP and API
aiohttp>=3.8.5
aiohttp-cors>=0.8.0
httpx>=0.24.0
requests>=2.31.0

# Database and Cache
sqlalchemy>=2.0.0
alembic>=1.11.0
psycopg2-binary>=2.9.7
redis>=4.6.0

# Authentication and Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4

# Utilities
python-multipart>=0.0.6
python-dotenv>=1.0.0
loguru>=0.7.0
jinja2>=3.1.2
schedule>=1.2.0
watchdog>=3.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Development
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
"""
    
    req_file = PROJECT_ROOT / "requirements.txt"
    with open(req_file, 'w') as f:
        f.write(requirements_content)
    
    print("✅ Updated requirements.txt with latest stable versions")
    return req_file

def install_requirements():
    """Install updated requirements"""
    try:
        print("📦 Installing updated requirements...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def freeze_requirements():
    """Freeze current environment to requirements_frozen.txt"""
    try:
        print("🧊 Freezing current environment...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "freeze"
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.returncode == 0:
            frozen_file = PROJECT_ROOT / "requirements_frozen.txt"
            with open(frozen_file, 'w') as f:
                f.write(f"# Frozen requirements for SutazAI\n")
                f.write(f"# Generated automatically - exact versions\n\n")
                f.write(result.stdout)
            
            print(f"✅ Frozen requirements saved to {frozen_file}")
            return True
        else:
            print(f"❌ Freeze failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error freezing requirements: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Updating SutazAI requirements to latest stable versions...")
    
    # Create updated requirements
    req_file = create_updated_requirements()
    
    # Install requirements
    if install_requirements():
        # Freeze exact versions
        freeze_requirements()
        print("🎉 Requirements update complete!")
        return True
    else:
        print("❌ Requirements update failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)