#!/usr/bin/env python3
"""
SutazAI System Cleanup and Organization Script
Cleans up redundant files, organizes structure, and prepares for automation
"""

import os
import shutil
import glob
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/sutazaiapp")

# Files to keep (essential for the system)
ESSENTIAL_FILES = {
    'main.py',
    'requirements_optimized.txt',
    'ENTERPRISE_LAUNCH_REPORT.md',
    'README.md',
    'setup.py'
}

# Directories to keep
ESSENTIAL_DIRS = {
    'backend',
    'web_ui', 
    'scripts',
    'bin',
    'data',
    'logs',
    'cache',
    'models',
    'temp',
    'run',
    'venv',
    'tests',
    'docs'
}

# Files/patterns to remove (redundant/temporary)
CLEANUP_PATTERNS = [
    '*.pyc',
    '__pycache__',
    '*.log.*',
    '*.broken',
    'fix_*.py',
    'fix_*.sh',
    'achieve_*.py',
    'analyze_*.py',
    'coverage*.py',
    'coverage*.json',
    'deploy_*.sh',
    'ensure_*.sh',
    'improve_*.sh',
    'master_*.sh',
    'run_*.sh',
    'requirements_*.txt',  # Keep only optimized version
    'TEST_*.md',
    'test_*.md',
    'cpu_*.sh',
    'memory_*.sh',
    'clean_*.sh',
    'pyrightconfig.json',
    'lang.json',
    'reorganize_*.py',
    'verify_*.py',
    'verify_*.sh',
    'full_automation_*.sh',
    'start_complete_*.sh',
    'restart_*.sh',
    'simple_web_server.py',
    'sutazai_cli.py',
    'sutazai_simple.sh',
    'setup_*.sh',
    'ssh_key_*.sh',
    'quick_access_*.txt',
    'test_summary.json',
    'create_admin.py'
]

def cleanup_redundant_files():
    """Remove redundant and temporary files"""
    logger.info("🧹 Starting cleanup of redundant files...")
    
    removed_count = 0
    
    for pattern in CLEANUP_PATTERNS:
        # Skip requirements_optimized.txt
        if pattern == 'requirements_*.txt':
            files_to_remove = glob.glob(str(PROJECT_ROOT / pattern))
            files_to_remove = [f for f in files_to_remove if not f.endswith('requirements_optimized.txt')]
        else:
            files_to_remove = glob.glob(str(PROJECT_ROOT / pattern))
        
        for file_path in files_to_remove:
            try:
                file_obj = Path(file_path)
                if file_obj.is_file():
                    os.remove(file_path)
                    logger.info(f"  ❌ Removed file: {file_obj.name}")
                    removed_count += 1
                elif file_obj.is_dir():
                    shutil.rmtree(file_path)
                    logger.info(f"  ❌ Removed directory: {file_obj.name}")
                    removed_count += 1
            except Exception as e:
                logger.warning(f"  ⚠️ Could not remove {file_path}: {e}")
    
    logger.info(f"✅ Cleanup complete: {removed_count} items removed")

def organize_directory_structure():
    """Ensure proper directory structure"""
    logger.info("📁 Organizing directory structure...")
    
    # Create essential directories if they don't exist
    for dir_name in ESSENTIAL_DIRS:
        dir_path = PROJECT_ROOT / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ Created directory: {dir_name}")
    
    # Move any remaining scripts to scripts directory
    scripts_dir = PROJECT_ROOT / "scripts"
    for py_file in PROJECT_ROOT.glob("*.py"):
        if py_file.name not in ESSENTIAL_FILES and py_file.name != "cleanup_system.py":
            try:
                target = scripts_dir / py_file.name
                if not target.exists():
                    shutil.move(str(py_file), str(target))
                    logger.info(f"  📦 Moved {py_file.name} to scripts/")
                else:
                    os.remove(str(py_file))
                    logger.info(f"  ❌ Removed duplicate: {py_file.name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not move {py_file.name}: {e}")

def clean_logs_directory():
    """Clean and organize logs directory"""
    logger.info("📋 Cleaning logs directory...")
    
    logs_dir = PROJECT_ROOT / "logs"
    if logs_dir.exists():
        # Remove old/temporary log files
        temp_logs = list(logs_dir.glob("*manual*")) + \
                   list(logs_dir.glob("*clean*")) + \
                   list(logs_dir.glob("*mock*")) + \
                   list(logs_dir.glob("*.log.*"))
        
        for log_file in temp_logs:
            try:
                os.remove(str(log_file))
                logger.info(f"  ❌ Removed temp log: {log_file.name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not remove {log_file.name}: {e}")

def create_gitignore():
    """Create comprehensive .gitignore file"""
    logger.info("📝 Creating .gitignore file...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Logs
logs/
*.log
*.log.*

# Runtime
run/
*.pid
*.lock

# Cache
cache/
temp/
*.tmp

# Database
*.db
*.sqlite3
data/*.db

# Models
models/ollama/*
!models/ollama/.gitkeep

# Environment
.env
.env.local
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.coverage
.pytest_cache/
htmlcov/

# Temporary files
*.broken
*_backup
*_old
*_temp
"""
    
    gitignore_path = PROJECT_ROOT / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    logger.info("✅ Created .gitignore file")

def create_directory_keepfiles():
    """Create .gitkeep files for empty essential directories"""
    logger.info("📌 Creating .gitkeep files...")
    
    keep_dirs = ['models/ollama', 'temp', 'cache', 'run']
    
    for dir_path in keep_dirs:
        full_path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        keep_file = full_path / ".gitkeep"
        if not keep_file.exists():
            keep_file.touch()
            logger.info(f"  ✅ Created .gitkeep in {dir_path}")

def main():
    """Main cleanup function"""
    logger.info("🚀 Starting SutazAI system cleanup and organization...")
    
    try:
        cleanup_redundant_files()
        organize_directory_structure()
        clean_logs_directory()
        create_gitignore()
        create_directory_keepfiles()
        
        logger.info("🎉 System cleanup and organization complete!")
        logger.info("📁 Clean directory structure ready for automation")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)