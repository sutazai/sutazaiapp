#!/usr/bin/env python3
"""
Requirements Cleanup Script - SutazAI System
Safely removes old requirements files after consolidation

Generated: 2025-08-07
Purpose: Clean up redundant requirements files while preserving critical ones
"""

import shutil
import logging
from pathlib import Path
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path("/opt/sutazaiapp")
BACKUP_DIR = BASE_DIR / "archive" / "requirements_consolidation_backup_20250807"

# Files to PRESERVE (critical to system operation)
PRESERVE_FILES = [
    "requirements.txt",           # New consolidated production
    "requirements-dev.txt",       # New consolidated development  
    "requirements-test.txt",      # New consolidated testing
    "pyproject.toml",            # Project configuration
]

# Files to REMOVE (redundant after consolidation)
REMOVE_FILES = [
    # Backend files - replaced by main requirements.txt
    "backend/requirements.txt",
    "backend/requirements.secure.txt",
    "backend/requirements-test.txt",
    "backend/requirements-dev.txt",
    "backend/requirements-minimal.txt",
    "backend/requirements-fast.txt",
    
    # Frontend files - replaced by main requirements.txt  
    "frontend/requirements.txt",
    "frontend/requirements.secure.txt",
    
    # Agent files - replaced by main requirements.txt
    "agents/base_requirements.txt", 
    "agents/ai_agent_orchestrator/requirements.txt",
    "agents/coordinator/requirements.txt",
    "agents/ollama_integration/requirements.txt",
    "agents/ai-agent-orchestrator/requirements.txt",
    
    # Docker base files - will update Dockerfiles to use main requirements
    "docker/base/base-requirements.txt",
    "docker/base/ai-requirements.txt", 
    "docker/base/monitoring-requirements.txt",
    "docker/base/gpu-requirements.txt",
    "docker/base/base-agent-requirements.txt",
    "docker/agent-base/agent-requirements.txt",
    
    # Jarvis duplicate files (opt/sutazaiapp/jarvis/* are duplicates)
    "opt/sutazaiapp/jarvis/backend/requirements.txt",
    "opt/sutazaiapp/jarvis/frontend/requirements.txt", 
    "opt/sutazaiapp/jarvis/agents/base_requirements.txt",
    "opt/sutazaiapp/jarvis/docker/base/base-requirements.txt",
    "opt/sutazaiapp/jarvis/docker/base/ai-requirements.txt",
    "opt/sutazaiapp/jarvis/docker/base/monitoring-requirements.txt",
    "opt/sutazaiapp/jarvis/docker/base/gpu-requirements.txt",
    "opt/sutazaiapp/jarvis/docker/base/base-agent-requirements.txt",
    "opt/sutazaiapp/jarvis/docker/agent-base/agent-requirements.txt",
]

def backup_file(file_path: Path) -> bool:
    """Backup a file to the backup directory"""
    try:
        if file_path.exists():
            backup_path = BACKUP_DIR / file_path.name
            # Handle naming conflicts
            counter = 1
            while backup_path.exists():
                backup_path = BACKUP_DIR / f"{file_path.stem}_{counter}{file_path.suffix}"
                counter += 1
            
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up: {file_path} -> {backup_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to backup {file_path}: {e}")
        return False
    return False

def remove_file(file_path: Path) -> bool:
    """Safely remove a file"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Removed: {file_path}")
            return True
        else:
            logger.warning(f"File not found (already removed?): {file_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to remove {file_path}: {e}")
        return False

def main():
    """Main cleanup function"""
    logger.info("Starting requirements files cleanup...")
    
    # Ensure backup directory exists
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    removed_count = 0
    backed_up_count = 0
    errors = []
    
    # Process files marked for removal
    for rel_path in REMOVE_FILES:
        file_path = BASE_DIR / rel_path
        
        # Skip if file doesn't exist
        if not file_path.exists():
            logger.info(f"Skipping (not found): {file_path}")
            continue
            
        # Backup before removal
        if backup_file(file_path):
            backed_up_count += 1
            
            # Remove the file
            if remove_file(file_path):
                removed_count += 1
            else:
                errors.append(f"Failed to remove: {file_path}")
        else:
            errors.append(f"Failed to backup: {file_path}")
    
    # Summary report
    logger.info(f"""
    CLEANUP SUMMARY:
    ===============
    Files backed up: {backed_up_count}
    Files removed: {removed_count}
    Errors: {len(errors)}
    
    Backup location: {BACKUP_DIR}
    
    Remaining requirements files:
    - requirements.txt (production)
    - requirements-dev.txt (development)
    - requirements-test.txt (testing)
    - pyproject.toml (project config)
    """)
    
    if errors:
        logger.error("Errors encountered:")
        for error in errors:
            logger.error(f"  - {error}")
    
    # List remaining requirements files
    remaining_files = list(BASE_DIR.rglob("*requirements*.txt"))
    logger.info(f"Remaining requirements files ({len(remaining_files)}):")
    for file in sorted(remaining_files):
        logger.info(f"  - {file.relative_to(BASE_DIR)}")
    
    logger.info("Cleanup complete!")

if __name__ == "__main__":
    main()