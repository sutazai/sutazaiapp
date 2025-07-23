#!/usr/bin/env python3

"""
Update all Docker service requirements files with secure versions
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_docker_requirements():
    """Update all Docker service requirements files"""
    project_root = Path("/opt/sutazaiapp")
    docker_dir = project_root / "docker"
    
    # Find all secure requirements files
    secure_files = list(project_root.rglob("*.secure.txt"))
    
    updated_count = 0
    
    for secure_file in secure_files:
        # Get the original requirements file path
        original_file = secure_file.with_suffix('')  # Remove .secure.txt -> .txt
        
        if original_file.exists():
            # Create backup
            backup_file = original_file.with_suffix('.txt.backup')
            if not backup_file.exists():
                shutil.copy2(original_file, backup_file)
                logger.info(f"Backup created: {backup_file}")
            
            # Replace with secure version
            shutil.copy2(secure_file, original_file)
            logger.info(f"Updated: {original_file}")
            updated_count += 1
            
    logger.info(f"Updated {updated_count} requirements files")
    return updated_count

if __name__ == "__main__":
    update_docker_requirements()