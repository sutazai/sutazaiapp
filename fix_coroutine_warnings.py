#!/usr/bin/env python3
"""
Fix unawaited coroutine warnings in agent_manager.py.
"""

import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_fixes.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fix_coroutines")

def fix_agent_manager():
    """Fix unawaited coroutine issues in agent_manager.py."""
    file_path = Path("/opt/sutazaiapp/core_system/orchestrator/agent_manager.py")
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    with open(file_path, "r") as f:
        content = f.read()
    
    original_content = content
    
    # Pattern for non-awaited coroutines: self.method_name(args) not in an await statement
    # and not assigned to a variable
    pattern = r'(?<!await\s)(?<!=\s)self\.(notify_agent_status|update_agent_status|notify_job_complete|notify_job_failed|notify_job_status)\('
    
    # Add await to these method calls
    content = re.sub(pattern, r'await self.\1(', content)
    
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"Fixed unawaited coroutines in {file_path}")
        return True
    else:
        logger.info(f"No changes needed in {file_path}")
        return False

def main():
    """Main function to fix coroutine warnings."""
    success = fix_agent_manager()
    if success:
        logger.info("Successfully fixed unawaited coroutine warnings")
    else:
        logger.warning("No changes made or errors occurred")

if __name__ == "__main__":
    main() 