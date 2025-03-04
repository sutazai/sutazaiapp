#!/usr/bin/env python3
"""
Fix the test_sync_exception method in test_sync_manager_complete_coverage.py.
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
logger = logging.getLogger("fix_sync_exception")

def fix_sync_exception_test():
    """Fix the test_sync_exception method implementation."""
    file_path = Path("/opt/sutazaiapp/tests/test_sync_manager_complete_coverage.py")
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    with open(file_path, "r") as f:
        content = f.read()
    
    original_content = content
    
    # Find the test_sync_exception method and replace it with the correct implementation
    pattern = r"(\s+)def test_sync_exception\(.*?\):(.*?)(?=\n\s+def|\Z)"
    
    replacement = r"""\1def test_sync_exception(self, sync_manager):
\1    # Arrange
\1    sync_manager.exception_handler = MagicMock()
\1    test_exception = Exception("Test exception")
\1    
\1    # Act
\1    sync_manager.sync_exception(test_exception)
\1    
\1    # Assert
\1    sync_manager.exception_handler.assert_called_once_with(test_exception)"""
    
    # Replace the pattern
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"Fixed test_sync_exception method in {file_path}")
        return True
    else:
        logger.info(f"No changes needed in {file_path}")
        return False

def main():
    """Main function to fix the sync exception test."""
    success = fix_sync_exception_test()
    if success:
        logger.info("Successfully fixed sync exception test")
    else:
        logger.warning("No changes made or errors occurred")

if __name__ == "__main__":
    main() 