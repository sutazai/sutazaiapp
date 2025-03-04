#!/usr/bin/env python3
"""
Fix indentation and duplicate decorators in test files.
"""

import os
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
logger = logging.getLogger("fix_indentation")

def fix_file(file_path):
    """Fix indentation and duplicate decorators in a file."""
    with open(file_path, "r") as f:
        content = f.read()
    
    # Fix duplicate pytest markers
    original_content = content
    
    # Remove duplicate @pytest.mark.asyncio decorators
    content = re.sub(r'@pytest\.mark\.asyncio\s+@pytest\.mark\.asyncio', '@pytest.mark.asyncio', content)
    
    # Fix indentation (4 spaces instead of inconsistent tabs/spaces)
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Replace tab characters with 4 spaces
        line_with_spaces = line.replace('\t', '    ')
        
        # Fix indentation within method bodies (especially after async def)
        if re.match(r'\s+async def test_', line_with_spaces):
            # Make sure async test methods are indented with 4 spaces (class level indentation)
            indentation = re.match(r'^(\s+)', line_with_spaces)
            if indentation:
                spaces = indentation.group(1)
                if len(spaces) != 4:
                    line_with_spaces = ' ' * 4 + line_with_spaces.lstrip()
        
        fixed_lines.append(line_with_spaces)
    
    fixed_content = '\n'.join(fixed_lines)
    
    if fixed_content != original_content:
        with open(file_path, "w") as f:
            f.write(fixed_content)
        logger.info(f"Fixed indentation and decorators in {file_path}")
        return True
    else:
        logger.info(f"No changes needed in {file_path}")
        return False

def main():
    """Main function to fix all test files."""
    # Path to the tests directory
    tests_dir = Path("/opt/sutazaiapp/tests")
    
    # Process all Python files in the tests directory
    fixed_files = 0
    total_files = 0
    
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py") and file.startswith("test_"):
                file_path = Path(root) / file
                total_files += 1
                
                if fix_file(file_path):
                    fixed_files += 1
    
    logger.info(f"Fixed {fixed_files} out of {total_files} test files")

if __name__ == "__main__":
    main() 