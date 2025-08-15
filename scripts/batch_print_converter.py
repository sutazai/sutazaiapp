#!/usr/bin/env python3
"""
EMERGENCY BATCH PRINT() STATEMENT CONVERTER
==========================================

This script converts all remaining logger.info() statements to proper structured logging
in compliance with Rule 8: Python Script Excellence requirements.

Usage: python batch_print_converter.py [directory]
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# Configure logging for this converter script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import mapping patterns
LOGGING_PATTERNS = [
    # Standard logger.info() statements
    (r'print\(f?"([^"]*?)"\)', r'logger.info(f"\1")'),
    (r'print\(f?\'([^\']*?)\'\)', r'logger.info(f\'\1\')'),
    
    # Print with variables
    (r'print\(f"([^"]*?)"\)', r'logger.info(f"\1")'),
    (r'print\(f\'([^\']*?)\'\)', r'logger.info(f\'\1\')'),
    
    # Error prints (convert to error level)
    (r'print\(f?"[Ee]rror[^"]*?"[^)]*\)', r'logger.error(f"Error: {match_content}")'),
    (r'print\(f?"[Ff]ailed[^"]*?"[^)]*\)', r'logger.error(f"Failed: {match_content}")'),
    
    # Debug prints (convert to debug level) 
    (r'print\(f?"[Dd]ebug[^"]*?"[^)]*\)', r'logger.debug(f"Debug: {match_content}")'),
    
    # Warning prints (convert to warning level)
    (r'print\(f?"[Ww]arning[^"]*?"[^)]*\)', r'logger.warning(f"Warning: {match_content}")'),
]

def has_logging_import(content: str) -> bool:
    """Check if file already has logging import"""
    return 'import logging' in content

def has_logger_configured(content: str) -> bool:
    """Check if file already has logger configured"""
    patterns = [
        r'logger = logging\.getLogger',
        r'self\.logger = logging\.getLogger',
        r'from.*logging_config.*import.*get_logger'
    ]
    return any(re.search(pattern, content) for pattern in patterns)

def add_logging_setup(content: str, file_path: Path) -> str:
    """Add logging import and configuration to file"""
    lines = content.split('\n')
    
    # Find the best place to insert logging import
    import_index = 0
    docstring_ended = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip shebang and docstrings
        if stripped.startswith('#!') or (stripped.startswith('"""') and not docstring_ended):
            continue
        if '"""' in stripped and not docstring_ended:
            docstring_ended = True
            continue
        
        # Found first import or other code
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_index = i
            break
        elif stripped and not stripped.startswith('#'):
            import_index = i
            break
    
    # Insert logging import and configuration
    if not has_logging_import(content):
        lines.insert(import_index, 'import logging')
        import_index += 1
    
    if not has_logger_configured(content):
        lines.insert(import_index, '')
        lines.insert(import_index + 1, 'logger = logging.getLogger(__name__)')
        
    return '\n'.join(lines)

def convert_print_statements(content: str) -> Tuple[str, int]:
    """Convert print statements to logging calls"""
    converted_count = 0
    
    # Simple pattern for most print statements
    def replace_print(match):
        nonlocal converted_count
        converted_count += 1
        
        original = match.group(0)
        
        # Determine appropriate log level based on content
        content_lower = original.lower()
        
        if any(word in content_lower for word in ['error', 'failed', 'exception', 'critical']):
            return original.replace('print(', 'logger.error(')
        elif any(word in content_lower for word in ['warning', 'warn']):
            return original.replace('print(', 'logger.warning(')
        elif any(word in content_lower for word in ['debug']):
            return original.replace('print(', 'logger.debug(')
        else:
            return original.replace('print(', 'logger.info(')
    
    # Pattern to match print statements
    print_pattern = r'print\([^)]*\)'
    content = re.sub(print_pattern, replace_print, content, flags=re.MULTILINE | re.DOTALL)
    
    return content, converted_count

def process_file(file_path: Path) -> Tuple[bool, int]:
    """Process a single Python file"""
    try:
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Check if file has print statements
        if 'print(' not in original_content:
            return True, 0
        
        # Add logging setup if needed
        content = add_logging_setup(original_content, file_path)
        
        # Convert print statements
        content, converted_count = convert_print_statements(content)
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ Converted {converted_count} print() statements in {file_path}")
        return True, converted_count
    
    except Exception as e:
        logger.error(f"❌ Error processing {file_path}: {e}")
        return False, 0

def process_directory(directory: Path) -> Dict[str, int]:
    """Process all Python files in a directory"""
    stats = {
        'files_processed': 0,
        'files_converted': 0,
        'total_conversions': 0,
        'errors': 0
    }
    
    # Find all Python files
    python_files = list(directory.rglob('*.py'))
    
    # Exclude virtual environments
    python_files = [f for f in python_files if '/venv/' not in str(f) and '/__pycache__/' not in str(f)]
    
    logger.info(f"Processing {len(python_files)} Python files in {directory}")
    
    for file_path in python_files:
        stats['files_processed'] += 1
        success, converted_count = process_file(file_path)
        
        if success:
            if converted_count > 0:
                stats['files_converted'] += 1
                stats['total_conversions'] += converted_count
        else:
            stats['errors'] += 1
    
    return stats

def main():
    """Main conversion process"""
    if len(sys.argv) < 2:
        directory = Path('/opt/sutazaiapp/scripts')
    else:
        directory = Path(sys.argv[1])
    
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        sys.exit(1)
    
    logger.info(f"🚀 Starting EMERGENCY PRINT() CONVERSION for {directory}")
    
    # Process directory
    stats = process_directory(directory)
    
    # Report results
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE - STATISTICS:")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Files converted: {stats['files_converted']}")
    logger.info(f"Total print() statements converted: {stats['total_conversions']}")
    logger.info(f"Errors encountered: {stats['errors']}")
    
    if stats['errors'] == 0 and stats['total_conversions'] > 0:
        logger.info("✅ EMERGENCY CONVERSION SUCCESSFUL - Rule 8 compliance restored!")
    elif stats['total_conversions'] == 0:
        logger.info("✅ No print() statements found - already compliant!")
    else:
        logger.warning(f"⚠️ Conversion completed with {stats['errors']} errors")

if __name__ == "__main__":
    main()