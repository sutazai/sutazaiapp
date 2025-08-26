#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Post-Cleanup Validation Script
==============================

Validates that all Python files have correct syntax after unused import cleanup.
"""

import ast
import os
import sys
from pathlib import Path

def validate_python_files(root_path: str) -> dict:
    """Validate all Python files for syntax errors"""
    results = {
        'total_files': 0,
        'validated_files': 0,
        'syntax_errors': 0,
        'error_files': []
    }
    
    for root, dirs, files in os.walk(root_path):
        # Skip certain directories
        if any(skip in root for skip in ['__pycache__', '.git', 'venv', 'node_modules', 'backup']):
            continue
        
        for filename in files:
            if not filename.endswith('.py'):
                continue
                
            filepath = os.path.join(root, filename)
            results['total_files'] += 1
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Try to parse the AST
                ast.parse(content)
                results['validated_files'] += 1
                
            except SyntaxError as e:
                results['syntax_errors'] += 1
                results['error_files'].append({
                    'file': filepath,
                    'error': str(e),
                    'line': e.lineno
                })
            except Exception as e:
                results['syntax_errors'] += 1
                results['error_files'].append({
                    'file': filepath,
                    'error': f"Parse error: {str(e)}",
                    'line': None
                })
    
    return results

if __name__ == "__main__":
    logger.info("🔍 Validating Python files after import cleanup...")
    
    results = validate_python_files('/opt/sutazaiapp')
    
    logger.info(f"\n📊 VALIDATION RESULTS:")
    logger.info(f"  Total Python files: {results['total_files']}")
    logger.info(f"  Successfully validated: {results['validated_files']}")
    logger.error(f"  Syntax errors: {results['syntax_errors']}")
    
    if results['syntax_errors'] == 0:
        logger.info("\n✅ ALL FILES PASSED SYNTAX VALIDATION!")
        sys.exit(0)
    else:
        logger.error(f"\n❌ {results['syntax_errors']} files have syntax errors:")
        for error_info in results['error_files']:
            logger.error(f"  - {error_info['file']}: {error_info['error']}")
        sys.exit(1)