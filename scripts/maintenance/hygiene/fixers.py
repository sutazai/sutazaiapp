#!/usr/bin/env python3
"""
Hygiene violation auto-fixers
Extracted from hygiene_orchestrator.py for modularity
"""

import os
import re
import ast
import logging
from typing import List, Dict, Set, Optional
from .core import ViolationPattern

logger = logging.getLogger(__name__)

class ViolationFixer:
    """Base class for violation fixing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dry_run = config.get('dry_run', True)
    
    def fix(self, violation: ViolationPattern) -> bool:
        """Fix a violation"""
        raise NotImplementedError

class UnusedImportFixer(ViolationFixer):
    """Fixes unused import violations"""
    
    def fix(self, violation: ViolationPattern) -> bool:
        """Remove unused import from file"""
        if violation.pattern_type != "unused_import":
            return False
            
        try:
            with open(violation.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Remove the import line
            if 1 <= violation.line_number <= len(lines):
                if not self.dry_run:
                    lines.pop(violation.line_number - 1)
                    
                    with open(violation.file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                        
                logger.info(f"{'Would remove' if self.dry_run else 'Removed'} unused import at {violation.file_path}:{violation.line_number}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to fix unused import in {violation.file_path}: {e}")
            
        return False

class HardcodedValueFixer(ViolationFixer):
    """Suggests fixes for hardcoded values"""
    
    def fix(self, violation: ViolationPattern) -> bool:
        """Generate environment variable suggestion"""
        if violation.pattern_type != "hardcoded_value":
            return False
            
        # This fixer only suggests, doesn't auto-fix
        suggestions = self._generate_suggestions(violation)
        
        logger.info(f"Hardcoded value fix suggestions for {violation.file_path}:{violation.line_number}")
        for suggestion in suggestions:
            logger.info(f"  - {suggestion}")
            
        return True
    
    def _generate_suggestions(self, violation: ViolationPattern) -> List[str]:
        """Generate specific fix suggestions"""
        suggestions = []
        
        if "localhost" in violation.description.lower():
            suggestions.append("Replace with os.getenv('HOST', 'localhost')")
            suggestions.append("Add HOST environment variable to .env file")
            
        elif "password" in violation.description.lower():
            suggestions.append("Replace with os.getenv('PASSWORD')")
            suggestions.append("Store password in secure credential store")
            
        elif "api_key" in violation.description.lower():
            suggestions.append("Replace with os.getenv('API_KEY')")
            suggestions.append("Use proper secret management")
            
        return suggestions

class FixerRegistry:
    """Registry for all violation fixers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fixers = {
            "unused_import": UnusedImportFixer(config),
            "hardcoded_value": HardcodedValueFixer(config)
        }
    
    def fix_violation(self, violation: ViolationPattern) -> bool:
        """Fix a specific violation"""
        if not violation.auto_fixable:
            logger.debug(f"Violation {violation.pattern_type} is not auto-fixable")
            return False
            
        fixer = self.fixers.get(violation.pattern_type)
        if not fixer:
            logger.warning(f"No fixer available for {violation.pattern_type}")
            return False
            
        return fixer.fix(violation)
    
    def fix_all(self, violations: List[ViolationPattern]) -> Dict[str, int]:
        """Fix all auto-fixable violations"""
        results = {"fixed": 0, "failed": 0, "skipped": 0}
        
        for violation in violations:
            try:
                if self.fix_violation(violation):
                    results["fixed"] += 1
                else:
                    results["skipped"] += 1
            except Exception as e:
                logger.error(f"Failed to fix violation: {e}")
                results["failed"] += 1
                
        return results