#!/usr/bin/env python3
"""
Hygiene violation detectors
Extracted from hygiene_orchestrator.py for modularity
"""

import os
import ast
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional
from .core import ViolationPattern

logger = logging.getLogger(__name__)

class ViolationDetector:
    """Base class for violation detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.file_size_limit = config.get('file_size_limit', 500)
    
    def detect(self, file_path: str) -> List[ViolationPattern]:
        """Detect violations in a file"""
        raise NotImplementedError

class FileSizeDetector(ViolationDetector):
    """Detects files exceeding size limits"""
    
    def detect(self, file_path: str) -> List[ViolationPattern]:
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_count = len(lines)
                
            if line_count > self.file_size_limit:
                violation = ViolationPattern(
                    pattern_type="file_size",
                    severity="medium" if line_count < 1000 else "high",
                    confidence=1.0,
                    description=f"File has {line_count} lines, exceeds limit of {self.file_size_limit}",
                    file_path=file_path,
                    line_number=line_count,
                    suggested_fix="Break file into smaller modules",
                    auto_fixable=False
                )
                violations.append(violation)
                
        except Exception as e:
            logger.debug(f"Error checking file size for {file_path}: {e}")
            
        return violations

class UnusedImportDetector(ViolationDetector):
    """Detects unused imports in Python files"""
    
    def detect(self, file_path: str) -> List[ViolationPattern]:
        if not file_path.endswith('.py'):
            return []
            
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            imports = self._extract_imports(tree)
            used_names = self._extract_used_names(tree)
            
            for import_info in imports:
                if not self._is_import_used(import_info, used_names):
                    violation = ViolationPattern(
                        pattern_type="unused_import",
                        severity="low",
                        confidence=0.8,
                        description=f"Unused import: {import_info['name']}",
                        file_path=file_path,
                        line_number=import_info['line'],
                        suggested_fix=f"Remove import statement",
                        auto_fixable=True
                    )
                    violations.append(violation)
                    
        except Exception as e:
            logger.debug(f"Error analyzing imports in {file_path}: {e}")
            
        return violations
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract all import statements"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'name': alias.name,
                        'asname': alias.asname,
                        'line': node.lineno,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        'name': alias.name,
                        'asname': alias.asname,
                        'module': node.module,
                        'line': node.lineno,
                        'type': 'from_import'
                    })
                    
        return imports
    
    def _extract_used_names(self, tree: ast.AST) -> Set[str]:
        """Extract all used names in the code"""
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle chained attributes
                parts = []
                current = node
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    used_names.update(parts)
                    
        return used_names
    
    def _is_import_used(self, import_info: Dict, used_names: Set[str]) -> bool:
        """Check if an import is actually used"""
        name = import_info.get('asname') or import_info['name']
        
        # Special cases for common patterns
        if name in ['os', 'sys', 'json', 'logging']:
            return True  # Often used indirectly
            
        return name in used_names

class HardcodedValueDetector(ViolationDetector):
    """Detects hardcoded values that should be configurable"""
    
    HARDCODED_PATTERNS = [
        (r'localhost:\d+', 'localhost URL'),
        (r'127\.0\.0\.1:\d+', 'localhost IP'),
        (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded password'),
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'hardcoded API key'),
    ]
    
    def detect(self, file_path: str) -> List[ViolationPattern]:
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                for pattern, description in self.HARDCODED_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        violation = ViolationPattern(
                            pattern_type="hardcoded_value",
                            severity="medium",
                            confidence=0.9,
                            description=f"Hardcoded {description} found",
                            file_path=file_path,
                            line_number=line_num,
                            suggested_fix="Move to environment variable or config file",
                            auto_fixable=False
                        )
                        violations.append(violation)
                        
        except Exception as e:
            logger.debug(f"Error checking hardcoded values in {file_path}: {e}")
            
        return violations

class DetectorRegistry:
    """Registry for all violation detectors"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detectors = [
            FileSizeDetector(config),
            UnusedImportDetector(config),
            HardcodedValueDetector(config)
        ]
    
    def detect_all(self, file_path: str) -> List[ViolationPattern]:
        """Run all detectors on a file"""
        all_violations = []
        
        for detector in self.detectors:
            try:
                violations = detector.detect(file_path)
                all_violations.extend(violations)
            except Exception as e:
                logger.error(f"Detector {detector.__class__.__name__} failed on {file_path}: {e}")
                
        return all_violations