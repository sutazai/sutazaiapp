#!/usr/bin/env python3
"""
Purpose: Check for script duplication and enforce reuse (Rule 4 enforcement)
Usage: python check-script-duplication.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Set
import ast
import re

class ScriptDuplicationChecker:
    """Checks for duplicate scripts and enforces reuse."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations = []
        self.warnings = []
        
    def get_function_signatures(self, filepath: Path) -> Set[str]:
        """Extract function signatures from Python files."""
        signatures = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Create a signature string
                    params = [arg.arg for arg in node.args.args]
                    signature = f"{node.name}({','.join(params)})"
                    signatures.add(signature)
                    
        except:
            pass
            
        return signatures
    
    def check_similar_functionality(self, filepath: Path, all_scripts: List[Path]) -> None:
        """Check if script has similar functionality to existing scripts."""
        if filepath.suffix != '.py':
            return
            
        current_sigs = self.get_function_signatures(filepath)
        if not current_sigs:
            return
            
        for other_script in all_scripts:
            if other_script == filepath or other_script.suffix != '.py':
                continue
                
            other_sigs = self.get_function_signatures(other_script)
            
            # Check for significant overlap
            common_sigs = current_sigs & other_sigs
            if len(common_sigs) >= 3 or (len(common_sigs) >= 2 and len(current_sigs) <= 4):
                self.warnings.append((
                    filepath,
                    f"Similar functionality found in {other_script}",
                    f"Common functions: {', '.join(list(common_sigs)[:3])}"
                ))
    
    def check_content_similarity(self, filepath: Path, all_scripts: List[Path]) -> None:
        """Check for highly similar content between scripts."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content1 = f.read()
                
            # Normalize content for comparison
            normalized1 = self._normalize_content(content1)
            
            for other_script in all_scripts:
                if other_script == filepath:
                    continue
                    
                try:
                    with open(other_script, 'r', encoding='utf-8') as f:
                        content2 = f.read()
                        
                    normalized2 = self._normalize_content(content2)
                    
                    # Calculate similarity
                    similarity = self._calculate_similarity(normalized1, normalized2)
                    
                    if similarity > 0.8:
                        self.violations.append((
                            filepath,
                            f"High content similarity ({similarity:.0%}) with {other_script}",
                            "Consolidate duplicate scripts or import shared functionality"
                        ))
                    elif similarity > 0.6:
                        self.warnings.append((
                            filepath,
                            f"Moderate content similarity ({similarity:.0%}) with {other_script}",
                            "Consider extracting common functionality"
                        ))
                        
                except:
                    pass
                    
        except:
            pass
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison."""
        # Remove comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'"""[\s\S]*?"""', '', content)
        content = re.sub(r"'''[\s\S]*?'''", '', content)
        
        # Remove empty lines and extra whitespace
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        return '\n'.join(lines)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        lines1 = set(text1.splitlines())
        lines2 = set(text2.splitlines())
        
        if not lines1 or not lines2:
            return 0.0
            
        intersection = len(lines1 & lines2)
        union = len(lines1 | lines2)
        
        return intersection / union if union > 0 else 0.0

def main():
    """Main function to check script duplication."""
    project_root = Path("/opt/sutazaiapp")
    checker = ScriptDuplicationChecker(project_root)
    
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
    
    print("🔍 Checking for script duplication (Rule 4)...")
    
    # Collect all script files for comparison
    script_files = []
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if filepath.suffix in ['.py', '.sh'] and filepath.exists():
            script_files.append(filepath)
    
    # Also get existing scripts in the project
    existing_scripts = list(project_root.glob('scripts/**/*.py'))
    existing_scripts.extend(list(project_root.glob('scripts/**/*.sh')))
    
    all_scripts = list(set(script_files + existing_scripts))
    
    # Check each file
    for filepath in script_files:
        checker.check_similar_functionality(filepath, all_scripts)
        checker.check_content_similarity(filepath, all_scripts)
    
    # Report results
    if checker.violations:
        print("\n❌ Rule 4 Violation: Script duplication detected")
        print("\n📋 Duplicate scripts found:")
        
        for filepath, issue, fix in checker.violations:
            print(f"\n  File: {filepath}")
            print(f"  Issue: {issue}")
            print(f"  Fix: {fix}")
        
        print("\n📋 How to fix:")
        print("  1. Consolidate duplicate scripts into one")
        print("  2. Extract common functionality into shared modules")
        print("  3. Use imports instead of copy-paste")
        print("  4. Delete redundant scripts")
        
        return 1
    
    if checker.warnings:
        print("\n⚠️  Potential duplication warnings:")
        for filepath, warning, detail in checker.warnings:
            print(f"\n  File: {filepath}")
            print(f"  Warning: {warning}")
            print(f"  Details: {detail}")
    
    print("✅ Rule 4: No script duplication detected")
    return 0

if __name__ == "__main__":
    sys.exit(main())