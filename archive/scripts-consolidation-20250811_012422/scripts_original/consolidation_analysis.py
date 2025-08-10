#!/usr/bin/env python3
"""
Python Script Consolidation Analysis
====================================
Analyzes all Python scripts in the codebase to identify:
1. Duplicate functionality
2. Consolidation opportunities  
3. Script categories and structure
4. Redundant files
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

class ScriptAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.scripts: List[Dict] = []
        self.duplicates: Dict[str, List[str]] = defaultdict(list)
        self.categories = {
            'monitoring': ['monitor', 'health', 'metrics', 'observe', 'track'],
            'deployment': ['deploy', 'orchestrat', 'launch', 'startup', 'activate'],
            'testing': ['test', 'valid', 'check', 'verify', 'smoke'],
            'maintenance': ['clean', 'fix', 'update', 'maintain', 'garbage', 'hygiene'],
            'utils': ['util', 'helper', 'tool', 'lib'],
            'automation': ['automat', 'schedul', 'cron', 'batch'],
            'security': ['secur', 'auth', 'cors', 'hardcode'],
            'analysis': ['analyz', 'audit', 'scan', 'inspect', 'profile']
        }
        self.consolidation_plan = {}
        
    def analyze_all_scripts(self):
        """Analyze all Python scripts in the codebase"""
        python_files = list(self.root_path.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            if self._should_analyze(file_path):
                script_info = self._analyze_script(file_path)
                if script_info:
                    self.scripts.append(script_info)
        
        print(f"Analyzed {len(self.scripts)} scripts")
        
    def _should_analyze(self, file_path: Path) -> bool:
        """Determine if a script should be analyzed"""
        # Skip certain directories
        skip_dirs = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv', 
            'site-packages', 'security_audit_env'
        }
        
        # Check if any parent is in skip_dirs
        if any(part in skip_dirs for part in file_path.parts):
            return False
            
        # Skip if it's clearly an application file, not a script
        app_patterns = ['/app/', '/frontend/', '/backend/', '/agents/']
        file_str = str(file_path)
        if any(pattern in file_str for pattern in app_patterns):
            # Only analyze scripts within these directories
            if '/scripts/' not in file_str:
                return False
                
        return True
        
    def _analyze_script(self, file_path: Path) -> Dict:
        """Analyze individual script"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Calculate hash for duplicate detection
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            script_info = {
                'path': str(file_path),
                'relative_path': str(file_path.relative_to(self.root_path)),
                'name': file_path.name,
                'size': len(content),
                'lines': len(content.split('\n')),
                'hash': content_hash,
                'category': self._categorize_script(file_path, content),
                'imports': self._extract_imports(content),
                'functions': self._extract_functions(content),
                'description': self._extract_description(content),
                'shebang': content.startswith('#!/'),
                'executable': os.access(file_path, os.X_OK)
            }
            
            return script_info
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
            
    def _categorize_script(self, file_path: Path, content: str) -> List[str]:
        """Categorize script based on path and content"""
        categories = []
        path_lower = str(file_path).lower()
        content_lower = content.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in path_lower for keyword in keywords):
                categories.append(category)
            elif any(keyword in content_lower[:1000] for keyword in keywords):
                categories.append(category)
                
        if not categories:
            categories = ['misc']
            
        return categories
        
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements"""
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except (IOError, OSError, FileNotFoundError) as e:
            # TODO: Review this exception handling
            logger.error(f"Unexpected exception: {e}", exc_info=True)
            # Fallback to regex if AST fails
            import_pattern = r'(?:from\s+(\S+)\s+)?import\s+([^\n]+)'
            matches = re.findall(import_pattern, content)
            for match in matches:
                if match[0]:
                    imports.append(match[0])
                    
        return list(set(imports))
        
    def _extract_functions(self, content: str) -> List[str]:
        """Extract function names"""
        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    functions.append(f"class:{node.name}")
        except (IOError, OSError, FileNotFoundError) as e:
            # TODO: Review this exception handling
            logger.error(f"Unexpected exception: {e}", exc_info=True)
            # Fallback to regex
            func_pattern = r'def\s+(\w+)\s*\('
            functions.extend(re.findall(func_pattern, content))
            class_pattern = r'class\s+(\w+)\s*[(:] '
            functions.extend([f"class:{name}" for name in re.findall(class_pattern, content)])
            
        return list(set(functions))
        
    def _extract_description(self, content: str) -> str:
        """Extract script description from docstring or comments"""
        # Try docstring first
        try:
            tree = ast.parse(content)
            if (isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and 
                isinstance(tree.body[0].value.value, str)):
                return tree.body[0].value.value.strip()[:200]
        except (IOError, OSError, FileNotFoundError) as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
            
        # Fallback to comments
        lines = content.split('\n')
        for line in lines[:20]:
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                # Find closing quotes
                desc_lines = []
                for i, l in enumerate(lines):
                    if i == 0 or (line.strip().startswith('"""') or line.strip().startswith("'''")):
                        continue
                    if '"""' in l or "'''" in l:
                        break
                    desc_lines.append(l.strip())
                return ' '.join(desc_lines)[:200]
                
            if line.strip().startswith('#') and len(line.strip()) > 10:
                return line.strip()[1:].strip()[:200]
                
        return ""
        
    def find_duplicates(self):
        """Find duplicate scripts by content hash and functionality"""
        # Hash-based duplicates
        hash_groups = defaultdict(list)
        for script in self.scripts:
            hash_groups[script['hash']].append(script['path'])
            
        for hash_val, paths in hash_groups.items():
            if len(paths) > 1:
                self.duplicates['exact_duplicate'].extend(paths)
                
        # Functionality-based duplicates
        func_groups = defaultdict(list)
        for script in self.scripts:
            # Create signature from functions and main imports
            sig_parts = []
            sig_parts.extend(sorted(script['functions'][:10]))  # Top 10 functions
            sig_parts.extend(sorted([imp for imp in script['imports'] 
                                   if not imp.startswith('_') and imp not in 
                                   ['os', 'sys', 'time', 'json', 'logging']][:5]))
            signature = '|'.join(sig_parts)
            if signature:
                func_groups[signature].append(script['path'])
                
        for signature, paths in func_groups.items():
            if len(paths) > 1:
                self.duplicates['functional_duplicate'].extend(paths)
        
    def create_consolidation_plan(self):
        """Create consolidation plan"""
        # Group scripts by category
        category_groups = defaultdict(list)
        for script in self.scripts:
            for category in script['category']:
                category_groups[category].append(script)
                
        # Create consolidation targets
        consolidation_targets = {
            'monitoring': 'scripts/monitoring/system_monitor.py',
            'deployment': 'scripts/deployment/deployment_manager.py', 
            'testing': 'scripts/testing/test_runner.py',
            'maintenance': 'scripts/maintenance/maintenance_manager.py',
            'utils': 'scripts/utils/common_utils.py',
            'automation': 'scripts/automation/automation_engine.py',
            'security': 'scripts/security/security_validator.py',
            'analysis': 'scripts/analysis/system_analyzer.py'
        }
        
        for category, target_file in consolidation_targets.items():
            scripts_in_category = category_groups.get(category, [])
            if len(scripts_in_category) > 5:  # Only consolidate if many scripts
                self.consolidation_plan[category] = {
                    'target_file': target_file,
                    'source_scripts': [s['path'] for s in scripts_in_category],
                    'total_scripts': len(scripts_in_category),
                    'total_lines': sum(s['lines'] for s in scripts_in_category)
                }
                
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'total_scripts': len(self.scripts),
                'total_lines': sum(s['lines'] for s in self.scripts),
                'categories': dict(Counter(cat for s in self.scripts for cat in s['category'])),
                'duplicates_found': {k: len(v) for k, v in self.duplicates.items()},
                'consolidation_opportunities': len(self.consolidation_plan)
            },
            'category_breakdown': {},
            'top_largest_scripts': sorted(self.scripts, key=lambda x: x['lines'], reverse=True)[:20],
            'duplicates': dict(self.duplicates),
            'consolidation_plan': self.consolidation_plan,
            'detailed_scripts': self.scripts
        }
        
        # Category breakdown
        for category in self.categories.keys():
            cat_scripts = [s for s in self.scripts if category in s['category']]
            if cat_scripts:
                report['category_breakdown'][category] = {
                    'count': len(cat_scripts),
                    'total_lines': sum(s['lines'] for s in cat_scripts),
                    'scripts': [{'path': s['relative_path'], 'lines': s['lines']} for s in cat_scripts]
                }
        
        return report
        
    def print_summary(self):
        """Print analysis summary"""
        report = self.generate_report()
        
        print("\n" + "="*80)
        print("PYTHON SCRIPT CONSOLIDATION ANALYSIS")
        print("="*80)
        
        print(f"\nSUMMARY:")
        print(f"  Total Python scripts: {report['summary']['total_scripts']}")
        print(f"  Total lines of code: {report['summary']['total_lines']:,}")
        print(f"  Consolidation opportunities: {report['summary']['consolidation_opportunities']}")
        
        print(f"\nCATEGORY BREAKDOWN:")
        for category, count in report['summary']['categories'].items():
            print(f"  {category:12}: {count:3} scripts")
            
        print(f"\nTOP 10 LARGEST SCRIPTS:")
        for script in report['top_largest_scripts'][:10]:
            print(f"  {script['lines']:4} lines - {script['relative_path']}")
            
        print(f"\nCONSOLIDATION PLAN:")
        for category, plan in report['consolidation_plan'].items():
            print(f"  {category:12}: {plan['total_scripts']:3} scripts → {plan['target_file']}")
            print(f"                 ({plan['total_lines']:,} lines total)")
            
        if report['duplicates']:
            print(f"\nDUPLICATES FOUND:")
            for dup_type, paths in report['duplicates'].items():
                if paths:
                    print(f"  {dup_type}: {len(paths)} files")
                    
        print("\n" + "="*80)
        
        # Save detailed report
        with open('/opt/sutazaiapp/script_consolidation_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("Detailed report saved to: script_consolidation_analysis.json")

def main():
    analyzer = ScriptAnalyzer('/opt/sutazaiapp')
    print("Analyzing Python scripts...")
    analyzer.analyze_all_scripts()
    analyzer.find_duplicates()
    analyzer.create_consolidation_plan()
    analyzer.print_summary()

if __name__ == "__main__":
    main()