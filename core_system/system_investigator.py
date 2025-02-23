#!/usr/bin/env python3
import ast
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Set, Tuple


class SystemInvestigator:
    def __init__(self, base_paths=None):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('/var/log/sutazai/system_investigation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Define base paths for investigation
        self.base_paths = base_paths or [
            '/home/ai/Desktop/SutazAI/v1/v1',
            '/media/ai/SutazAI_Storage/SutazAI/v1'
        ]
        
        # Consolidated storage path
        self.consolidated_path = '/media/ai/SutazAI_Storage/SutazAI/v1'
        
        # Configuration for investigation
        self.config = {
            'ignore_patterns': [
                '.git', '.venv', '__pycache__', 
                '.pytest_cache', 'node_modules', 
                '.log', '.tmp', '.pyc'
            ],
            'allowed_extensions': [
                '.py', '.js', '.ts', '.json', 
                '.yaml', '.yml', '.md', '.txt', 
                '.sh', '.dockerfile', '.env'
            ]
        }
        
        # Investigation report
        self.investigation_report = {
            'duplicates': {},
            'architectural_issues': [],
            'dependency_conflicts': [],
            'code_quality_issues': []
        }

    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def find_duplicate_files(self) -> Dict[str, List[str]]:
        """Find duplicate files across base paths."""
        file_hashes = {}
        duplicates = {}

        for base_path in self.base_paths:
            for root, _, files in os.walk(base_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # Skip files not in allowed extensions or matching ignore patterns
                    if not any(ext in filepath for ext in self.config['allowed_extensions']) or \
                       any(pattern in filepath for pattern in self.config['ignore_patterns']):
                        continue

                    try:
                        file_hash = self.calculate_file_hash(filepath)
                        if file_hash in file_hashes:
                            duplicates.setdefault(file_hash, []).append(filepath)
                            file_hashes[file_hash].append(filepath)
                        else:
                            file_hashes[file_hash] = [filepath]
                    except Exception as e:
                        self.logger.error(f"Error processing {filepath}: {e}")

        return {k: v for k, v in duplicates.items() if len(v) > 1}

    def consolidate_files(self, duplicates: Dict[str, List[str]]):
        """Consolidate duplicate files, keeping the most recent version."""
        for hash_key, file_paths in duplicates.items():
            # Sort by modification time, most recent first
            sorted_files = sorted(file_paths, key=os.path.getmtime, reverse=True)
            
            # Keep the first (most recent) file, remove others
            for filepath in sorted_files[1:]:
                try:
                    # Move to consolidated path, preserving directory structure
                    relative_path = os.path.relpath(filepath, start=self.base_paths[0])
                    dest_path = os.path.join(self.consolidated_path, relative_path)
                    
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.move(filepath, dest_path)
                    
                    self.logger.info(f"Consolidated file: {filepath} -> {dest_path}")
                except Exception as e:
                    self.logger.error(f"Error consolidating {filepath}: {e}")

    def analyze_code_structure(self):
        """
        Analyze the code structure of Python files in the project.
        Handles potential encoding issues by trying multiple encodings.
        """
        code_structure_report = {
            'total_files': 0,
            'parsed_files': 0,
            'unparsed_files': [],
            'complexity_metrics': []
        }

        # List of encodings to try
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

        for root, _, files in os.walk(self.base_paths[0]):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    code_structure_report['total_files'] += 1

                    parsed_successfully = False
                    for encoding in encodings_to_try:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                try:
                                    tree = ast.parse(f.read())
                                    
                                    # Analyze complexity
                                    complexity = self._calculate_code_complexity(tree)
                                    code_structure_report['complexity_metrics'].append({
                                        'file': file_path,
                                        'complexity': complexity
                                    })
                                    
                                    code_structure_report['parsed_files'] += 1
                                    parsed_successfully = True
                                    break
                                except SyntaxError as e:
                                    logging.warning(f"Syntax error in {file_path} with {encoding} encoding: {e}")
                        except UnicodeDecodeError:
                            continue

                    if not parsed_successfully:
                        logging.error(f"Could not parse {file_path} with any of the tried encodings")
                        code_structure_report['unparsed_files'].append(file_path)

        # Log the report
        logging.info("Code Structure Analysis Report:")
        logging.info(f"Total Python files: {code_structure_report['total_files']}")
        logging.info(f"Successfully parsed files: {code_structure_report['parsed_files']}")
        
        if code_structure_report['unparsed_files']:
            logging.warning("Files that could not be parsed:")
            for unparsed_file in code_structure_report['unparsed_files']:
                logging.warning(unparsed_file)

        return code_structure_report

    def _calculate_code_complexity(self, tree: ast.Module) -> int:
        """Calculate cyclomatic complexity of a code structure."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
        return complexity

    def check_dependencies(self):
        """Check for dependency conflicts and compatibility."""
        try:
            # Find all requirements.txt files
            requirements_files = []
            for base_path in self.base_paths:
                for root, _, files in os.walk(base_path):
                    if 'requirements.txt' in files:
                        requirements_files.append(os.path.join(root, 'requirements.txt'))
            
            # Compare dependencies
            if len(requirements_files) > 1:
                dependencies = {}
                for req_file in requirements_files:
                    with open(req_file, 'r') as f:
                        deps = [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]
                        dependencies[req_file] = deps
                
                # Check for conflicts
                for file1, deps1 in dependencies.items():
                    for file2, deps2 in dependencies.items():
                        if file1 != file2:
                            conflicts = set(deps1) & set(deps2)
                            if conflicts:
                                self.investigation_report['dependency_conflicts'].append({
                                    'file1': file1,
                                    'file2': file2,
                                    'conflicts': list(conflicts)
                                })
        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")

    def generate_system_report(self, duplicates: Dict[str, List[str]]):
        """Generate a comprehensive system report."""
        self.investigation_report.update({
            'total_duplicate_files': sum(len(files) for files in duplicates.values()),
            'duplicate_file_details': duplicates,
            'base_paths': self.base_paths,
            'consolidated_path': self.consolidated_path
        })
        
        # Write detailed report
        with open('/var/log/sutazai/system_investigation_report.json', 'w') as f:
            json.dump(self.investigation_report, f, indent=2)
        
        # Print summary
        print("🔍 Comprehensive System Investigation Report 🔍")
        print(f"Total Duplicate Files: {self.investigation_report['total_duplicate_files']}")
        
        if self.investigation_report['architectural_issues']:
            print("\n🚨 Architectural Issues:")
            for issue in self.investigation_report['architectural_issues']:
                print(f"  - {issue}")
        
        if self.investigation_report['code_quality_issues']:
            print("\n🔬 Code Quality Issues:")
            for issue in self.investigation_report['code_quality_issues']:
                print(f"  - {issue['file']}: {issue['function']} (Complexity: {issue['complexity']})")
        
        if self.investigation_report['dependency_conflicts']:
            print("\n🧩 Dependency Conflicts:")
            for conflict in self.investigation_report['dependency_conflicts']:
                print(f"  - Between {conflict['file1']} and {conflict['file2']}")
                print(f"    Conflicting packages: {conflict['conflicts']}")
        
        print("\nDetailed report saved to /var/log/sutazai/system_investigation_report.json")

    def run_investigation(self):
        """Run comprehensive system investigation."""
        # Find and consolidate duplicate files
        duplicates = self.find_duplicate_files()
        self.consolidate_files(duplicates)
        
        # Perform in-depth analysis
        self.analyze_code_structure()
        self.check_dependencies()
        
        # Generate final report
        self.generate_system_report(duplicates)

def main():
    investigator = SystemInvestigator()
    investigator.run_investigation()

if __name__ == '__main__':
    main() 