#!/usr/bin/env python3
"""
ULTRA Requirements Cleaner
Agent_86 (Import_Analyzer) - ULTRA Cleanup Mission Part 2

This script performs comprehensive cleanup of requirements files by:
- Finding all requirements files in the codebase
- Analyzing actual import usage to determine needed packages
- Removing unused dependencies
- Consolidating duplicate requirements
- Standardizing version specifications

Usage:
    python3 scripts/ultra_requirements_cleaner.py --scan
    python3 scripts/ultra_requirements_cleaner.py --clean --consolidate
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import logging
import pkg_resources
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RequirementsAnalyzer:
    """Analyze and clean requirements files"""
    
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.requirements_files = []
        self.used_packages = set()
        self.package_mappings = self._create_package_mappings()
        self.stats = {
            'requirements_files_found': 0,
            'total_requirements': 0,
            'unused_requirements': 0,
            'duplicate_requirements': 0,
            'consolidated_files': 0
        }
    
    def _create_package_mappings(self) -> Dict[str, str]:
        """Map import names to package names"""
        return {
            # Common mappings where import name differs from package name
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow',
            'yaml': 'PyYAML',
            'bs4': 'beautifulsoup4',
            'dateutil': 'python-dateutil',
            'serial': 'pyserial',
            'psycopg2': 'psycopg2-binary',
            'MySQLdb': 'mysqlclient',
            'ldap': 'python-ldap',
            'magic': 'python-magic',
            'docx': 'python-docx',
            'pptx': 'python-pptx',
            'jwt': 'PyJWT',
            'redis': 'redis',
            'sqlalchemy': 'SQLAlchemy',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'pydantic': 'pydantic',
            'streamlit': 'streamlit',
            'ollama': 'ollama',
            'httpx': 'httpx',
            'aioredis': 'aioredis',
            'asyncpg': 'asyncpg',
            'chromadb': 'chromadb',
            'qdrant_client': 'qdrant-client',
            'neo4j': 'neo4j',
            'prometheus_client': 'prometheus-client'
        }
    
    def find_requirements_files(self) -> List[Path]:
        """Find all requirements files in the codebase"""
        requirements_patterns = [
            'requirements*.txt',
            'requirements*.in',
            'pyproject.toml',
            'setup.py',
            'Pipfile',
            'environment.yml'
        ]
        
        requirements_files = []
        
        for pattern in requirements_patterns:
            for req_file in self.root_path.rglob(pattern):
                # Skip files in excluded directories
                if any(exclude in str(req_file) for exclude in ['.git', '__pycache__', 'venv', 'env', 'node_modules']):
                    continue
                requirements_files.append(req_file)
        
        self.requirements_files = requirements_files
        self.stats['requirements_files_found'] = len(requirements_files)
        logger.info(f"Found {len(requirements_files)} requirements files")
        
        return requirements_files
    
    def parse_requirements_file(self, file_path: Path) -> List[Dict]:
        """Parse a requirements file and extract packages"""
        requirements = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.name == 'pyproject.toml':
                return self._parse_pyproject_toml(content)
            elif file_path.name == 'setup.py':
                return self._parse_setup_py(content)
            elif file_path.name == 'Pipfile':
                return self._parse_pipfile(content)
            elif file_path.name.endswith('.yml') or file_path.name.endswith('.yaml'):
                return self._parse_conda_env(content)
            else:
                return self._parse_pip_requirements(content)
                
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []
    
    def _parse_pip_requirements(self, content: str) -> List[Dict]:
        """Parse pip requirements format"""
        requirements = []
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Remove inline comments
            if '#' in line:
                line = line.split('#')[0].strip()
            
            # Skip options and URLs
            if line.startswith('-') or line.startswith('http'):
                continue
            
            # Parse package specification
            match = re.match(r'^([a-zA-Z0-9][a-zA-Z0-9_.-]*[a-zA-Z0-9])\s*([<>=!]+.*)?$', line)
            if match:
                package_name = match.group(1)
                version_spec = match.group(2) or ''
                
                requirements.append({
                    'name': package_name,
                    'version': version_spec.strip(),
                    'line': line_num,
                    'original': line
                })
        
        return requirements
    
    def _parse_pyproject_toml(self, content: str) -> List[Dict]:
        """Parse pyproject.toml dependencies"""
        requirements = []
        
        # Simple regex-based parsing for dependencies
        dependencies_section = False
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            if line == 'dependencies = [':
                dependencies_section = True
                continue
            elif line == ']' and dependencies_section:
                dependencies_section = False
                continue
            
            if dependencies_section and line.startswith('"'):
                # Extract package from "package>=version" format
                package_line = line.strip('"",')
                match = re.match(r'^([a-zA-Z0-9][a-zA-Z0-9_.-]*[a-zA-Z0-9])\s*([<>=!]+.*)?$', package_line)
                if match:
                    requirements.append({
                        'name': match.group(1),
                        'version': match.group(2) or '',
                        'line': line_num,
                        'original': line
                    })
        
        return requirements
    
    def _parse_setup_py(self, content: str) -> List[Dict]:
        """Parse setup.py install_requires"""
        requirements = []
        
        # Extract install_requires section
        install_requires_match = re.search(
            r'install_requires\s*=\s*\[(.*?)\]',
            content, re.DOTALL
        )
        
        if install_requires_match:
            deps_content = install_requires_match.group(1)
            for line in deps_content.split('\n'):
                line = line.strip().strip(',"\' ')
                if line and not line.startswith('#'):
                    match = re.match(r'^([a-zA-Z0-9][a-zA-Z0-9_.-]*[a-zA-Z0-9])\s*([<>=!]+.*)?$', line)
                    if match:
                        requirements.append({
                            'name': match.group(1),
                            'version': match.group(2) or '',
                            'line': 0,  # Line numbers not available from regex
                            'original': line
                        })
        
        return requirements
    
    def _parse_pipfile(self, content: str) -> List[Dict]:
        """Parse Pipfile dependencies"""
        # This would require toml parsing for proper implementation
        # For now, return empty list
        return []
    
    def _parse_conda_env(self, content: str) -> List[Dict]:
        """Parse conda environment.yml"""
        # This would require yaml parsing for proper implementation
        # For now, return empty list
        return []
    
    def get_actually_used_packages(self) -> Set[str]:
        """Scan Python files to find actually imported packages"""
        used_packages = set()
        
        # Find all Python files
        for py_file in self.root_path.rglob("*.py"):
            if any(exclude in str(py_file) for exclude in ['.git', '__pycache__', 'venv', 'env']):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find import statements
                import_matches = re.findall(r'^(?:from\s+(\w+)|import\s+(\w+))', content, re.MULTILINE)
                
                for match in import_matches:
                    package = match[0] or match[1]
                    if package:
                        # Get root package name
                        root_package = package.split('.')[0]
                        used_packages.add(root_package)
                        
                        # Also add the mapped package name if available
                        if root_package in self.package_mappings:
                            used_packages.add(self.package_mappings[root_package])
            
            except Exception as e:
                logger.debug(f"Error reading {py_file}: {e}")
        
        self.used_packages = used_packages
        logger.info(f"Found {len(used_packages)} actually used packages")
        return used_packages
    
    def analyze_requirements(self) -> Dict[str, Any]:
        """Analyze all requirements files for unused dependencies"""
        results = {}
        
        # Get actually used packages
        used_packages = self.get_actually_used_packages()
        
        for req_file in self.requirements_files:
            requirements = self.parse_requirements_file(req_file)
            
            if not requirements:
                continue
            
            unused_requirements = []
            duplicate_requirements = defaultdict(list)
            
            # Track package names for duplicate detection
            seen_packages = {}
            
            for req in requirements:
                package_name = req['name'].lower().replace('_', '-')
                
                # Check for duplicates
                if package_name in seen_packages:
                    duplicate_requirements[package_name].append(req)
                else:
                    seen_packages[package_name] = req
                
                # Check if package is actually used
                is_used = any(
                    package_name == used_pkg.lower().replace('_', '-') or
                    req['name'].lower().replace('_', '-') == used_pkg.lower().replace('_', '-')
                    for used_pkg in used_packages
                )
                
                if not is_used:
                    # Check if it's a core dependency that might not appear in imports
                    if not self._is_core_dependency(req['name']):
                        unused_requirements.append(req)
            
            relative_path = str(req_file.relative_to(self.root_path))
            results[relative_path] = {
                'total_requirements': len(requirements),
                'unused_requirements': unused_requirements,
                'duplicate_requirements': dict(duplicate_requirements),
                'requirements': requirements
            }
            
            self.stats['total_requirements'] += len(requirements)
            self.stats['unused_requirements'] += len(unused_requirements)
            self.stats['duplicate_requirements'] += len(duplicate_requirements)
        
        return results
    
    def _is_core_dependency(self, package_name: str) -> bool:
        """Check if package is a core dependency that might not appear in imports"""
        core_deps = {
            # Build tools
            'setuptools', 'wheel', 'pip', 'build',
            # Testing
            'pytest', 'pytest-cov', 'pytest-Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test', 'coverage',
            # Linting/Formatting
            'black', 'flake8', 'mypy', 'isort', 'pylint',
            # Development tools
            'pre-commit', 'bandit', 'safety',
            # Documentation
            'sphinx', 'mkdocs',
            # Server/deployment
            'gunicorn', 'uvicorn', 'supervisor',
            # Database drivers (might be used indirectly)
            'psycopg2-binary', 'mysqlclient', 'pymongo'
        }
        
        return package_name.lower() in core_deps
    
    def clean_requirements_file(self, file_path: Path, unused_reqs: List[Dict]) -> bool:
        """Remove unused requirements from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Mark lines for removal (in reverse order to maintain line numbers)
            lines_to_remove = set()
            for unused in unused_reqs:
                if unused['line'] > 0:  # Valid line number
                    lines_to_remove.add(unused['line'] - 1)  # Convert to 0-based
            
            # Remove lines
            cleaned_lines = [
                line for i, line in enumerate(lines)
                if i not in lines_to_remove
            ]
            
            # Write cleaned file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            
            logger.info(f"Cleaned {file_path}: removed {len(unused_reqs)} unused requirements")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {e}")
            return False
    
    def consolidate_requirements(self) -> bool:
        """Consolidate duplicate requirements files"""
        # Find the main requirements.txt file
        main_req_file = self.root_path / "requirements.txt"
        
        all_requirements = set()
        files_to_consolidate = []
        
        for req_file in self.requirements_files:
            if req_file.name.startswith('requirements') and req_file.suffix == '.txt':
                requirements = self.parse_requirements_file(req_file)
                
                for req in requirements:
                    all_requirements.add(f"{req['name']}{req['version']}")
                
                if req_file != main_req_file:
                    files_to_consolidate.append(req_file)
        
        if not files_to_consolidate:
            logger.info("No requirements files to consolidate")
            return False
        
        # Write consolidated requirements
        try:
            with open(main_req_file, 'w', encoding='utf-8') as f:
                f.write("# Consolidated requirements file\n")
                f.write(f"# Generated on {datetime.now().isoformat()}\n")
                f.write("# by Agent_86 (Import_Analyzer) ULTRA cleanup\n\n")
                
                for req in sorted(all_requirements):
                    f.write(f"{req}\n")
            
            logger.info(f"Consolidated {len(files_to_consolidate)} requirements files into {main_req_file}")
            
            # Optionally remove duplicate files
            for file_path in files_to_consolidate:
                logger.info(f"Consider removing duplicate file: {file_path}")
            
            self.stats['consolidated_files'] = len(files_to_consolidate)
            return True
            
        except Exception as e:
            logger.error(f"Error consolidating requirements: {e}")
            return False
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive requirements cleanup report"""
        report_lines = [
            "=" * 80,
            "ULTRA REQUIREMENTS CLEANUP REPORT",
            f"Generated by Agent_86 (Import_Analyzer) on {datetime.now().isoformat()}",
            "=" * 80,
            "",
            "SUMMARY:",
            f"- Requirements files found: {self.stats['requirements_files_found']}",
            f"- Total requirements analyzed: {self.stats['total_requirements']}",
            f"- Unused requirements found: {self.stats['unused_requirements']}",
            f"- Duplicate requirements found: {self.stats['duplicate_requirements']}",
            f"- Files consolidated: {self.stats['consolidated_files']}",
            "",
            "ACTUALLY USED PACKAGES:",
        ]
        
        for package in sorted(self.used_packages):
            report_lines.append(f"- {package}")
        
        report_lines.extend([
            "",
            "DETAILED ANALYSIS BY FILE:",
            ""
        ])
        
        for file_path, analysis in analysis_results.items():
            report_lines.append(f"File: {file_path}")
            report_lines.append(f"  Total requirements: {analysis['total_requirements']}")
            report_lines.append(f"  Unused requirements: {len(analysis['unused_requirements'])}")
            report_lines.append(f"  Duplicate requirements: {len(analysis['duplicate_requirements'])}")
            
            if analysis['unused_requirements']:
                report_lines.append("  Unused:")
                for unused in analysis['unused_requirements']:
                    report_lines.append(f"    - {unused['name']}{unused['version']}")
            
            if analysis['duplicate_requirements']:
                report_lines.append("  Duplicates:")
                for pkg_name, duplicates in analysis['duplicate_requirements'].items():
                    report_lines.append(f"    - {pkg_name}: {len(duplicates)} versions")
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            "CLEANUP RECOMMENDATIONS:",
            "1. Remove unused requirements to reduce installation time",
            "2. Consolidate duplicate requirements files",
            "3. Pin versions for production deployments",
            "4. Use virtual environments for development",
            "5. Consider using Poetry or pipenv for dependency management",
            "=" * 80
        ])
        
        return '\n'.join(report_lines)


def main():
    """Main function for requirements cleanup"""
    parser = argparse.ArgumentParser(description="ULTRA Requirements Cleaner")
    parser.add_argument('--scan', action='store_true', help='Scan requirements files')
    parser.add_argument('--clean', action='store_true', help='Remove unused requirements')
    parser.add_argument('--consolidate', action='store_true', help='Consolidate duplicate files')
    parser.add_argument('--report', default='ultra_requirements_cleanup_report.txt',
                       help='Report output file')
    
    args = parser.parse_args()
    
    if not any([args.scan, args.clean, args.consolidate]):
        parser.print_help()
        return
    
    analyzer = RequirementsAnalyzer()
    
    # Find requirements files
    requirements_files = analyzer.find_requirements_files()
    if not requirements_files:
        logger.info("❌ No requirements files found")
        return
    
    logger.info(f"📦 Found {len(requirements_files)} requirements files")
    
    # Analyze requirements
    if args.scan or args.clean:
        logger.info("🔍 Analyzing requirements...")
        results = analyzer.analyze_requirements()
        
        logger.info(f"📊 Found {analyzer.stats['unused_requirements']} unused requirements")
        logger.info(f"📊 Found {analyzer.stats['duplicate_requirements']} duplicate requirements")
        
        # Generate and save report
        report = analyzer.generate_report(results)
        with open(args.report, 'w') as f:
            f.write(report)
        logger.info(f"📋 Report saved to: {args.report}")
        
        # Clean unused requirements if requested
        if args.clean:
            logger.info("🧹 Cleaning unused requirements...")
            cleaned_files = 0
            
            for file_path_str, analysis in results.items():
                if analysis['unused_requirements']:
                    file_path = analyzer.root_path / file_path_str
                    if analyzer.clean_requirements_file(file_path, analysis['unused_requirements']):
                        cleaned_files += 1
            
            logger.info(f"✅ Cleaned {cleaned_files} requirements files")
    
    # Consolidate requirements if requested
    if args.consolidate:
        logger.info("📝 Consolidating requirements files...")
        if analyzer.consolidate_requirements():
            logger.info("✅ Requirements files consolidated")
        else:
            logger.info("ℹ️ No consolidation needed")
    
    logger.info("🎉 ULTRA requirements cleanup complete!")


if __name__ == "__main__":
    main()