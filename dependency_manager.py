#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from typing import Dict, List, Any

class DependencyManager:
    def __init__(self, base_path: str = '/media/ai/SutazAI_Storage/SutazAI/v1'):
        self.base_path = base_path
        self.requirements_files = self._find_requirements_files()
        self.consolidated_requirements = {}
    
    def _find_requirements_files(self) -> List[str]:
        """Find all requirements.txt files in the project."""
        requirements_files = []
        for root, _, files in os.walk(self.base_path):
            if 'requirements.txt' in files:
                requirements_files.append(os.path.join(root, 'requirements.txt'))
        return requirements_files
    
    def parse_requirements(self) -> Dict[str, str]:
        """Parse and consolidate requirements from multiple files."""
        for req_file in self.requirements_files:
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            package, version = line.split('==')
                            # Prefer the latest version
                            if package not in self.consolidated_requirements or \
                               self._compare_versions(version, self.consolidated_requirements[package]) > 0:
                                self.consolidated_requirements[package] = version
                        except ValueError:
                            # Handle packages without explicit version
                            if line not in self.consolidated_requirements:
                                self.consolidated_requirements[line] = ''
        return self.consolidated_requirements
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings."""
        if not version1:
            return -1
        if not version2:
            return 1
        
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        for i in range(min(len(v1_parts), len(v2_parts))):
            if v1_parts[i] > v2_parts[i]:
                return 1
            elif v1_parts[i] < v2_parts[i]:
                return -1
        
        return len(v1_parts) - len(v2_parts)
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate dependencies and check for potential conflicts."""
        validation_report = {
            'total_packages': len(self.consolidated_requirements),
            'conflicts': [],
            'recommendations': []
        }
        
        # Perform dependency validation
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run'] + 
                           [f"{pkg}=={ver}" if ver else pkg for pkg, ver in self.consolidated_requirements.items()], 
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            validation_report['conflicts'].append(str(e))
        
        return validation_report
    
    def generate_consolidated_requirements(self, output_path: str = None):
        """Generate a consolidated requirements file."""
        if not output_path:
            output_path = os.path.join(self.base_path, 'requirements.txt')
        
        with open(output_path, 'w') as f:
            for pkg, ver in sorted(self.consolidated_requirements.items()):
                if ver:
                    f.write(f"{pkg}=={ver}\n")
                else:
                    f.write(f"{pkg}\n")
        
        return output_path
    
    def generate_report(self, validation_report: Dict[str, Any]) -> str:
        """Generate a comprehensive dependency management report."""
        report_path = '/var/log/sutazai/dependency_report.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        report = {
            'consolidated_requirements': self.consolidated_requirements,
            'validation_results': validation_report
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path

def main():
    dependency_manager = DependencyManager()
    
    # Parse requirements
    dependency_manager.parse_requirements()
    
    # Validate dependencies
    validation_report = dependency_manager.validate_dependencies()
    
    # Generate consolidated requirements
    consolidated_req_path = dependency_manager.generate_consolidated_requirements()
    
    # Generate report
    report_path = dependency_manager.generate_report(validation_report)
    
    print(f"Consolidated requirements saved to: {consolidated_req_path}")
    print(f"Dependency report saved to: {report_path}")

if __name__ == '__main__':
    main() 