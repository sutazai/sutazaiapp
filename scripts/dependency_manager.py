#!/usr/bin/env python3
"""
Ultra-Comprehensive Dependency Management System

Provides advanced capabilities for:
- Dependency tracking
- Version management
- Compatibility checking
- Security vulnerability scanning
- Automated dependency updates
"""

import os
import sys
import json
import logging
import subprocess
import yaml
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DependencyAnalysisReport:
    """
    Comprehensive dependency analysis report
    """
    timestamp: str
    total_dependencies: int
    outdated_packages: List[Dict[str, str]]
    security_vulnerabilities: List[Dict[str, str]]
    compatibility_issues: List[Dict[str, str]]
    optimization_recommendations: List[str]

class AdvancedDependencyManager:
    """
    Ultra-Comprehensive Dependency Management System
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        config_path: Optional[str] = None
    ):
        """
        Initialize Advanced Dependency Manager
        
        Args:
            base_dir (str): Base project directory
            config_path (Optional[str]): Path to dependency configuration
        """
        # Core configuration
        self.base_dir = base_dir
        self.config_path = config_path or os.path.join(base_dir, 'config', 'dependency_management_config.yml')
        
        # Logging setup
        log_dir = os.path.join(base_dir, 'logs', 'dependency_management')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(log_dir, 'dependency_manager.log')
        )
        self.logger = logging.getLogger('SutazAI.DependencyManager')
        
        # Load configuration
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = self._generate_default_config()
    
    def _generate_default_config(self) -> Dict[str, Any]:
        """
        Generate a default dependency management configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'requirements_files': [
                'requirements.txt',
                'requirements-prod.txt',
                'requirements-test.txt'
            ],
            'update_strategy': {
                'check_frequency': 'daily',
                'auto_update': False,
                'update_type': 'minor'
            },
            'security_scanning': {
                'enabled': True,
                'scan_frequency': 'weekly'
            },
            'compatibility_checks': {
                'python_versions': ['3.8', '3.9', '3.10'],
                'strict_version_matching': True
            }
        }
    
    def analyze_dependencies(self) -> DependencyAnalysisReport:
        """
        Perform comprehensive dependency analysis
        
        Returns:
            Detailed dependency analysis report
        """
        # Initialize analysis report
        analysis_report = DependencyAnalysisReport(
            timestamp=datetime.now().isoformat(),
            total_dependencies=0,
            outdated_packages=[],
            security_vulnerabilities=[],
            compatibility_issues=[],
            optimization_recommendations=[]
        )
        
        # Analyze each requirements file
        for req_file in self.config.get('requirements_files', []):
            full_path = os.path.join(self.base_dir, req_file)
            
            if not os.path.exists(full_path):
                self.logger.warning(f"Requirements file not found: {full_path}")
                continue
            
            # Analyze dependencies in the file
            dependencies = self._parse_requirements_file(full_path)
            analysis_report.total_dependencies += len(dependencies)
            
            # Check for outdated packages
            outdated = self._check_outdated_packages(dependencies)
            analysis_report.outdated_packages.extend(outdated)
            
            # Perform security vulnerability scanning
            vulnerabilities = self._scan_security_vulnerabilities(dependencies)
            analysis_report.security_vulnerabilities.extend(vulnerabilities)
            
            # Check package compatibility
            compatibility_issues = self._check_package_compatibility(dependencies)
            analysis_report.compatibility_issues.extend(compatibility_issues)
        
        # Generate optimization recommendations
        analysis_report.optimization_recommendations = self._generate_optimization_recommendations(
            analysis_report
        )
        
        # Persist and log analysis report
        self._persist_analysis_report(analysis_report)
        
        return analysis_report
    
    def _parse_requirements_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parse requirements file and extract package details
        
        Args:
            file_path (str): Path to requirements file
        
        Returns:
            List of package details
        """
        dependencies = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse package specification
                match = re.match(r'^([a-zA-Z0-9_-]+)([=<>]=?.*)?$', line)
                if match:
                    package_name = match.group(1)
                    version = match.group(2)[1:] if match.group(2) else 'latest'
                    
                    dependencies.append({
                        'name': package_name,
                        'version': version,
                        'source_file': file_path
                    })
        
        return dependencies
    
    def _check_outdated_packages(self, dependencies: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Check for outdated packages
        
        Args:
            dependencies (List): List of package details
        
        Returns:
            List of outdated packages
        """
        outdated_packages = []
        
        try:
            # Use pip list --outdated to check package versions
            outdated_output = subprocess.check_output(
                [sys.executable, '-m', 'pip', 'list', '--outdated'], 
                universal_newlines=True
            )
            
            for line in outdated_output.splitlines()[2:]:
                package, current, latest, type_update = line.split()
                
                # Find matching dependency
                matching_deps = [
                    dep for dep in dependencies 
                    if dep['name'].lower() == package.lower()
                ]
                
                if matching_deps:
                    outdated_packages.append({
                        'name': package,
                        'current_version': current,
                        'latest_version': latest,
                        'update_type': type_update,
                        'source_dependencies': [dep['source_file'] for dep in matching_deps]
                    })
        
        except subprocess.CalledProcessError:
            self.logger.warning("Could not check for outdated packages")
        
        return outdated_packages
    
    def _scan_security_vulnerabilities(self, dependencies: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Scan dependencies for known security vulnerabilities
        
        Args:
            dependencies (List): List of package details
        
        Returns:
            List of security vulnerabilities
        """
        vulnerabilities = []
        
        try:
            # Use safety to check for known vulnerabilities
            safety_output = subprocess.check_output(
                [sys.executable, '-m', 'safety', 'check'], 
                universal_newlines=True
            )
            
            for line in safety_output.splitlines():
                if 'is affected' in line:
                    package, version, vulnerability_id = line.split(' ')[:3]
                    
                    # Find matching dependency
                    matching_deps = [
                        dep for dep in dependencies 
                        if dep['name'].lower() == package.lower()
                    ]
                    
                    vulnerabilities.append({
                        'package': package,
                        'version': version,
                        'vulnerability_id': vulnerability_id,
                        'source_dependencies': [dep['source_file'] for dep in matching_deps]
                    })
        
        except subprocess.CalledProcessError:
            self.logger.warning("Security vulnerability scanning failed")
        
        return vulnerabilities
    
    def _check_package_compatibility(self, dependencies: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Check package compatibility with specified Python versions
        
        Args:
            dependencies (List): List of package details
        
        Returns:
            List of compatibility issues
        """
        compatibility_issues = []
        
        for dep in dependencies:
            try:
                # Check package compatibility using pip
                output = subprocess.check_output(
                    [sys.executable, '-m', 'pip', 'show', dep['name']], 
                    universal_newlines=True
                )
                
                # Extract Python version requirements
                requires_python = re.search(r'Requires-Python: (.*)', output)
                
                if requires_python:
                    python_req = requires_python.group(1)
                    
                    # Check against configured Python versions
                    compatible_versions = self._check_version_compatibility(
                        python_req, 
                        self.config['compatibility_checks']['python_versions']
                    )
                    
                    if not compatible_versions:
                        compatibility_issues.append({
                            'package': dep['name'],
                            'version': dep['version'],
                            'python_requirement': python_req,
                            'source_file': dep['source_file']
                        })
            
            except subprocess.CalledProcessError:
                self.logger.warning(f"Could not check compatibility for {dep['name']}")
        
        return compatibility_issues
    
    def _check_version_compatibility(
        self, 
        package_requirement: str, 
        python_versions: List[str]
    ) -> List[str]:
        """
        Check package compatibility with specified Python versions
        
        Args:
            package_requirement (str): Package Python version requirement
            python_versions (List): List of Python versions to check
        
        Returns:
            List of compatible Python versions
        """
        import packaging.requirements
        import packaging.version
        
        compatible_versions = []
        
        try:
            requirement = packaging.requirements.Requirement(package_requirement)
            
            for version in python_versions:
                parsed_version = packaging.version.parse(version)
                if parsed_version in requirement.specifier:
                    compatible_versions.append(version)
        
        except Exception as e:
            self.logger.warning(f"Version compatibility check failed: {e}")
        
        return compatible_versions
    
    def _generate_optimization_recommendations(
        self, 
        analysis_report: DependencyAnalysisReport
    ) -> List[str]:
        """
        Generate dependency optimization recommendations
        
        Args:
            analysis_report (DependencyAnalysisReport): Comprehensive dependency analysis
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Outdated package recommendations
        if analysis_report.outdated_packages:
            recommendations.append(
                f"Update {len(analysis_report.outdated_packages)} outdated packages"
            )
        
        # Security vulnerability recommendations
        if analysis_report.security_vulnerabilities:
            recommendations.append(
                f"Address {len(analysis_report.security_vulnerabilities)} security vulnerabilities"
            )
        
        # Compatibility issue recommendations
        if analysis_report.compatibility_issues:
            recommendations.append(
                f"Resolve {len(analysis_report.compatibility_issues)} package compatibility issues"
            )
        
        return recommendations
    
    def _persist_analysis_report(self, analysis_report: DependencyAnalysisReport):
        """
        Persist dependency analysis report
        
        Args:
            analysis_report (DependencyAnalysisReport): Comprehensive dependency analysis
        """
        try:
            report_path = os.path.join(
                self.base_dir, 
                'logs', 
                'dependency_management', 
                f'dependency_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(report_path, 'w') as f:
                json.dump(asdict(analysis_report), f, indent=2)
            
            self.logger.info(f"Dependency analysis report persisted: {report_path}")
        
        except Exception as e:
            self.logger.error(f"Dependency analysis report persistence failed: {e}")
    
    def update_dependencies(self, update_type: str = 'minor'):
        """
        Update project dependencies
        
        Args:
            update_type (str): Type of updates to apply (minor/major/patch)
        """
        # Implement dependency update logic
        for req_file in self.config.get('requirements_files', []):
            full_path = os.path.join(self.base_dir, req_file)
            
            try:
                # Use pip-compile or similar tool to update dependencies
                subprocess.run([
                    'pip-compile', 
                    '--upgrade', 
                    f'--upgrade-{update_type}',
                    full_path
                ], check=True)
                
                self.logger.info(f"Updated dependencies in {req_file}")
            
            except subprocess.CalledProcessError:
                self.logger.error(f"Dependency update failed for {req_file}")

def main():
    """
    Execute comprehensive dependency management
    """
    dependency_manager = AdvancedDependencyManager()
    
    # Perform dependency analysis
    analysis_report = dependency_manager.analyze_dependencies()
    
    print("\n🔍 Dependency Analysis Results 🔍")
    
    print(f"\nTotal Dependencies: {analysis_report.total_dependencies}")
    
    print("\nOutdated Packages:")
    for pkg in analysis_report.outdated_packages:
        print(f"- {pkg['name']}: {pkg['current_version']} → {pkg['latest_version']}")
    
    print("\nSecurity Vulnerabilities:")
    for vuln in analysis_report.security_vulnerabilities:
        print(f"- {vuln['package']} (Version {vuln['version']}): {vuln['vulnerability_id']}")
    
    print("\nCompatibility Issues:")
    for issue in analysis_report.compatibility_issues:
        print(f"- {issue['package']}: Incompatible with Python requirement {issue['python_requirement']}")
    
    print("\nOptimization Recommendations:")
    for recommendation in analysis_report.optimization_recommendations:
        print(f"- {recommendation}")

if __name__ == '__main__':
    main()