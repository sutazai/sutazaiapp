#!/usr/bin/env python3
"""
SutazAI Advanced Dependency Management and Tracking Framework

Comprehensive utility for:
- Intelligent dependency discovery
- Version compatibility analysis
- Automated dependency resolution
- Security vulnerability tracking
- Performance optimization
"""

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx
import safety
import yaml
from packaging import version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazai_project/SutazAI/logs/advanced_dependency_management.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SutazAI.AdvancedDependencyManager')

class ComprehensiveDependencyManager:
    """
    Ultra-Advanced Dependency Management Framework
    
    Provides intelligent, autonomous dependency tracking,
    security analysis, and optimization capabilities
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        requirements_path: Optional[str] = None
    ):
        """
        Initialize comprehensive dependency manager
        
        Args:
            base_dir (str): Base project directory
            requirements_path (Optional[str]): Path to requirements file
        """
        self.base_dir = base_dir
        self.requirements_path = requirements_path or os.path.join(base_dir, 'requirements.txt')
        self.dependency_cache_dir = os.path.join(base_dir, 'dependency_cache')
        
        # Ensure cache directory exists
        os.makedirs(self.dependency_cache_dir, exist_ok=True)
    
    def discover_dependencies(self) -> Dict[str, Any]:
        """
        Comprehensively discover and analyze project dependencies
        
        Returns:
            Detailed dependency analysis report
        """
        dependencies = {
            'direct_dependencies': {},
            'transitive_dependencies': {},
            'version_compatibility': {},
            'security_analysis': {}
        }
        
        try:
            # Read requirements file
            with open(self.requirements_path, 'r') as f:
                requirements = f.readlines()
            
            for req in requirements:
                req = req.strip()
                if req and not req.startswith('#'):
                    try:
                        # Parse dependency details
                        name, current_version = req.split('==')
                        
                        # Analyze dependency
                        dependency_details = self._analyze_single_dependency(name, current_version)
                        dependencies['direct_dependencies'][name] = dependency_details
                    
                    except Exception as e:
                        logger.warning(f"Could not process dependency {req}: {e}")
            
            # Perform transitive dependency analysis
            dependencies['transitive_dependencies'] = self._discover_transitive_dependencies()
            
            # Perform security vulnerability analysis
            dependencies['security_analysis'] = self._perform_security_analysis()
            
            # Persist dependency report
            self._persist_dependency_report(dependencies)
            
            return dependencies
        
        except FileNotFoundError:
            logger.error(f"Requirements file not found: {self.requirements_path}")
            return dependencies
    
    def _analyze_single_dependency(self, name: str, current_version: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of a single dependency
        
        Args:
            name (str): Dependency name
            current_version (str): Current version
        
        Returns:
            Detailed dependency analysis
        """
        try:
            # Get latest version
            latest_version = self._get_latest_version(name)
            
            # Check version compatibility
            version_compatibility = self._check_version_compatibility(
                current_version, 
                latest_version
            )
            
            return {
                'name': name,
                'current_version': current_version,
                'latest_version': latest_version,
                'needs_update': version.parse(latest_version) > version.parse(current_version),
                'version_compatibility': version_compatibility
            }
        
        except Exception as e:
            logger.warning(f"Dependency analysis failed for {name}: {e}")
            return {
                'name': name,
                'current_version': current_version,
                'analysis_status': 'failed',
                'error': str(e)
            }
    
    def _get_latest_version(self, package_name: str) -> str:
        """
        Retrieve the latest version of a package
        
        Args:
            package_name (str): Name of the package
        
        Returns:
            Latest version string
        """
        try:
            result = subprocess.run(
                ['pip', 'index', 'versions', package_name], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse output to get latest version
            versions = result.stdout.split('\n')
            latest = versions[0].split()[0] if versions else 'Unknown'
            
            return latest
        
        except subprocess.CalledProcessError:
            logger.warning(f"Could not retrieve latest version for {package_name}")
            return 'Unknown'
    
    def _check_version_compatibility(
        self, 
        current_version: str, 
        latest_version: str
    ) -> Dict[str, Any]:
        """
        Analyze version compatibility and potential upgrade risks
        
        Args:
            current_version (str): Current package version
            latest_version (str): Latest available version
        
        Returns:
            Version compatibility analysis
        """
        try:
            current = version.parse(current_version)
            latest = version.parse(latest_version)
            
            compatibility_report = {
                'current_version': current_version,
                'latest_version': latest_version,
                'upgrade_type': self._determine_upgrade_type(current, latest),
                'potential_risks': []
            }
            
            # Identify potential upgrade risks
            if latest > current:
                # Major version differences might indicate breaking changes
                if current.major != latest.major:
                    compatibility_report['potential_risks'].append(
                        'Potential breaking changes in major version upgrade'
                    )
                
                # Pre-release or development versions
                if latest.is_prerelease or latest.is_devrelease:
                    compatibility_report['potential_risks'].append(
                        'Upgrade involves pre-release or development version'
                    )
            
            return compatibility_report
        
        except Exception as e:
            logger.warning(f"Version compatibility check failed: {e}")
            return {
                'current_version': current_version,
                'latest_version': latest_version,
                'compatibility_status': 'unknown',
                'error': str(e)
            }
    
    def _determine_upgrade_type(
        self, 
        current: version.Version, 
        latest: version.Version
    ) -> str:
        """
        Determine the type of version upgrade
        
        Args:
            current (Version): Current package version
            latest (Version): Latest package version
        
        Returns:
            Upgrade type description
        """
        if current.major != latest.major:
            return 'Major Version Upgrade'
        elif current.minor != latest.minor:
            return 'Minor Version Upgrade'
        elif current.micro != latest.micro:
            return 'Patch/Bugfix Upgrade'
        else:
            return 'No Significant Upgrade'
    
    def _discover_transitive_dependencies(self) -> Dict[str, List[str]]:
        """
        Discover and analyze transitive dependencies
        
        Returns:
            Dictionary of transitive dependencies
        """
        try:
            # Use pip to list dependencies
            result = subprocess.run(
                ['pip', 'list', '--format=json'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            installed_packages = json.loads(result.stdout)
            
            # Build dependency graph
            dependency_graph = nx.DiGraph()
            for package in installed_packages:
                dependency_graph.add_node(package['name'], version=package['version'])
            
            # Analyze dependencies
            transitive_deps = {}
            for package in installed_packages:
                try:
                    # Get package requirements
                    req_result = subprocess.run(
                        ['pip', 'show', package['name']], 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                    
                    # Extract required packages
                    required_packages = re.findall(
                        r'Requires: (.+)', 
                        req_result.stdout
                    )
                    
                    if required_packages:
                        transitive_deps[package['name']] = required_packages[0].split(', ')
                        
                        # Add edges to dependency graph
                        for dep in transitive_deps[package['name']]:
                            dependency_graph.add_edge(package['name'], dep)
                
                except Exception as e:
                    logger.warning(f"Could not analyze transitive deps for {package['name']}: {e}")
            
            return transitive_deps
        
        except Exception as e:
            logger.error(f"Transitive dependency discovery failed: {e}")
            return {}
    
    def _perform_security_analysis(self) -> Dict[str, Any]:
        """
        Conduct comprehensive security vulnerability analysis
        
        Returns:
            Security vulnerability report
        """
        try:
            # Use safety to check for known vulnerabilities
            vulnerabilities = safety.check(
                files=[self.requirements_path],
                ignore_ids=[],
                cached=True
            )
            
            security_report = {
                'total_vulnerabilities': len(vulnerabilities),
                'vulnerable_packages': [],
                'severity_breakdown': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                }
            }
            
            for vuln in vulnerabilities:
                vulnerability_details = {
                    'package': vuln[0],
                    'version': vuln[1],
                    'vulnerability_id': vuln[2],
                    'description': vuln[3]
                }
                
                security_report['vulnerable_packages'].append(vulnerability_details)
                
                # Severity classification (placeholder logic)
                if 'critical' in vuln[3].lower():
                    security_report['severity_breakdown']['critical'] += 1
                elif 'high' in vuln[3].lower():
                    security_report['severity_breakdown']['high'] += 1
                elif 'medium' in vuln[3].lower():
                    security_report['severity_breakdown']['medium'] += 1
                else:
                    security_report['severity_breakdown']['low'] += 1
            
            return security_report
        
        except Exception as e:
            logger.error(f"Security vulnerability analysis failed: {e}")
            return {
                'analysis_status': 'failed',
                'error': str(e)
            }
    
    def _persist_dependency_report(self, report: Dict[str, Any]):
        """
        Persist comprehensive dependency report
        
        Args:
            report (Dict): Dependency analysis report
        """
        report_filename = f'dependency_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path = os.path.join(self.dependency_cache_dir, report_filename)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Dependency Report Generated: {report_path}")
        
        except Exception as e:
            logger.error(f"Could not persist dependency report: {e}")
    
    def update_dependencies(self, force: bool = False) -> Dict[str, Any]:
        """
        Intelligently update project dependencies
        
        Args:
            force (bool): Force update all dependencies
        
        Returns:
            Dependency update results
        """
        update_results = {
            'updated_packages': [],
            'failed_updates': [],
            'skipped_packages': []
        }
        
        dependencies = self.discover_dependencies()
        
        for name, details in dependencies['direct_dependencies'].items():
            try:
                # Determine if update is needed
                if force or details.get('needs_update', False):
                    # Check version compatibility
                    compatibility = details.get('version_compatibility', {})
                    potential_risks = compatibility.get('potential_risks', [])
                    
                    # Warn about potential risks
                    if potential_risks:
                        logger.warning(f"Potential risks for {name}: {potential_risks}")
                    
                    # Perform update
                    subprocess.run(
                        ['pip', 'install', '--upgrade', f'{name}=={details["latest_version"]}'],
                        check=True,
                        capture_output=True
                    )
                    
                    update_results['updated_packages'].append({
                        'name': name,
                        'old_version': details['current_version'],
                        'new_version': details['latest_version'],
                        'risks': potential_risks
                    })
                else:
                    update_results['skipped_packages'].append(name)
            
            except subprocess.CalledProcessError as e:
                update_results['failed_updates'].append({
                    'name': name,
                    'version': details['latest_version'],
                    'error': str(e)
                })
        
        # Update requirements file
        self._update_requirements_file(update_results)
        
        return update_results
    
    def _update_requirements_file(self, update_results: Dict[str, Any]):
        """
        Update requirements file with new package versions
        
        Args:
            update_results (Dict): Dependency update results
        """
        try:
            with open(self.requirements_path, 'r') as f:
                requirements = f.readlines()
            
            # Update versions for successfully updated packages
            updated_requirements = []
            for line in requirements:
                line = line.strip()
                if line and not line.startswith('#'):
                    package_name = line.split('==')[0]
                    
                    # Find matching updated package
                    updated_package = next(
                        (pkg for pkg in update_results.get('updated_packages', []) 
                         if pkg['name'] == package_name), 
                        None
                    )
                    
                    if updated_package:
                        updated_requirements.append(
                            f"{package_name}=={updated_package['new_version']}\n"
                        )
                    else:
                        updated_requirements.append(line + '\n')
                else:
                    updated_requirements.append(line + '\n')
            
            # Write updated requirements
            with open(self.requirements_path, 'w') as f:
                f.writelines(updated_requirements)
            
            logger.info("Requirements file updated successfully")
        
        except Exception as e:
            logger.error(f"Could not update requirements file: {e}")

def main():
    """
    Main execution for advanced dependency management
    """
    try:
        dependency_manager = ComprehensiveDependencyManager()
        
        # Discover and analyze dependencies
        dependencies = dependency_manager.discover_dependencies()
        print("Dependency Analysis:")
        print(json.dumps(dependencies, indent=2))
        
        # Update dependencies
        update_results = dependency_manager.update_dependencies()
        print("\nDependency Update Results:")
        print(json.dumps(update_results, indent=2))
    
    except Exception as e:
        logger.error(f"Dependency management failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 