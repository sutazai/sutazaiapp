#!/usr/bin/env python3
"""
Dependency Analysis Tool for SutazAI System
Analyzes Python dependencies, Docker images, and environment variables
"""

import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyAnalyzer:
    """Comprehensive dependency analyzer for SutazAI"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.results = {
            "python_dependencies": {},
            "version_conflicts": [],
            "security_vulnerabilities": [],
            "docker_images": [],
            "environment_variables": {},
            "recommendations": []
        }
        
        # Known security vulnerabilities and minimum safe versions
        self.security_baselines = {
            'urllib3': '2.2.2',
            'pillow': '10.4.0', 
            'cryptography': '42.0.8',
            'jinja2': '3.1.4',
            'requests': '2.32.0',
            'transformers': '4.42.0',
            'torch': '2.3.1',
            'langchain': '0.2.6',
            'aiohttp': '3.9.5',
            'fastapi': '0.111.0',
            'pydantic': '2.8.0',
            'sqlalchemy': '2.0.31',
            'werkzeug': '3.0.3',
            'flask': '3.0.3',
            'django': '4.2.14',
            'lxml': '5.2.2',
            'streamlit': '1.36.0',
            'redis': '5.0.7',
            'psycopg2-binary': '2.9.9',
            'celery': '5.4.0',
            'beautifulsoup4': '4.12.3',
            'selenium': '4.21.0',
            'playwright': '1.45.0'
        }
    
    def parse_requirements_file(self, file_path: Path) -> Dict[str, Tuple[str, str]]:
        """Parse requirements.txt file and extract package versions"""
        requirements = {}
        
        if not file_path.exists():
            return requirements
            
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Match package==version, package>=version, etc.
                        match = re.match(r'^([a-zA-Z0-9_-]+)([><=!]+)([0-9.]+)', line)
                        if match:
                            pkg, op, ver = match.groups()
                            requirements[pkg.lower()] = (op, ver, line_num)
                        else:
                            # Handle packages without version specifiers
                            simple_match = re.match(r'^([a-zA-Z0-9_-]+)$', line)
                            if simple_match:
                                pkg = simple_match.group(1)
                                requirements[pkg.lower()] = ('', 'latest', line_num)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            
        return requirements
    
    def analyze_python_dependencies(self):
        """Analyze all Python dependency files"""
        logger.info("Analyzing Python dependencies...")
        
        # Find all requirements files
        req_files = []
        for pattern in ['requirements*.txt', '**/requirements*.txt']:
            req_files.extend(self.project_root.glob(pattern))
        
        all_requirements = {}
        file_requirements = {}
        
        for req_file in req_files:
            rel_path = str(req_file.relative_to(self.project_root))
            reqs = self.parse_requirements_file(req_file)
            file_requirements[rel_path] = reqs
            
            # Track all requirements across files
            for pkg, (op, ver, line_num) in reqs.items():
                if pkg not in all_requirements:
                    all_requirements[pkg] = []
                all_requirements[pkg].append({
                    'file': rel_path,
                    'operator': op,
                    'version': ver,
                    'line': line_num
                })
        
        self.results["python_dependencies"] = file_requirements
        
        # Find version conflicts
        self.find_version_conflicts(all_requirements)
        
        # Check security vulnerabilities
        self.check_security_vulnerabilities(all_requirements)
    
    def find_version_conflicts(self, all_requirements: Dict):
        """Find version conflicts between requirement files"""
        logger.info("Checking for version conflicts...")
        
        for pkg, occurrences in all_requirements.items():
            if len(occurrences) > 1:
                versions = set()
                for occ in occurrences:
                    if occ['version'] != 'latest':
                        versions.add(occ['version'])
                
                if len(versions) > 1:
                    self.results["version_conflicts"].append({
                        'package': pkg,
                        'occurrences': occurrences,
                        'versions': list(versions),
                        'severity': 'high' if pkg in ['fastapi', 'pydantic', 'sqlalchemy'] else 'medium'
                    })
    
    def check_security_vulnerabilities(self, all_requirements: Dict):
        """Check for known security vulnerabilities"""
        logger.info("Checking for security vulnerabilities...")
        
        for pkg, baseline_version in self.security_baselines.items():
            if pkg in all_requirements:
                for occ in all_requirements[pkg]:
                    if occ['version'] != 'latest':
                        try:
                            from packaging import version
                            if version.parse(occ['version']) < version.parse(baseline_version):
                                self.results["security_vulnerabilities"].append({
                                    'package': pkg,
                                    'current_version': occ['version'],
                                    'secure_version': baseline_version,
                                    'file': occ['file'],
                                    'line': occ['line'],
                                    'severity': 'critical' if pkg in ['cryptography', 'urllib3', 'requests'] else 'high'
                                })
                        except Exception as e:
                            logger.warning(f"Could not parse version for {pkg}: {e}")
    
    def analyze_docker_images(self):
        """Analyze Docker base images and versions"""
        logger.info("Analyzing Docker images...")
        
        dockerfile_patterns = ['**/Dockerfile*', 'Dockerfile*']
        dockerfiles = []
        
        for pattern in dockerfile_patterns:
            dockerfiles.extend(self.project_root.glob(pattern))
        
        for dockerfile in dockerfiles:
            rel_path = str(dockerfile.relative_to(self.project_root))
            
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    
                # Find FROM statements
                from_statements = re.findall(r'^FROM\s+([^\s]+)', content, re.MULTILINE)
                
                for base_image in from_statements:
                    # Skip multi-stage build references
                    if not base_image.startswith('python') and not base_image.startswith('node') and not any(c in base_image for c in [':', '/', '@']):
                        continue
                        
                    image_info = {
                        'file': rel_path,
                        'base_image': base_image,
                        'needs_update': False,
                        'security_issues': []
                    }
                    
                    # Check for known outdated or vulnerable images
                    if 'python:3.11-slim' in base_image:
                        # Check if newer version available
                        image_info['recommendation'] = 'Consider python:3.12-slim for latest security patches'
                    elif 'node:' in base_image and not any(v in base_image for v in ['20', '21']):
                        image_info['needs_update'] = True
                        image_info['security_issues'].append('Outdated Node.js version')
                    elif ':latest' in base_image:
                        image_info['security_issues'].append('Using :latest tag is not recommended for production')
                    
                    self.results["docker_images"].append(image_info)
                    
            except Exception as e:
                logger.error(f"Error analyzing {dockerfile}: {e}")
    
    def analyze_environment_variables(self):
        """Analyze environment variable configurations"""
        logger.info("Analyzing environment variables...")
        
        env_files = ['.env', '.env.example', '.env.production', '.env.ollama']
        
        for env_file in env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                try:
                    with open(env_path, 'r') as f:
                        content = f.read()
                        
                    # Find all environment variables
                    env_vars = re.findall(r'^([A-Z_]+)=(.*)$', content, re.MULTILINE)
                    
                    file_vars = {}
                    security_issues = []
                    
                    for var_name, var_value in env_vars:
                        file_vars[var_name] = var_value
                        
                        # Check for security issues
                        if 'password' in var_name.lower() or 'secret' in var_name.lower() or 'key' in var_name.lower():
                            if var_value in ['', 'your-secret-key-here', 'secure-password-here', 'redis-password-here']:
                                security_issues.append({
                                    'variable': var_name,
                                    'issue': 'Default or empty security credential',
                                    'severity': 'critical'
                                })
                    
                    self.results["environment_variables"][env_file] = {
                        'variables': file_vars,
                        'security_issues': security_issues
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing {env_path}: {e}")
    
    def analyze_javascript_dependencies(self):
        """Analyze JavaScript/Node.js dependencies"""
        logger.info("Analyzing JavaScript dependencies...")
        
        # Find package.json files
        package_files = list(self.project_root.glob('**/package.json'))
        
        for package_file in package_files:
            rel_path = str(package_file.relative_to(self.project_root))
            
            try:
                with open(package_file, 'r') as f:
                    package_data = json.load(f)
                
                dependencies = package_data.get('dependencies', {})
                dev_dependencies = package_data.get('devDependencies', {})
                
                # Check for vulnerable packages (simplified check)
                vulnerable_packages = {
                    'express': '4.18.0',
                    'axios': '1.6.0', 
                    'lodash': '4.17.21',
                    'moment': '2.29.4'  # moment.js is deprecated
                }
                
                for pkg, min_version in vulnerable_packages.items():
                    if pkg in dependencies:
                        current_version = dependencies[pkg].lstrip('^~')
                        # Simple version comparison (would need more robust parsing in production)
                        if pkg == 'moment':
                            self.results["recommendations"].append({
                                'type': 'javascript_deprecation',
                                'file': rel_path,
                                'package': pkg,
                                'message': 'moment.js is deprecated, consider migrating to date-fns or dayjs'
                            })
                
            except Exception as e:
                logger.error(f"Error analyzing {package_file}: {e}")
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        logger.info("Generating recommendations...")
        
        # Security vulnerabilities recommendations
        if self.results["security_vulnerabilities"]:
            self.results["recommendations"].append({
                'type': 'security_critical',
                'priority': 'high',
                'message': f'Found {len(self.results["security_vulnerabilities"])} security vulnerabilities that need immediate attention'
            })
        
        # Version conflicts recommendations
        if self.results["version_conflicts"]:
            high_priority_conflicts = [c for c in self.results["version_conflicts"] if c['severity'] == 'high']
            if high_priority_conflicts:
                self.results["recommendations"].append({
                    'type': 'version_conflicts',
                    'priority': 'high', 
                    'message': f'Found {len(high_priority_conflicts)} high-priority version conflicts in core packages'
                })
        
        # Environment security recommendations
        env_issues = []
        for env_file, data in self.results["environment_variables"].items():
            env_issues.extend(data.get('security_issues', []))
        
        if env_issues:
            critical_issues = [i for i in env_issues if i['severity'] == 'critical']
            if critical_issues:
                self.results["recommendations"].append({
                    'type': 'environment_security',
                    'priority': 'critical',
                    'message': f'Found {len(critical_issues)} critical environment security issues (default passwords/keys)'
                })
    
    def run_analysis(self):
        """Run complete dependency analysis"""
        logger.info("Starting comprehensive dependency analysis...")
        
        self.analyze_python_dependencies()
        self.analyze_docker_images()
        self.analyze_environment_variables()
        self.analyze_javascript_dependencies()
        self.generate_recommendations()
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate human-readable report"""
        report = []
        report.append("="*80)
        report.append("SUTAZAI DEPENDENCY ANALYSIS REPORT")
        report.append("="*80)
        
        # Summary
        total_conflicts = len(self.results["version_conflicts"])
        total_vulnerabilities = len(self.results["security_vulnerabilities"])
        total_recommendations = len(self.results["recommendations"])
        
        report.append(f"\nSUMMARY:")
        report.append(f"  - Version Conflicts: {total_conflicts}")
        report.append(f"  - Security Vulnerabilities: {total_vulnerabilities}")
        report.append(f"  - Total Recommendations: {total_recommendations}")
        
        # Critical Issues
        critical_issues = []
        for vuln in self.results["security_vulnerabilities"]:
            if vuln['severity'] == 'critical':
                critical_issues.append(f"  - {vuln['package']}: {vuln['current_version']} → {vuln['secure_version']} (in {vuln['file']})")
        
        if critical_issues:
            report.append(f"\nCRITICAL SECURITY VULNERABILITIES:")
            report.extend(critical_issues)
        
        # High Priority Version Conflicts
        high_conflicts = [c for c in self.results["version_conflicts"] if c['severity'] == 'high']
        if high_conflicts:
            report.append(f"\nHIGH PRIORITY VERSION CONFLICTS:")
            for conflict in high_conflicts:
                report.append(f"  - {conflict['package']}: {', '.join(conflict['versions'])}")
                for occ in conflict['occurrences']:
                    report.append(f"    * {occ['file']}: {occ['operator']}{occ['version']}")
        
        # Environment Issues
        env_issues = []
        for env_file, data in self.results["environment_variables"].items():
            for issue in data.get('security_issues', []):
                if issue['severity'] == 'critical':
                    env_issues.append(f"  - {env_file}: {issue['variable']} - {issue['issue']}")
        
        if env_issues:
            report.append(f"\nCRITICAL ENVIRONMENT ISSUES:")
            report.extend(env_issues)
        
        # Immediate Actions Required
        report.append(f"\nIMMEDIATE ACTIONS REQUIRED:")
        
        if total_vulnerabilities > 0:
            report.append("  1. Update vulnerable Python packages:")
            for vuln in self.results["security_vulnerabilities"][:5]:  # Show top 5
                report.append(f"     pip install {vuln['package']}>={vuln['secure_version']}")
        
        if env_issues:
            report.append("  2. Generate secure environment variables:")
            report.append("     openssl rand -hex 32  # For SECRET_KEY")
            report.append("     openssl rand -base64 32  # For passwords")
        
        if total_conflicts > 0:
            report.append("  3. Resolve version conflicts by standardizing versions across all requirements files")
        
        report.append("="*80)
        
        return "\n".join(report)

def main():
    """Main execution function"""
    analyzer = DependencyAnalyzer()
    results = analyzer.run_analysis()
    
    # Print report
    print(analyzer.generate_report())
    
    # Save detailed results
    output_file = Path("/opt/sutazaiapp/logs/dependency_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()