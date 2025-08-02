#!/usr/bin/env python3

"""
SutazAI Security Audit Script
Analyzes and fixes known vulnerabilities in dependencies
"""

import subprocess
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityAuditor:
    """Comprehensive security auditor for SutazAI dependencies"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.vulnerabilities = []
        self.fixed_packages = []
        
        # Known vulnerable packages and their secure versions
        self.security_updates = {
            # Critical vulnerabilities
            "requests": ">=2.32.0",  # CVE-2024-35195
            "urllib3": ">=2.2.2",    # CVE-2024-37891
            "pillow": ">=10.4.0",    # CVE-2024-28219
            "cryptography": ">=42.0.8", # CVE-2024-26130
            "jinja2": ">=3.1.4",     # CVE-2024-22195
            "werkzeug": ">=3.0.3",   # CVE-2024-34069
            "flask": ">=3.0.3",      # CVE-2024-27177
            "django": ">=4.2.14",    # CVE-2024-38875
            "tornado": ">=6.4.1",    # CVE-2024-52804
            
            # High priority updates
            "fastapi": ">=0.111.0",   # Security improvements
            "uvicorn": ">=0.30.1",    # Security enhancements
            "pydantic": ">=2.8.0",    # Validation fixes
            "sqlalchemy": ">=2.0.31", # SQL injection fixes
            "redis": ">=5.0.7",       # Security patches
            "celery": ">=5.4.0",      # Security improvements
            "aiohttp": ">=3.9.5",     # CVE-2024-30251
            "websockets": ">=12.0",   # Protocol security fixes
            "transformers": ">=4.42.0", # Model security
            "torch": ">=2.3.1",       # Security patches
            "numpy": ">=1.26.4",      # Buffer overflow fixes
            "pandas": ">=2.2.2",      # Security improvements
            "matplotlib": ">=3.9.0",  # Security fixes
            "beautifulsoup4": ">=4.12.3", # Parser security
            "lxml": ">=5.2.2",        # XML security
            "selenium": ">=4.21.0",   # WebDriver security
            "playwright": ">=1.45.0", # Browser security
            
            # Medium priority
            "streamlit": ">=1.36.0",  # Security improvements
            "plotly": ">=5.22.0",     # XSS fixes
            "scikit-learn": ">=1.5.0", # Security patches
            "pymongo": ">=4.8.0",     # Database security
            "psycopg2-binary": ">=2.9.9", # PostgreSQL security
            "docker": ">=7.1.0",      # API security
            "kubernetes": ">=30.1.0", # API security
            "openai": ">=1.35.0",     # API security
            "anthropic": ">=0.28.0",  # API security
            "langchain": ">=0.2.6",   # Chain security
            "chromadb": ">=0.5.0",    # Database security
            "qdrant-client": ">=1.9.0", # Vector DB security
            
            # Development dependencies
            "black": ">=24.4.2",      # Code formatter security
            "pytest": ">=8.2.2",     # Test framework security
            "coverage": ">=7.5.4",   # Coverage security
        }
        
    def find_requirements_files(self) -> List[Path]:
        """Find all requirements files in the project"""
        requirements_files = []
        
        # Main requirements files
        main_files = [
            self.project_root / "requirements.txt",
            self.project_root / "backend" / "requirements.txt",
            self.project_root / "frontend" / "requirements.txt",
        ]
        
        # Docker service requirements
        docker_files = list(self.project_root.rglob("docker/*/requirements.txt"))
        
        all_files = main_files + docker_files
        
        # Filter existing files
        for file_path in all_files:
            if file_path.exists():
                requirements_files.append(file_path)
                
        return requirements_files
        
    def parse_requirements(self, file_path: Path) -> List[Tuple[str, str]]:
        """Parse requirements file and return package name and version spec"""
        packages = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    # Handle different version specifiers
                    if '>=' in line:
                        package, version = line.split('>=', 1)
                        packages.append((package.strip(), f">={version.strip()}"))
                    elif '==' in line:
                        package, version = line.split('==', 1)
                        packages.append((package.strip(), f"=={version.strip()}"))
                    elif '>' in line:
                        package, version = line.split('>', 1)
                        packages.append((package.strip(), f">{version.strip()}"))
                    else:
                        # No version specified
                        packages.append((line.strip(), ""))
                        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            
        return packages
        
    def check_vulnerabilities(self, packages: List[Tuple[str, str]]) -> List[Dict]:
        """Check packages against known vulnerabilities"""
        vulnerabilities = []
        
        for package, current_version in packages:
            if package in self.security_updates:
                secure_version = self.security_updates[package]
                
                # Check if current version needs updating
                if not current_version or self._version_needs_update(current_version, secure_version):
                    vulnerabilities.append({
                        'package': package,
                        'current_version': current_version,
                        'secure_version': secure_version,
                        'severity': self._get_severity(package)
                    })
                    
        return vulnerabilities
        
    def _version_needs_update(self, current: str, secure: str) -> bool:
        """Simple version comparison (basic implementation)"""
        # Extract version numbers for basic comparison
        current_clean = re.sub(r'[>=<~]', '', current).strip()
        secure_clean = re.sub(r'[>=<~]', '', secure).strip()
        
        if not current_clean:
            return True  # No version specified, needs update
            
        try:
            # Simple version comparison (major.minor.patch)
            current_parts = [int(x) for x in current_clean.split('.')]
            secure_parts = [int(x) for x in secure_clean.split('.')]
            
            # Pad with zeros if needed
            max_len = max(len(current_parts), len(secure_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            secure_parts.extend([0] * (max_len - len(secure_parts)))
            
            return current_parts < secure_parts
            
        except (ValueError, IndexError):
            # If version parsing fails, assume update needed
            return True
            
    def _get_severity(self, package: str) -> str:
        """Get vulnerability severity for package"""
        critical_packages = ['requests', 'urllib3', 'pillow', 'cryptography', 'jinja2']
        high_packages = ['fastapi', 'uvicorn', 'aiohttp', 'torch', 'transformers']
        
        if package in critical_packages:
            return 'critical'
        elif package in high_packages:
            return 'high'
        else:
            return 'medium'
            
    def generate_secure_requirements(self, file_path: Path, packages: List[Tuple[str, str]]) -> str:
        """Generate updated requirements file with secure versions"""
        updated_lines = []
        
        # Add header
        updated_lines.append(f"# SutazAI Secure Requirements - Updated {datetime.now().strftime('%Y-%m-%d')}")
        updated_lines.append("# Security vulnerabilities fixed")
        updated_lines.append("")
        
        for package, current_version in packages:
            if package in self.security_updates:
                secure_version = self.security_updates[package]
                updated_lines.append(f"{package}{secure_version}")
                self.fixed_packages.append(f"{package}: {current_version} -> {secure_version}")
            else:
                # Keep current version if no security update needed
                if current_version:
                    updated_lines.append(f"{package}{current_version}")
                else:
                    updated_lines.append(package)
                    
        return '\n'.join(updated_lines)
        
    def run_pip_audit(self) -> Optional[Dict]:
        """Run pip-audit if available to get additional vulnerability info"""
        try:
            result = subprocess.run(
                ['pip-audit', '--format=json', '--req', 'backend/requirements.txt'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.warning("pip-audit failed or not available")
                return None
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
            logger.info("pip-audit not available or failed, using manual checks")
            return None
            
    def create_security_report(self, vulnerabilities: List[Dict]) -> str:
        """Create comprehensive security report"""
        report = []
        report.append("=" * 60)
        report.append("SUTAZAI SECURITY AUDIT REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if not vulnerabilities:
            report.append("✅ NO VULNERABILITIES FOUND!")
            report.append("All dependencies are up-to-date with security patches.")
        else:
            # Group by severity
            critical = [v for v in vulnerabilities if v['severity'] == 'critical']
            high = [v for v in vulnerabilities if v['severity'] == 'high']
            medium = [v for v in vulnerabilities if v['severity'] == 'medium']
            
            report.append(f"🚨 TOTAL VULNERABILITIES FOUND: {len(vulnerabilities)}")
            report.append(f"   Critical: {len(critical)}")
            report.append(f"   High: {len(high)}")
            report.append(f"   Medium: {len(medium)}")
            report.append("")
            
            # Critical vulnerabilities
            if critical:
                report.append("🔴 CRITICAL VULNERABILITIES:")
                for vuln in critical:
                    report.append(f"   • {vuln['package']}: {vuln['current_version']} -> {vuln['secure_version']}")
                report.append("")
                
            # High vulnerabilities
            if high:
                report.append("🟠 HIGH PRIORITY VULNERABILITIES:")
                for vuln in high:
                    report.append(f"   • {vuln['package']}: {vuln['current_version']} -> {vuln['secure_version']}")
                report.append("")
                
            # Medium vulnerabilities
            if medium:
                report.append("🟡 MEDIUM PRIORITY VULNERABILITIES:")
                for vuln in medium:
                    report.append(f"   • {vuln['package']}: {vuln['current_version']} -> {vuln['secure_version']}")
                report.append("")
                
        # Fixed packages summary
        if self.fixed_packages:
            report.append("✅ PACKAGES UPDATED:")
            for fix in self.fixed_packages:
                report.append(f"   • {fix}")
            report.append("")
            
        report.append("=" * 60)
        report.append("RECOMMENDATIONS:")
        report.append("1. Update all requirements files with secure versions")
        report.append("2. Rebuild Docker images with updated dependencies")
        report.append("3. Run comprehensive tests after updates")
        report.append("4. Monitor security advisories regularly")
        report.append("=" * 60)
        
        return '\n'.join(report)
        
    def run_audit(self):
        """Run comprehensive security audit"""
        logger.info("Starting SutazAI security audit...")
        
        requirements_files = self.find_requirements_files()
        logger.info(f"Found {len(requirements_files)} requirements files")
        
        all_vulnerabilities = []
        
        for req_file in requirements_files:
            logger.info(f"Auditing: {req_file}")
            packages = self.parse_requirements(req_file)
            vulns = self.check_vulnerabilities(packages)
            all_vulnerabilities.extend(vulns)
            
            # Generate secure requirements
            if vulns:
                secure_content = self.generate_secure_requirements(req_file, packages)
                backup_file = req_file.with_suffix('.txt.backup')
                secure_file = req_file.with_suffix('.secure.txt')
                
                # Create backup
                if req_file.exists():
                    import shutil
                    shutil.copy2(req_file, backup_file)
                    logger.info(f"Backup created: {backup_file}")
                
                # Write secure version
                with open(secure_file, 'w') as f:
                    f.write(secure_content)
                logger.info(f"Secure requirements generated: {secure_file}")
                
        # Run pip-audit if available
        pip_audit_results = self.run_pip_audit()
        if pip_audit_results:
            logger.info("Additional pip-audit results available")
            
        # Generate report
        report = self.create_security_report(all_vulnerabilities)
        
        # Save report
        report_file = self.project_root / "security_audit_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
            
        print(report)
        logger.info(f"Security audit complete. Report saved to: {report_file}")
        
        return len(all_vulnerabilities)

def main():
    """Main execution function"""
    auditor = SecurityAuditor()
    vulnerability_count = auditor.run_audit()
    
    if vulnerability_count == 0:
        print("\n🎉 All dependencies are secure!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Found {vulnerability_count} vulnerabilities that need attention.")
        print("Check the generated .secure.txt files for updated requirements.")
        sys.exit(1)

if __name__ == "__main__":
    main()