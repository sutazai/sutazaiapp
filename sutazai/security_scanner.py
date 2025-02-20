import os
import sys
import json
import logging
import subprocess
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

@dataclass
class SecurityScanResult:
    """
    Structured security scan result for detailed reporting.
    """
    vulnerability_count: int = 0
    critical_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    high_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    medium_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    low_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    scan_timestamp: str = ''
    scan_duration: float = 0.0

class SecurityScanner:
    """
    Advanced security scanning framework for SutazAI.
    Provides comprehensive vulnerability detection and reporting.
    """

    def __init__(
        self, 
        project_root: str = '.', 
        log_dir: str = 'logs/security'
    ):
        """
        Initialize security scanner.

        Args:
            project_root (str): Root directory of the project
            log_dir (str): Directory to store security scan logs
        """
        self.project_root = os.path.abspath(project_root)
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger('SecurityScanner')

    def run_bandit_scan(self) -> List[Dict[str, Any]]:
        """
        Run Bandit security scanner for Python code.

        Returns:
            List[Dict[str, Any]]: Bandit security scan results
        """
        try:
            bandit_output = subprocess.run(
                [
                    sys.executable, 
                    '-m', 
                    'bandit', 
                    '-r', 
                    self.project_root, 
                    '-f', 
                    'json'
                ],
                capture_output=True,
                text=True,
                check=False
            )
            return json.loads(bandit_output.stdout)['results']
        except Exception as e:
            self.logger.error(f"Bandit scan failed: {e}")
            return []

    def run_safety_scan(self) -> List[Dict[str, Any]]:
        """
        Run Safety vulnerability scanner for Python dependencies.

        Returns:
            List[Dict[str, Any]]: Safety vulnerability scan results
        """
        try:
            safety_output = subprocess.run(
                [
                    sys.executable, 
                    '-m', 
                    'safety', 
                    'check', 
                    '--full-report', 
                    '--json'
                ],
                capture_output=True,
                text=True,
                check=False
            )
            return json.loads(safety_output.stdout)
        except Exception as e:
            self.logger.error(f"Safety scan failed: {e}")
            return []

    def scan_file_permissions(self) -> List[Dict[str, Any]]:
        """
        Scan file and directory permissions for security risks.

        Returns:
            List[Dict[str, Any]]: File permission scan results
        """
        permission_issues = []
        for root, _, files in os.walk(self.project_root):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_stat = os.stat(file_path)
                    mode = file_stat.st_mode

                    # Check for overly permissive files
                    if (mode & 0o777) > 0o644:  # More permissive than read/write for owner
                        permission_issues.append({
                            'path': file_path,
                            'permissions': oct(mode & 0o777),
                            'issue': 'Overly permissive file permissions'
                        })
                except Exception as e:
                    self.logger.warning(f"Could not check permissions for {file_path}: {e}")

        return permission_issues

    def analyze_network_ports(self) -> List[Dict[str, Any]]:
        """
        Analyze open network ports for potential security risks.

        Returns:
            List[Dict[str, Any]]: Network port scan results
        """
        try:
            netstat_output = subprocess.run(
                ['netstat', '-tuln'],
                capture_output=True,
                text=True,
                check=True
            )
            
            port_pattern = re.compile(r'tcp\s+\d+\s+\d+\s+[0-9.]+:(\d+)\s+[0-9.]+:\*\s+LISTEN')
            open_ports = port_pattern.findall(netstat_output.stdout)

            return [
                {
                    'port': port,
                    'risk_level': self._assess_port_risk(int(port))
                }
                for port in open_ports
            ]
        except Exception as e:
            self.logger.error(f"Network port scan failed: {e}")
            return []

    def _assess_port_risk(self, port: int) -> str:
        """
        Assess the risk level of an open port.

        Args:
            port (int): Port number to assess

        Returns:
            str: Risk level ('low', 'medium', 'high')
        """
        high_risk_ports = {
            22, 23, 3389,  # SSH, Telnet, RDP
            80, 443,       # HTTP/HTTPS
            21, 20,        # FTP
            25, 587, 465,  # SMTP
            3306, 5432,    # MySQL, PostgreSQL
        }

        if port in high_risk_ports:
            return 'high'
        elif 1024 < port < 49151:
            return 'medium'
        else:
            return 'low'

    def generate_security_report(self) -> SecurityScanResult:
        """
        Generate a comprehensive security scan report.

        Returns:
            SecurityScanResult: Detailed security scan results
        """
        import time
        start_time = time.time()

        # Perform various security scans
        bandit_results = self.run_bandit_scan()
        safety_results = self.run_safety_scan()
        permission_issues = self.scan_file_permissions()
        network_port_risks = self.analyze_network_ports()

        # Categorize vulnerabilities
        scan_result = SecurityScanResult(
            vulnerability_count=len(bandit_results) + len(safety_results),
            scan_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            scan_duration=time.time() - start_time
        )

        # Categorize Bandit vulnerabilities
        for vuln in bandit_results:
            severity = vuln.get('issue_severity', 'low').lower()
            vuln_entry = {
                'file': vuln.get('filename', 'Unknown'),
                'line': vuln.get('line_number', 'Unknown'),
                'code': vuln.get('code', 'Unknown'),
                'description': vuln.get('issue_text', 'No description')
            }

            if severity == 'high':
                scan_result.high_vulnerabilities.append(vuln_entry)
            elif severity == 'medium':
                scan_result.medium_vulnerabilities.append(vuln_entry)
            else:
                scan_result.low_vulnerabilities.append(vuln_entry)

        # Categorize Safety vulnerabilities
        for vuln in safety_results:
            # Implement safety vulnerability categorization logic
            pass

        # Add file permission and network port risks
        scan_result.low_vulnerabilities.extend(permission_issues)
        scan_result.medium_vulnerabilities.extend(network_port_risks)

        # Generate report file
        report_path = os.path.join(
            self.log_dir, 
            f'security_report_{time.strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(asdict(scan_result), f, indent=2)

        self.logger.info(f"Security report generated: {report_path}")
        return scan_result

def main():
    scanner = SecurityScanner()
    report = scanner.generate_security_report()
    print(f"Total Vulnerabilities: {report.vulnerability_count}")

if __name__ == '__main__':
    main() 