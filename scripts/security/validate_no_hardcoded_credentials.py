#!/usr/bin/env python3
"""
Security Validation Script: No Hardcoded Credentials
======================================================
Scans the entire codebase for hardcoded passwords, API keys, and secrets.
Part of Rule 5 (Professional Project Standards) compliance.

Author: Security Auditor (Claude Code)
Created: 2025-08-16
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

class CredentialScanner:
    """Scans codebase for hardcoded credentials"""
    
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.violations = []
        self.files_scanned = 0
        self.patterns = [
            # Password patterns
            (r'password\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w@#$%^&*()_+-=]+["\']', "Hardcoded password"),
            (r'PASSWORD\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w@#$%^&*()_+-=]+["\']', "Hardcoded password"),
            (r'passwd\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w@#$%^&*()_+-=]+["\']', "Hardcoded password"),
            
            # API key patterns
            (r'api_key\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w-]+["\']', "Hardcoded API key"),
            (r'API_KEY\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w-]+["\']', "Hardcoded API key"),
            (r'apikey\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w-]+["\']', "Hardcoded API key"),
            
            # Secret/token patterns
            (r'secret\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w-]+["\']', "Hardcoded secret"),
            (r'SECRET\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w-]+["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w-]+["\']', "Hardcoded token"),
            (r'TOKEN\s*=\s*["\'](?![\s"\']*$)(?!\$\{)(?!os\.getenv)[\w-]+["\']', "Hardcoded token"),
            
            # Database connection strings
            (r'(postgres|mysql|mongodb)://[\w]+:[\w]+@', "Hardcoded database credentials in URL"),
            (r'redis://:[^@]+@', "Hardcoded Redis password in URL"),
        ]
        
        # Directories to skip
        self.skip_dirs = {
            '.git', '__pycache__', 'node_modules', '.pytest_cache',
            'venv', 'env', '.venv', 'dist', 'build', 'egg-info',
            'logs', 'data', 'backups', '.claude'
        }
        
        # File extensions to scan
        self.scan_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go',
            '.rb', '.php', '.cs', '.cpp', '.c', '.h', '.hpp',
            '.yml', '.yaml', '.json', '.xml', '.conf', '.cfg',
            '.sh', '.bash', '.zsh', '.env.example'
        }
        
        # Known safe patterns (allowlist)
        self.safe_patterns = [
            'password=""',  # Empty password
            'password=None',  # None value
            "password=''",  # Empty string
            'password=os.getenv',  # Environment variable
            'password=environ',  # Environment variable
            'password=${',  # Template variable
            'password=$(',  # Shell variable
            'password=getpass',  # Interactive input
            'password=input(',  # User input
            'change_me',  # Placeholder
            'your_password',  # Placeholder
            'example_password',  # Example
            'test_password',  # Test placeholder
        ]
    
    def should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned"""
        # Skip if in skip directory
        for parent in file_path.parents:
            if parent.name in self.skip_dirs:
                return False
        
        # Check extension
        return file_path.suffix in self.scan_extensions
    
    def is_safe_pattern(self, line: str) -> bool:
        """Check if line contains a known safe pattern"""
        line_lower = line.lower()
        for safe in self.safe_patterns:
            if safe.lower() in line_lower:
                return True
        return False
    
    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Scan a single file for credentials"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('//'):
                    continue
                
                # Skip safe patterns
                if self.is_safe_pattern(line):
                    continue
                
                # Check against patterns
                for pattern, violation_type in self.patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append((line_num, violation_type, line.strip()))
                        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return violations
    
    def scan_codebase(self) -> Dict:
        """Scan entire codebase for hardcoded credentials"""
        print(f"🔍 Scanning codebase for hardcoded credentials...")
        print(f"Root path: {self.root_path}")
        print("=" * 60)
        
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and self.should_scan_file(file_path):
                self.files_scanned += 1
                
                violations = self.scan_file(file_path)
                if violations:
                    relative_path = file_path.relative_to(self.root_path)
                    for line_num, violation_type, line_content in violations:
                        self.violations.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'type': violation_type,
                            'content': line_content[:100]  # Truncate for security
                        })
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate security scan report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'files_scanned': self.files_scanned,
            'violations_found': len(self.violations),
            'status': 'PASS' if len(self.violations) == 0 else 'FAIL',
            'violations': self.violations
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted report"""
        print("\n" + "=" * 60)
        print("📊 SECURITY SCAN REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Files Scanned: {report['files_scanned']}")
        print(f"Violations Found: {report['violations_found']}")
        print(f"Status: {report['status']}")
        
        if report['violations']:
            print("\n🚨 VIOLATIONS DETECTED:")
            print("-" * 60)
            for v in report['violations']:
                print(f"\n📁 File: {v['file']}")
                print(f"   Line {v['line']}: {v['type']}")
                print(f"   Content: {v['content']}")
        else:
            print("\n✅ NO HARDCODED CREDENTIALS FOUND!")
            print("The codebase is clean and follows security best practices.")
        
        print("\n" + "=" * 60)
        
        if report['status'] == 'PASS':
            print("✅ SECURITY VALIDATION PASSED")
        else:
            print("❌ SECURITY VALIDATION FAILED")
            print("\n⚠️ Action Required:")
            print("1. Replace hardcoded credentials with environment variables")
            print("2. Use os.getenv() or similar secure methods")
            print("3. Never commit credentials to version control")
            print("4. Use .env files (not committed) for local development")
        
        print("=" * 60)

def main():
    """Main execution"""
    scanner = CredentialScanner()
    report = scanner.scan_codebase()
    
    # Save report
    report_path = Path("/opt/sutazaiapp/reports/security_credential_scan.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report
    scanner.print_report(report)
    
    # Exit with appropriate code
    sys.exit(0 if report['status'] == 'PASS' else 1)

if __name__ == "__main__":
    main()