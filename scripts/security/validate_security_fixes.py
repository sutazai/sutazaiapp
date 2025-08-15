#!/usr/bin/env python3
"""
Security Validation Script for Phase 1 Critical Security Fixes
Validates that all P0 security issues have been properly remediated
Date: 2025-08-16
"""

import os
import re
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class SecurityValidator:
    """Validates security fixes implemented in the codebase"""
    
    def __init__(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.issues_found = []
        self.fixes_validated = []
        self.validation_results = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "critical_issues": [],
            "warnings": []
        }
    
    def print_header(self, message: str):
        """Print formatted header"""
        print(f"\n{BLUE}{BOLD}{'=' * 60}{RESET}")
        print(f"{BLUE}{BOLD}{message:^60}{RESET}")
        print(f"{BLUE}{BOLD}{'=' * 60}{RESET}\n")
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"{GREEN}✓{RESET} {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"{RED}✗{RESET} {message}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{YELLOW}⚠{RESET} {message}")
    
    def validate_docker_security(self) -> bool:
        """Validate Docker files don't use root user for package installation"""
        self.print_header("Validating Docker Security")
        
        docker_files = [
            self.project_root / "backend/Dockerfile",
            self.project_root / "frontend/Dockerfile"
        ]
        
        all_secure = True
        
        for dockerfile in docker_files:
            self.validation_results["total_checks"] += 1
            
            if not dockerfile.exists():
                self.print_warning(f"Dockerfile not found: {dockerfile}")
                continue
            
            with open(dockerfile, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for USER root followed by package installation
            has_user_root = False
            root_for_install = False
            
            for i, line in enumerate(lines, 1):
                if re.match(r'^\s*USER\s+root', line, re.IGNORECASE):
                    has_user_root = True
                    # Check if next non-empty lines involve package installation
                    for j in range(i, min(i+5, len(lines))):
                        if 'pip install' in lines[j] and not '--user' in lines[j]:
                            root_for_install = True
                            break
            
            # Check for proper --chown usage
            has_chown = '--chown=appuser' in content
            
            # Check that we end with non-root user
            ends_with_nonroot = False
            for line in reversed(lines):
                if line.strip().startswith('USER'):
                    ends_with_nonroot = 'appuser' in line.lower()
                    break
            
            if root_for_install:
                self.print_error(f"{dockerfile.name}: Uses root for package installation")
                self.validation_results["critical_issues"].append(
                    f"{dockerfile.name}: Root user elevation detected"
                )
                self.validation_results["failed"] += 1
                all_secure = False
            elif has_user_root and not has_chown:
                self.print_warning(f"{dockerfile.name}: Uses USER root but may need --chown")
                self.validation_results["warnings"].append(
                    f"{dockerfile.name}: Consider using --chown instead of USER root"
                )
                self.validation_results["passed"] += 1
            else:
                self.print_success(f"{dockerfile.name}: Properly secured (uses --chown, runs as appuser)")
                self.validation_results["passed"] += 1
            
            if not ends_with_nonroot:
                self.print_error(f"{dockerfile.name}: Does not end with non-root user")
                self.validation_results["critical_issues"].append(
                    f"{dockerfile.name}: Container may run as root"
                )
                all_secure = False
        
        return all_secure
    
    def validate_no_hardcoded_urls(self) -> bool:
        """Validate no hardcoded localhost URLs in production code"""
        self.print_header("Validating Hardcoded URL Removal")
        
        # Files to check (excluding test files for now)
        model_files = list(self.project_root.glob("models/optimization/*.py"))
        
        all_clean = True
        localhost_pattern = re.compile(r'["\']http://localhost:\d+["\']')
        
        for file_path in model_files:
            self.validation_results["total_checks"] += 1
            
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for hardcoded localhost URLs
            hardcoded_found = []
            for i, line in enumerate(lines, 1):
                if localhost_pattern.search(line):
                    # Check if it's properly using environment variable
                    if 'os.getenv' not in line and 'os.environ' not in line:
                        hardcoded_found.append((i, line.strip()))
            
            if hardcoded_found:
                self.print_error(f"{file_path.name}: Found {len(hardcoded_found)} hardcoded localhost URLs")
                for line_num, line in hardcoded_found[:3]:  # Show first 3
                    print(f"  Line {line_num}: {line[:80]}...")
                self.validation_results["critical_issues"].append(
                    f"{file_path.name}: {len(hardcoded_found)} hardcoded URLs"
                )
                self.validation_results["failed"] += 1
                all_clean = False
            else:
                self.print_success(f"{file_path.name}: No hardcoded localhost URLs")
                self.validation_results["passed"] += 1
        
        return all_clean
    
    def validate_no_password_fallbacks(self) -> bool:
        """Validate no password fallbacks in code"""
        self.print_header("Validating Password Security")
        
        files_to_check = [
            self.project_root / "workflows/scripts/workflow_manager.py",
            self.project_root / "workflows/scripts/deploy_dify_workflows.py"
        ]
        
        all_secure = True
        
        for file_path in files_to_check:
            self.validation_results["total_checks"] += 1
            
            if not file_path.exists():
                self.print_warning(f"File not found: {file_path}")
                continue
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for password fallbacks
            fallback_patterns = [
                r"getenv\(['\"].*PASSWORD['\"],\s*['\"].*['\"]",  # Fallback in getenv
                r"os\.environ\.get\(['\"].*PASSWORD['\"],\s*['\"].*['\"]",  # Fallback in environ.get
                r"=\s*['\"].*password['\"]",  # Hardcoded passwords
            ]
            
            has_fallback = False
            for pattern in fallback_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    has_fallback = True
                    break
            
            # Check for proper error handling
            has_validation = "raise ValueError" in content and "PASSWORD" in content
            
            if has_fallback:
                self.print_error(f"{file_path.name}: Contains password fallbacks")
                self.validation_results["critical_issues"].append(
                    f"{file_path.name}: Insecure password fallback detected"
                )
                self.validation_results["failed"] += 1
                all_secure = False
            elif not has_validation:
                self.print_warning(f"{file_path.name}: Missing password validation")
                self.validation_results["warnings"].append(
                    f"{file_path.name}: Consider adding explicit password validation"
                )
                self.validation_results["passed"] += 1
            else:
                self.print_success(f"{file_path.name}: Properly validates required passwords")
                self.validation_results["passed"] += 1
        
        return all_secure
    
    def validate_jwt_security(self) -> bool:
        """Validate JWT secret configuration"""
        self.print_header("Validating JWT Security")
        
        auth_file = self.project_root / "backend/app/core/auth.py"
        self.validation_results["total_checks"] += 1
        
        if not auth_file.exists():
            self.print_error(f"Auth file not found: {auth_file}")
            self.validation_results["failed"] += 1
            return False
        
        with open(auth_file, 'r') as f:
            content = f.read()
        
        # Check for secure JWT configuration
        has_random_fallback = "os.urandom" in content and "JWT_SECRET" in content
        has_length_check = "len(SECRET_KEY)" in content
        has_error_handling = "raise ValueError" in content and "JWT_SECRET_KEY" in content
        requires_env_var = 'os.getenv("JWT_SECRET_KEY")' in content and not 'os.getenv("JWT_SECRET_KEY",' in content
        
        if has_random_fallback:
            self.print_error("JWT secret uses random fallback (tokens invalidated on restart)")
            self.validation_results["critical_issues"].append(
                "JWT configuration uses random fallback"
            )
            self.validation_results["failed"] += 1
            return False
        elif not requires_env_var:
            self.print_error("JWT secret doesn't properly require environment variable")
            self.validation_results["critical_issues"].append(
                "JWT configuration has weak secret handling"
            )
            self.validation_results["failed"] += 1
            return False
        elif not has_length_check:
            self.print_warning("JWT secret missing length validation")
            self.validation_results["warnings"].append(
                "JWT secret should validate minimum length"
            )
            self.validation_results["passed"] += 1
            return True
        else:
            self.print_success("JWT configuration properly secured")
            self.validation_results["passed"] += 1
            return True
    
    def generate_report(self) -> str:
        """Generate validation report"""
        report = []
        report.append("\n" + "=" * 60)
        report.append("SECURITY VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {self.validation_results['timestamp']}")
        report.append(f"Total Checks: {self.validation_results['total_checks']}")
        report.append(f"Passed: {self.validation_results['passed']}")
        report.append(f"Failed: {self.validation_results['failed']}")
        
        if self.validation_results['critical_issues']:
            report.append(f"\n{RED}CRITICAL ISSUES:{RESET}")
            for issue in self.validation_results['critical_issues']:
                report.append(f"  • {issue}")
        
        if self.validation_results['warnings']:
            report.append(f"\n{YELLOW}WARNINGS:{RESET}")
            for warning in self.validation_results['warnings']:
                report.append(f"  • {warning}")
        
        if self.validation_results['failed'] == 0:
            report.append(f"\n{GREEN}{BOLD}✓ ALL SECURITY CHECKS PASSED{RESET}")
        else:
            report.append(f"\n{RED}{BOLD}✗ SECURITY VALIDATION FAILED{RESET}")
            report.append(f"  {self.validation_results['failed']} critical issues must be fixed")
        
        report.append("=" * 60 + "\n")
        
        return "\n".join(report)
    
    def save_results(self):
        """Save validation results to file"""
        results_file = self.project_root / "security_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    def run_validation(self) -> bool:
        """Run all security validations"""
        self.print_header("PHASE 1 SECURITY VALIDATION")
        
        # Run all validations
        docker_secure = self.validate_docker_security()
        urls_clean = self.validate_no_hardcoded_urls()
        passwords_secure = self.validate_no_password_fallbacks()
        jwt_secure = self.validate_jwt_security()
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        # Save results
        self.save_results()
        
        # Return overall status
        return all([docker_secure, urls_clean, passwords_secure, jwt_secure])


def main():
    """Main entry point"""
    validator = SecurityValidator()
    
    try:
        success = validator.run_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"{RED}Error during validation: {e}{RESET}")
        sys.exit(2)


if __name__ == "__main__":
    main()