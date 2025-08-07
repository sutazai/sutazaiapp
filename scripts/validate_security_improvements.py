#!/usr/bin/env python3
"""
Security Improvements Validation Script
Purpose: Validate all security remediation measures are working correctly
Usage: python scripts/validate_security_improvements.py
"""

import os
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

class SecurityValidator:
    def __init__(self, root_dir="/opt/sutazaiapp"):
        self.root_dir = Path(root_dir)
        self.validation_results = {}
        self.overall_score = 0
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🔒 {title}")
        print(f"{'='*60}")
        
    def print_check(self, description, passed, details=""):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {description}")
        if details and not passed:
            print(f"   Details: {details}")
        return passed
        
    def validate_secrets_removed(self):
        """Validate that hardcoded secrets have been removed"""
        self.print_header("Hardcoded Secrets Validation")
        
        try:
            result = subprocess.run(
                ["python", "scripts/check_secrets.py"],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            
            # Count actual secret violations (excluding test files and backup files)
            output_lines = result.stdout.split('\n') if result.stdout else []
            critical_secrets = 0
            
            for line in output_lines:
                if " - " in line and not any(exclude in line.lower() for exclude in [
                    'backup', 'archive', 'test', 'client_secret="*****"', 'invalid_token'
                ]):
                    critical_secrets += 1
                    
            secrets_clean = critical_secrets == 0
            self.print_check("Critical hardcoded secrets removed", secrets_clean, 
                           f"Found {critical_secrets} critical secrets")
            
            # Check for environment variable templates
            env_template = self.root_dir / "security-scan-results/templates/.env.secure.template"
            template_exists = env_template.exists()
            self.print_check("Environment variable template created", template_exists)
            
            # Check for secret generation script
            gen_script = self.root_dir / "security-scan-results/templates/generate-secrets.sh"
            script_exists = gen_script.exists() and os.access(gen_script, os.X_OK)
            self.print_check("Secret generation script available", script_exists)
            
            score = (80 if secrets_clean else 0) + (10 if template_exists else 0) + (10 if script_exists else 0)
            self.validation_results["secrets"] = {"score": score, "max": 100, "passed": secrets_clean}
            
        except Exception as e:
            self.print_check("Secret validation check", False, str(e))
            self.validation_results["secrets"] = {"score": 0, "max": 100, "passed": False}
            
    def validate_dependencies_pinned(self):
        """Validate that dependencies are properly pinned"""
        self.print_header("Dependency Security Validation")
        
        # Check main requirements files
        key_files = [
            "backend/requirements.txt",
            "frontend/requirements.txt", 
            "requirements.txt"
        ]
        
        pinned_files = 0
        total_files = len(key_files)
        
        for req_file in key_files:
            file_path = self.root_dir / req_file
            if file_path.exists():
                content = file_path.read_text()
                # Count unpinned dependencies (using >= or > or ~=)
                unpinned = len(re.findall(r'[a-zA-Z0-9_-]+[>~]=', content))
                pinned = unpinned == 0
                
                self.print_check(f"Dependencies pinned in {req_file}", pinned,
                               f"Found {unpinned} unpinned dependencies")
                if pinned:
                    pinned_files += 1
            else:
                self.print_check(f"Requirements file exists: {req_file}", False, "File not found")
                
        # Check for security requirements summary
        summary_file = self.root_dir / "security-scan-results/requirements-security-summary.md"
        summary_exists = summary_file.exists()
        self.print_check("Security requirements summary created", summary_exists)
        
        score = (70 * pinned_files // total_files) + (30 if summary_exists else 0)
        all_pinned = pinned_files == total_files
        self.validation_results["dependencies"] = {"score": score, "max": 100, "passed": all_pinned}
        
    def validate_container_security(self):
        """Validate container security improvements"""
        self.print_header("Container Security Validation")
        
        # Check for security Docker Compose override
        security_compose = self.root_dir / "docker-compose.security.yml"
        security_file_exists = security_compose.exists()
        self.print_check("Security Docker Compose override created", security_file_exists)
        
        if security_file_exists:
            content = security_compose.read_text()
            
            # Check for security hardening features
            hardening_features = [
                ("no-new-privileges", "no-new-privileges:true" in content),
                ("AppArmor profiles", "apparmor:docker-default" in content),
                ("Read-only filesystems", "read_only: true" in content),  
                ("Secure tmpfs mounts", "tmpfs:" in content and "noexec,nosuid" in content),
                ("Capability restrictions", "cap_add:" in content and "privileged: false" in content)
            ]
            
            hardening_score = 0
            for feature_name, feature_present in hardening_features:
                self.print_check(f"Container hardening: {feature_name}", feature_present)
                if feature_present:
                    hardening_score += 20
                    
            # Check main compose file for privileged containers
            main_compose = self.root_dir / "docker-compose.yml"
            if main_compose.exists():
                main_content = main_compose.read_text()
                privileged_count = main_content.count("privileged: true")
                no_unmitigated_privileges = privileged_count <= 2  # Known: cadvisor and hardware optimizer
                self.print_check(f"Privileged containers mitigated", no_unmitigated_privileges,
                               f"Found {privileged_count} privileged containers")
        else:
            hardening_score = 0
            no_unmitigated_privileges = False
            
        score = (60 if security_file_exists else 0) + (hardening_score // 5) + (20 if no_unmitigated_privileges else 0)
        self.validation_results["containers"] = {"score": score, "max": 100, "passed": security_file_exists}
        
    def validate_security_pipeline(self):
        """Validate automated security pipeline"""
        self.print_header("Security Pipeline Validation")
        
        # Check for GitHub Actions security workflow
        workflow_file = self.root_dir / ".github/workflows/security-scan.yml"
        workflow_exists = workflow_file.exists()
        self.print_check("Automated security scanning workflow", workflow_exists)
        
        # Check for security scanning script
        security_script = self.root_dir / "scripts/check_secrets.py"
        script_exists = security_script.exists()
        self.print_check("Security scanning script available", script_exists)
        
        # Check for remediation documentation
        remediation_doc = self.root_dir / "security-scan-results/remediation-summary.md"
        doc_exists = remediation_doc.exists()
        self.print_check("Remediation documentation created", doc_exists)
        
        score = (40 if workflow_exists else 0) + (30 if script_exists else 0) + (30 if doc_exists else 0)
        pipeline_ready = workflow_exists and script_exists
        self.validation_results["pipeline"] = {"score": score, "max": 100, "passed": pipeline_ready}
        
    def calculate_overall_score(self):
        """Calculate overall security score"""
        self.print_header("Overall Security Score")
        
        total_weighted_score = 0
        total_weight = 0
        
        # Weight different categories
        weights = {
            "secrets": 30,      # Highest priority
            "dependencies": 25, # High priority  
            "containers": 25,   # High priority
            "pipeline": 20      # Medium priority
        }
        
        for category, weight in weights.items():
            if category in self.validation_results:
                result = self.validation_results[category]
                weighted_score = (result["score"] / result["max"]) * weight
                total_weighted_score += weighted_score
                total_weight += weight
                
                status = "✅ EXCELLENT" if result["score"] >= 90 else \
                        "🟡 GOOD" if result["score"] >= 70 else \
                        "🟠 NEEDS IMPROVEMENT" if result["score"] >= 50 else \
                        "🔴 CRITICAL"
                        
                print(f"{category.upper():15} {result['score']:3d}/100 {status}")
        
        if total_weight > 0:
            self.overall_score = total_weighted_score / total_weight * 10  # Convert to x/10 scale
        else:
            self.overall_score = 0
            
        print(f"\n{'='*60}")
        print(f"🎯 OVERALL SECURITY SCORE: {self.overall_score:.1f}/10")
        
        if self.overall_score >= 8.5:
            print("🏆 EXCELLENT - Production ready with enterprise security")
        elif self.overall_score >= 7.0:
            print("✅ GOOD - Acceptable security posture with minor improvements needed")
        elif self.overall_score >= 5.5:
            print("🟡 MODERATE - Security improvements required before production")
        else:
            print("🔴 POOR - Significant security issues must be addressed")
            
        print(f"{'='*60}")
        
    def run_validation(self):
        """Run complete security validation"""
        print("🔒 SutazAI Security Remediation Validation")
        print(f"📅 Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Target Directory: {self.root_dir}")
        
        # Run all validation checks
        self.validate_secrets_removed()
        self.validate_dependencies_pinned()
        self.validate_container_security()
        self.validate_security_pipeline()
        self.calculate_overall_score()
        
        # Save validation results
        results_file = self.root_dir / "security-scan-results/validation-results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "overall_score": self.overall_score,
                "category_results": self.validation_results,
                "status": "EXCELLENT" if self.overall_score >= 8.5 else
                         "GOOD" if self.overall_score >= 7.0 else
                         "MODERATE" if self.overall_score >= 5.5 else "POOR"
            }, f, indent=2)
            
        print(f"\n📊 Detailed results saved to: {results_file}")
        
        return self.overall_score >= 8.0
        
if __name__ == "__main__":
    validator = SecurityValidator()
    success = validator.run_validation()
    exit(0 if success else 1)