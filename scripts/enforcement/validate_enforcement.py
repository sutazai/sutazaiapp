"""
Enforcement Rules Validation Script
Validates compliance with all 20 fundamental rules
Date: 2025-08-18 21:10:00 UTC
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path("/opt/sutazaiapp")

class EnforcementValidator:
    def __init__(self):
        self.violations = []
        self.warnings = []
        self.passed = []
        
    def check_rule_4_consolidation(self):
        """Rule 4: Investigate & Consolidate - Check for duplicate files"""
        print("\n📌 Checking Rule 4: Consolidation...")
        
        docker_files = list(PROJECT_ROOT.glob("**/docker-compose*.yml"))
        docker_files.extend(list(PROJECT_ROOT.glob("**/docker-compose*.yaml")))
        
        docker_files = [f for f in docker_files if 'node_modules' not in str(f)]
        
        if len(docker_files) > 1:
            self.violations.append(f"Rule 4: Found {len(docker_files)} docker-compose files (should be 1)")
            for f in docker_files:
                print(f"  ✗ {f.relative_to(PROJECT_ROOT)}")
        else:
            self.passed.append("Rule 4: Docker files consolidated ✓")
            
    def check_rule_5_structure(self):
        """Rule 5: Professional Project Structure"""
        print("\n📌 Checking Rule 5: Project Structure...")
        
        root_violations = []
        for pattern in ["*.py", "*.js", "*.test.js", "*.spec.js"]:
            for file in PROJECT_ROOT.glob(pattern):
                if file.name not in ["setup.py", "webpack.config.js"]:
                    root_violations.append(file.name)
        
        if root_violations:
            self.violations.append(f"Rule 5: {len(root_violations)} files in root folder")
            for f in root_violations:
                print(f"  ✗ {f} should not be in root")
        else:
            self.passed.append("Rule 5: Root folder clean ✓")
            
    def check_rule_6_documentation(self):
        """Rule 6: Centralized Documentation"""
        print("\n📌 Checking Rule 6: Documentation Structure...")
        
        required_dirs = [
            "docs/setup",
            "docs/architecture", 
            "docs/development",
            "docs/operations",
            "docs/api",
            "docs/team"
        ]
        
        missing = []
        for dir_path in required_dirs:
            if not (PROJECT_ROOT / dir_path).exists():
                missing.append(dir_path)
        
        if missing:
            self.violations.append(f"Rule 6: Missing {len(missing)} required doc directories")
            for d in missing:
                print(f"  ✗ Missing: {d}")
        else:
            self.passed.append("Rule 6: Documentation structure complete ✓")
            
    def check_rule_11_docker(self):
        """Rule 11: Docker Excellence"""
        print("\n📌 Checking Rule 11: Docker Excellence...")
        
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'health=unhealthy', '--format', '{{.Names}}'],
                capture_output=True, text=True
            )
            unhealthy = [c for c in result.stdout.strip().split('\n') if c]
            
            if unhealthy:
                self.violations.append(f"Rule 11: {len(unhealthy)} unhealthy containers")
                for c in unhealthy:
                    print(f"  ✗ Unhealthy: {c}")
            else:
                self.passed.append("Rule 11: All containers healthy ✓")
                
        except Exception as e:
            self.warnings.append(f"Rule 11: Could not check container health: {e}")
            
    def check_security(self):
        """Check security violations"""
        print("\n🔒 Checking Security...")
        
        env_files = []
        for pattern in [".env", "*.env"]:
            env_files.extend(PROJECT_ROOT.glob(f"**/{pattern}"))
        
        env_files = [f for f in env_files if 'node_modules' not in str(f) and '.venv' not in str(f)]
        
        if env_files:
            self.violations.append(f"Security: {len(env_files)} .env files in repository")
            for f in env_files[:5]:  # Show first 5
                print(f"  ✗ {f.relative_to(PROJECT_ROOT)}")
        else:
            self.passed.append("Security: No .env files in repository ✓")
            
    def check_duplicates(self):
        """Check for duplicate implementations"""
        print("\n🔍 Checking for Duplicates...")
        
        main_files = list(PROJECT_ROOT.glob("**/main.py"))
        main_files.extend(list(PROJECT_ROOT.glob("**/app.py")))
        
        main_files = [f for f in main_files if 'node_modules' not in str(f) and '.venv' not in str(f)]
        
        if len(main_files) > 2:  # Allow backend/main.py and frontend/app.py
            self.warnings.append(f"Duplicates: {len(main_files)} main/app files found")
            
        req_files = list(PROJECT_ROOT.glob("**/requirements*.txt"))
        req_files = [f for f in req_files if 'node_modules' not in str(f) and '.venv' not in str(f)]
        
        if len(req_files) > 3:  # Allow requirements.txt, requirements-dev.txt, requirements-test.txt
            self.warnings.append(f"Duplicates: {len(req_files)} requirements files found")
            
    def check_container_naming(self):
        """Check container naming standards"""
        print("\n🏷️ Checking Container Naming...")
        
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True
            )
            containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            unnamed = []
            for c in containers:
                if c and not (c.startswith('sutazai-') or c.startswith('mcp-')):
                    unnamed.append(c)
            
            if unnamed:
                self.violations.append(f"Naming: {len(unnamed)} containers with improper names")
                for c in unnamed:
                    print(f"  ✗ Bad name: {c}")
            else:
                self.passed.append("Naming: All containers properly named ✓")
                
        except Exception as e:
            self.warnings.append(f"Naming: Could not check container names: {e}")
            
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\n✅ PASSED ({len(self.passed)}):")
        for p in self.passed:
            print(f"  • {p}")
        
        if self.violations:
            print(f"\n❌ VIOLATIONS ({len(self.violations)}):")
            for v in self.violations:
                print(f"  • {v}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  • {w}")
        
        print("\n" + "=" * 60)
        
        if not self.violations:
            print("🎉 FULL COMPLIANCE ACHIEVED!")
        else:
            print(f"🚨 {len(self.violations)} VIOLATIONS REQUIRE IMMEDIATE ACTION")
        
        print("=" * 60)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "passed": len(self.passed),
            "violations": len(self.violations),
            "warnings": len(self.warnings),
            "details": {
                "passed": self.passed,
                "violations": self.violations,
                "warnings": self.warnings
            }
        }
        
        report_path = PROJECT_ROOT / "docs" / "reports" / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {report_path}")
        
        return len(self.violations) == 0

def main():
    print("🔍 ENFORCEMENT RULES VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    validator = EnforcementValidator()
    
    validator.check_rule_4_consolidation()
    validator.check_rule_5_structure()
    validator.check_rule_6_documentation()
    validator.check_rule_11_docker()
    validator.check_security()
    validator.check_duplicates()
    validator.check_container_naming()
    
    success = validator.generate_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())