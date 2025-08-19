"""
Rule Compliance Validation Script
Created: 2025-08-18 17:05:00 UTC
Purpose: Validate rule compliance after enforcement fixes
"""

import os
import sys
from pathlib import Path
from datetime import datetime

class ComplianceValidator:
    def __init__(self):
        self.root_dir = Path('/opt/sutazaiapp')
        self.violations = []
        self.compliant = []
        self.rules_checked = 0
        self.rules_passed = 0
    
    def check_rule_4_docker_consolidation(self):
        """Check if Docker configurations are consolidated."""
        print("\n📋 Checking Rule 4: Docker Consolidation...")
        self.rules_checked += 1
        
        docker_dir = self.root_dir / 'docker'
        consolidated_file = docker_dir / 'docker-compose.consolidated.yml'
        
        if not consolidated_file.exists():
            self.violations.append("Rule 4: docker-compose.consolidated.yml does not exist")
            print("   ❌ Consolidated Docker file not found")
            return False
        
        compose_files = list(docker_dir.glob('docker-compose*.yml'))
        compose_files.extend(docker_dir.glob('docker-compose*.yaml'))
        
        active_files = [f for f in compose_files 
                       if 'consolidated' not in f.name and 'deprecated' not in f.suffix]
        
        if len(active_files) > 0:
            self.violations.append(f"Rule 4: {len(active_files)} unconsolidated docker-compose files still exist")
            print(f"   ❌ Found {len(active_files)} unconsolidated Docker files")
            for f in active_files[:5]:  # Show first 5
                print(f"      - {f.name}")
            return False
        
        self.compliant.append("Rule 4: Docker configurations properly consolidated")
        self.rules_passed += 1
        print("   ✅ Docker configurations properly consolidated")
        return True
    
    def check_rule_6_test_files(self):
        """Check if test files are in proper location."""
        print("\n📋 Checking Rule 6: Test File Organization...")
        self.rules_checked += 1
        
        root_test_files = []
        for pattern in ['*.test.*', '*test*.py', 'test-*', 'Test*']:
            root_test_files.extend(self.root_dir.glob(pattern))
        
        root_test_files = [f for f in root_test_files if f.name != 'tests' and f.is_file()]
        
        if root_test_files:
            self.violations.append(f"Rule 6: {len(root_test_files)} test files found in root directory")
            print(f"   ❌ Found {len(root_test_files)} test files in root:")
            for f in root_test_files[:5]:  # Show first 5
                print(f"      - {f.name}")
            return False
        
        tests_dir = self.root_dir / 'tests'
        if not tests_dir.exists():
            self.violations.append("Rule 6: /tests directory does not exist")
            print("   ❌ /tests directory not found")
            return False
        
        self.compliant.append("Rule 6: Test files properly organized")
        self.rules_passed += 1
        print("   ✅ Test files properly organized in /tests")
        return True
    
    def check_rule_18_changelogs(self):
        """Check if all directories have CHANGELOG.md."""
        print("\n📋 Checking Rule 18: CHANGELOG.md Coverage...")
        self.rules_checked += 1
        
        total_dirs = 0
        dirs_with_changelog = 0
        dirs_without_changelog = []
        
        skip_patterns = ['node_modules', '__pycache__', '.git', '.venv', 'venv', 
                        '.pytest_cache', '.mypy_cache', 'htmlcov', 'dist', 'build']
        
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            skip = False
            for pattern in skip_patterns:
                if pattern in dirpath:
                    skip = True
                    break
            
            if skip:
                dirnames[:] = []  # Don't walk into subdirectories
                continue
            
            total_dirs += 1
            if 'CHANGELOG.md' in filenames:
                dirs_with_changelog += 1
            else:
                dirs_without_changelog.append(dirpath)
        
        coverage = (dirs_with_changelog / total_dirs * 100) if total_dirs > 0 else 0
        
        if coverage < 95:  # Allow 95% threshold for system directories
            self.violations.append(f"Rule 18: Only {coverage:.1f}% directories have CHANGELOG.md ({dirs_with_changelog}/{total_dirs})")
            print(f"   ❌ Only {coverage:.1f}% directories have CHANGELOG.md")
            print(f"      Missing in {len(dirs_without_changelog)} directories")
            if dirs_without_changelog:
                print("      First 5 missing:")
                for d in dirs_without_changelog[:5]:
                    print(f"      - {Path(d).relative_to(self.root_dir)}")
            return False
        
        self.compliant.append(f"Rule 18: {coverage:.1f}% CHANGELOG.md coverage")
        self.rules_passed += 1
        print(f"   ✅ {coverage:.1f}% directories have CHANGELOG.md")
        return True
    
    def check_rule_4_requirements(self):
        """Check if requirements files are consolidated."""
        print("\n📋 Checking Rule 4: Requirements Consolidation...")
        self.rules_checked += 1
        
        main_requirements = self.root_dir / 'requirements.txt'
        
        if not main_requirements.exists():
            self.violations.append("Rule 4: Main requirements.txt does not exist")
            print("   ❌ Main requirements.txt not found")
            return False
        
        scattered_requirements = []
        for pattern in ['requirements*.txt', 'requirements*.in']:
            for req_file in self.root_dir.rglob(pattern):
                rel_path = req_file.relative_to(self.root_dir)
                if (req_file.parent == self.root_dir and 
                    req_file.name in ['requirements.txt', 'requirements-dev.txt', 
                                      'requirements-test.txt', 'requirements-prod.txt']):
                    continue
                scattered_requirements.append(rel_path)
        
        if scattered_requirements:
            self.violations.append(f"Rule 4: {len(scattered_requirements)} scattered requirements files found")
            print(f"   ❌ Found {len(scattered_requirements)} scattered requirements files:")
            for f in scattered_requirements[:5]:
                print(f"      - {f}")
            return False
        
        self.compliant.append("Rule 4: Requirements properly consolidated")
        self.rules_passed += 1
        print("   ✅ Requirements properly consolidated")
        return True
    
    def check_rule_13_waste(self):
        """Check for waste and unnecessary files."""
        print("\n📋 Checking Rule 13: Zero Waste...")
        self.rules_checked += 1
        
        waste_patterns = [
            ('*backup*', 'backup'),
            ('*old*', 'old'),
            ('*deprecated*', 'deprecated'),
            ('*temp*', 'temporary'),
            ('*tmp*', 'temporary'),
            ('*archive*', 'archive'),
            ('*obsolete*', 'obsolete'),
            ('*legacy*', 'legacy'),
        ]
        
        waste_found = []
        for pattern, desc in waste_patterns:
            for item in self.root_dir.rglob(pattern):
                if any(skip in str(item) for skip in ['.venv', 'node_modules', '.git']):
                    continue
                if str(item) == str(self.root_dir / 'archive'):
                    continue
                waste_found.append((item, desc))
        
        if len(waste_found) > 10:  # More than 10 waste items is too much
            self.violations.append(f"Rule 13: {len(waste_found)} waste/temporary files found")
            print(f"   ❌ Found {len(waste_found)} waste/temporary files")
            for item, desc in waste_found[:5]:
                print(f"      - {item.relative_to(self.root_dir)} ({desc})")
            return False
        
        self.compliant.append(f"Rule 13: Minimal waste ({len(waste_found)} items)")
        self.rules_passed += 1
        print(f"   ✅ Minimal waste found ({len(waste_found)} acceptable items)")
        return True
    
    def generate_compliance_report(self):
        """Generate compliance report."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        report_path = self.root_dir / 'docs' / 'reports' / f'COMPLIANCE_VALIDATION_{timestamp}.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        compliance_rate = (self.rules_passed / self.rules_checked * 100) if self.rules_checked > 0 else 0
        
        with open(report_path, 'w') as f:
            f.write(f"# Rule Compliance Validation Report\n")
            f.write(f"**Generated**: {datetime.utcnow().isoformat()}Z\n\n")
            
            f.write(f"## Overall Compliance: {compliance_rate:.1f}%\n")
            f.write(f"- Rules Checked: {self.rules_checked}\n")
            f.write(f"- Rules Passed: {self.rules_passed}\n")
            f.write(f"- Rules Failed: {self.rules_checked - self.rules_passed}\n\n")
            
            if self.compliant:
                f.write("## ✅ Compliant Rules\n")
                for item in self.compliant:
                    f.write(f"- {item}\n")
                f.write("\n")
            
            if self.violations:
                f.write("## ❌ Violations Found\n")
                for violation in self.violations:
                    f.write(f"- {violation}\n")
                f.write("\n")
                
                f.write("## 🔧 Required Actions\n")
                if any('docker' in v.lower() for v in self.violations):
                    f.write("1. Run `python3 scripts/enforcement/consolidate_docker.py`\n")
                if any('changelog' in v.lower() for v in self.violations):
                    f.write("2. Run `python3 scripts/enforcement/add_missing_changelogs.py`\n")
                if any('test' in v.lower() for v in self.violations):
                    f.write("3. Run `python3 scripts/enforcement/priority_fixes.py`\n")
            
            f.write("\n## Validation Details\n")
            f.write(f"- Script: validate_compliance.py\n")
            f.write(f"- Executed: {datetime.utcnow().isoformat()}Z\n")
            f.write(f"- Root Directory: {self.root_dir}\n")
        
        return report_path, compliance_rate

def main():
    print("=" * 60)
    print("🔍 RULE COMPLIANCE VALIDATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
    
    validator = ComplianceValidator()
    
    validator.check_rule_4_docker_consolidation()
    validator.check_rule_6_test_files()
    validator.check_rule_18_changelogs()
    validator.check_rule_4_requirements()
    validator.check_rule_13_waste()
    
    report_path, compliance_rate = validator.generate_compliance_report()
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Overall Compliance: {compliance_rate:.1f}%")
    print(f"Rules Passed: {validator.rules_passed}/{validator.rules_checked}")
    print(f"Violations Found: {len(validator.violations)}")
    print(f"Report Saved: {report_path}")
    
    if compliance_rate >= 80:
        print("\n✅ System is approaching compliance!")
    elif compliance_rate >= 50:
        print("\n⚠️  System needs significant work to achieve compliance")
    else:
        print("\n🚨 CRITICAL: System is severely non-compliant!")
        print("   Run enforcement scripts immediately:")
        print("   1. python3 scripts/enforcement/priority_fixes.py")
        print("   2. python3 scripts/enforcement/consolidate_docker.py")
        print("   3. python3 scripts/enforcement/add_missing_changelogs.py")
        sys.exit(1)

if __name__ == '__main__':
    main()