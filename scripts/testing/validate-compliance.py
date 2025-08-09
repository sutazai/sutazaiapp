#!/usr/bin/env python3
"""
Compliance Validation Script
Verifies the codebase follows all 5 critical rules
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class ComplianceValidator:
    def __init__(self):
        self.root = Path("/opt/sutazaiapp")
        self.violations = []
        self.scores = {}
        
    def check_rule1_no_fantasy(self) -> Tuple[int, List[str]]:
        """Check for fantasy elements"""
        violations = []
        fantasy_terms = [
            'wizard', 'magic', 'teleport', 'black-box', 'telekinesis',
            'quantum', 'agi', 'superintelligence', 'sentient', 'consciousness'
        ]
        
        # Exclude validation scripts themselves
        exclude_paths = ['validate-compliance.py', 'check-fantasy-elements.py']
        
        for ext in ['*.py', '*.js', '*.ts', '*.yaml', '*.yml']:
            for filepath in self.root.rglob(ext):
                if any(ex in str(filepath) for ex in exclude_paths):
                    continue
                    
                try:
                    content = filepath.read_text().lower()
                    for term in fantasy_terms:
                        if term in content:
                            violations.append(f"{filepath.relative_to(self.root)}: contains '{term}'")
                            break
                except:
                    pass
                    
        score = 100 if not violations else max(0, 100 - len(violations))
        return score, violations[:10]  # Return first 10
        
    def check_rule2_dont_break(self) -> Tuple[int, List[str]]:
        """Check if core functionality is preserved"""
        checks = []
        score = 100
        
        # Check essential files exist
        essential_files = [
            "backend/main.py",
            "frontend/app.py", 
            "docker-compose.yml"
        ]
        
        for rel_path in essential_files:
            filepath = self.root / rel_path
            if not filepath.exists():
                checks.append(f"Missing essential file: {rel_path}")
                score -= 20
                
        # Check for test coverage
        test_dir = self.root / "tests"
        if not test_dir.exists() or len(list(test_dir.rglob("test_*.py"))) < 5:
            checks.append("Insufficient test coverage")
            score -= 10
            
        return max(0, score), checks
        
    def check_rule3_hygiene(self) -> Tuple[int, List[str]]:
        """Check codebase organization and cleanliness"""
        issues = []
        score = 100
        
        # Check docker-compose proliferation
        compose_files = list(self.root.glob("docker-compose*.yml"))
        if len(compose_files) > 2:
            issues.append(f"Too many docker-compose files: {len(compose_files)} (max 2)")
            score -= 30
            
        # Check script organization
        script_count = len(list((self.root / "scripts").rglob("*.py"))) + \
                      len(list((self.root / "scripts").rglob("*.sh")))
        if script_count > 50:
            issues.append(f"Too many scripts: {script_count} (max 50)")
            score -= 20
            
        # Check documentation bloat
        doc_count = len(list(self.root.rglob("*.md")))
        if doc_count > 50:
            issues.append(f"Too many documentation files: {doc_count} (max 50)")
            score -= 20
            
        # Check for junk patterns
        junk_patterns = ["*_old.*", "*_backup.*", "*_temp.*", "*.tmp"]
        junk_count = 0
        for pattern in junk_patterns:
            junk_count += len(list(self.root.rglob(pattern)))
        if junk_count > 0:
            issues.append(f"Found {junk_count} junk files")
            score -= 10
            
        return max(0, score), issues
        
    def check_rule4_reuse(self) -> Tuple[int, List[str]]:
        """Check for code duplication"""
        duplicates = []
        score = 100
        
        # Check for duplicate functionality patterns
        patterns_to_check = [
            ("deploy*.sh", "scripts"),
            ("monitor*.py", "scripts"),
            ("validate*.py", "scripts"),
            ("requirements*.txt", ".")
        ]
        
        for pattern, directory in patterns_to_check:
            search_dir = self.root / directory if directory != "." else self.root
            matches = list(search_dir.rglob(pattern))
            if len(matches) > 3:
                duplicates.append(f"Potential duplicates: {len(matches)} files matching {pattern}")
                score -= 15
                
        return max(0, score), duplicates
        
    def check_rule5_local_llms(self) -> Tuple[int, List[str]]:
        """Check for external API usage"""
        violations = []
        score = 100
        
        external_apis = [
            'OPENAI_API_KEY', 'openai.api_key', 'ChatOpenAI',
            'anthropic', 'claude-3', 'gpt-3', 'gpt-4'
        ]
        
        for ext in ['*.py', '*.yaml', '*.yml', '*.env*']:
            for filepath in self.root.rglob(ext):
                try:
                    content = filepath.read_text()
                    for api in external_apis:
                        if api in content:
                            violations.append(f"{filepath.relative_to(self.root)}: references '{api}'")
                            score -= 5
                            break
                except:
                    pass
                    
        # Check for Ollama configuration
        if not (self.root / "docker-compose.yml").exists():
            violations.append("No docker-compose.yml with Ollama configuration")
            score -= 20
            
        return max(0, score), violations[:10]
        
    def generate_report(self) -> Dict:
        """Generate comprehensive compliance report"""
        rules = [
            ("Rule 1: No Fantasy Elements", self.check_rule1_no_fantasy),
            ("Rule 2: Don't Break Existing", self.check_rule2_dont_break),
            ("Rule 3: Codebase Hygiene", self.check_rule3_hygiene),
            ("Rule 4: Reuse Before Creating", self.check_rule4_reuse),
            ("Rule 5: Local LLMs Only", self.check_rule5_local_llms)
        ]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0,
            "rules": {}
        }
        
        total_score = 0
        print("\n" + "=" * 60)
        print("COMPLIANCE VALIDATION REPORT")
        print("=" * 60)
        
        for rule_name, check_func in rules:
            score, issues = check_func()
            total_score += score
            
            status = "✅ PASS" if score >= 90 else "⚠️ WARN" if score >= 70 else "❌ FAIL"
            print(f"\n{rule_name}: {score}% {status}")
            
            if issues:
                print("  Issues found:")
                for issue in issues[:5]:  # Show first 5
                    print(f"    - {issue}")
                    
            report["rules"][rule_name] = {
                "score": score,
                "status": status,
                "issues": issues
            }
            
        overall_score = total_score // len(rules)
        report["overall_score"] = overall_score
        
        print("\n" + "=" * 60)
        print(f"OVERALL COMPLIANCE SCORE: {overall_score}%")
        
        if overall_score >= 90:
            print("STATUS: ✅ COMPLIANT")
        elif overall_score >= 70:
            print("STATUS: ⚠️ NEEDS IMPROVEMENT")
        else:
            print("STATUS: ❌ NON-COMPLIANT")
            
        print("=" * 60)
        
        # Save report
        report_path = self.root / "compliance_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nDetailed report saved to: {report_path}")
        
        return report
        
    def run(self):
        """Run full compliance validation"""
        return self.generate_report()

if __name__ == "__main__":
    validator = ComplianceValidator()
    report = validator.run()
    
    # Exit with non-zero if non-compliant
    if report["overall_score"] < 70:
        exit(1)