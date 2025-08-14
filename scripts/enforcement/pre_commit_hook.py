#!/usr/bin/env python3
"""
🔧 PRE-COMMIT HOOK - Zero-Tolerance Rule Enforcement
Prevents any code from being committed that violates the 20 Fundamental Rules
"""

import sys
import os
import subprocess
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_rule_enforcer import ComprehensiveRuleEnforcer

def get_staged_files():
    """Get list of staged files for commit"""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout.strip().split('\n')
    return []

def main():
    """Pre-commit hook main function"""
    print("🔧 SUPREME VALIDATOR - Pre-Commit Rule Enforcement")
    print("=" * 60)
    
    # Get repository root
    repo_root = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    if not repo_root:
        repo_root = "/opt/sutazaiapp"
    
    # Get staged files
    staged_files = get_staged_files()
    if not staged_files or staged_files == ['']:
        print("✅ No files staged for commit")
        return 0
    
    print(f"Validating {len(staged_files)} staged files...")
    
    # Initialize enforcer
    enforcer = ComprehensiveRuleEnforcer(repo_root, auto_fix=False)
    
    # Run validation
    report = enforcer.validate_all_rules()
    
    # Check for critical violations
    critical_count = report["violations_by_severity"].get("CRITICAL", 0)
    high_count = report["violations_by_severity"].get("HIGH", 0)
    
    if critical_count > 0:
        print("\n❌ COMMIT BLOCKED: Critical rule violations detected!")
        print(f"   Found {critical_count} CRITICAL violations")
        print("\nCritical violations must be fixed before committing:")
        
        for violation in report["violations"][:5]:  # Show first 5
            if violation["severity"] == "CRITICAL":
                print(f"  • Rule {violation['rule']}: {violation['description']}")
                print(f"    File: {violation['file']}:{violation['line']}")
                print(f"    Fix: {violation['remediation']}")
        
        print("\nRun 'python scripts/enforcement/comprehensive_rule_enforcer.py' for full report")
        return 1
    
    if high_count > 0:
        print(f"\n⚠️  WARNING: {high_count} HIGH priority violations detected")
        print("Consider fixing these before committing:")
        
        for violation in report["violations"][:3]:  # Show first 3
            if violation["severity"] == "HIGH":
                print(f"  • Rule {violation['rule']}: {violation['description']}")
        
        # Allow commit with warning for HIGH violations
        response = input("\nProceed with commit anyway? (y/N): ")
        if response.lower() != 'y':
            print("Commit cancelled")
            return 1
    
    print(f"\n✅ Compliance Score: {report['compliance_score']}%")
    print("✅ Pre-commit validation passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())