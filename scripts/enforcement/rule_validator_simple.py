#!/usr/bin/env python3
"""
🔧 SutazAI Rule Enforcement Validator (Simplified)
Core rule validation system for immediate deployment
"""

import os
import sys
import subprocess
from pathlib import Path
import json


def validate_critical_rules(root_path="/opt/sutazaiapp"):
    """Validate the most critical rules"""
    violations = []
    root = Path(root_path)
    
    print("🔧 SUTAZAI RULE ENFORCEMENT VALIDATION")
    print("=" * 50)
    
    # Rule 1: Check for fantasy/placeholder code
    print("📌 Rule 1: Real Implementation Only...")
    fantasy_count = 0
    try:
        result = subprocess.run([
            'grep', '-r', 'TODO.*magic\|placeholder.*service\|imaginary.*system', 
            str(root), '--include=*.py'
        ], capture_output=True, text=True)
        if result.stdout:
            fantasy_count = len(result.stdout.strip().split('\n'))
            violations.append(f"Rule 1: {fantasy_count} fantasy code instances found")
    except:
        pass
    
    # Rule 5: Professional Standards - Test Coverage
    print("📌 Rule 5: Professional Standards...")
    test_files = list(root.rglob("test_*.py")) + list(root.rglob("*_test.py"))
    python_files = list(root.rglob("*.py"))
    if python_files:
        test_ratio = len(test_files) / len(python_files)
        if test_ratio < 0.2:  # Less than 20% test coverage
            violations.append(f"Rule 5: Test coverage {test_ratio:.1%} below professional standards")
    
    # Rule 7: Script Organization
    print("📌 Rule 7: Script Organization...")
    scripts_dir = root / "scripts"
    if not scripts_dir.exists():
        violations.append("Rule 7: Missing centralized /scripts/ directory")
    
    # Rule 11: Docker Excellence - Check for USER directive
    print("📌 Rule 11: Docker Excellence...")
    dockerfiles = list(root.rglob("Dockerfile*"))
    docker_violations = 0
    for dockerfile in dockerfiles:
        try:
            content = dockerfile.read_text()
            if "USER " not in content:
                docker_violations += 1
        except:
            pass
    if docker_violations > 0:
        violations.append(f"Rule 11: {docker_violations} Dockerfiles missing USER directive")
    
    # Rule 13: Zero Tolerance for Waste
    print("📌 Rule 13: Zero Tolerance for Waste...")
    try:
        result = subprocess.run([
            'grep', '-r', 'TODO:\|FIXME:\|XXX:\|HACK:', 
            str(root), '--include=*.py'
        ], capture_output=True, text=True)
        if result.stdout:
            waste_count = len(result.stdout.strip().split('\n'))
            violations.append(f"Rule 13: {waste_count} technical debt markers found")
    except:
        pass
    
    # Rule 20: MCP Protection
    print("📌 Rule 20: MCP Server Protection...")
    mcp_json = root / ".mcp.json"
    if mcp_json.exists():
        import time
        mtime = os.path.getmtime(mcp_json)
        if time.time() - mtime < 3600:  # Modified in last hour
            violations.append("Rule 20: MCP configuration modified recently - verify authorization")
    
    # Generate summary
    print("\n" + "=" * 50)
    if violations:
        print("⚠️  RULE VIOLATIONS DETECTED:")
        for violation in violations:
            print(f"  • {violation}")
        print(f"\nCompliance Status: {max(0, 100 - len(violations) * 10):.0f}%")
        return len(violations)
    else:
        print("✅ ALL CRITICAL RULES COMPLIANT")
        print("Compliance Status: 100%")
        return 0


def main():
    """Main execution"""
    try:
        violation_count = validate_critical_rules()
        
        if violation_count > 3:
            print("\n❌ CRITICAL: Too many violations detected!")
            sys.exit(2)
        elif violation_count > 0:
            print("\n⚠️  WARNING: Some violations detected")
            sys.exit(1)
        else:
            print("\n✅ SUCCESS: No critical violations")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n💥 ERROR: Validation failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()