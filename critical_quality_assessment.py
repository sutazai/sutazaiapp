#!/usr/bin/env python3
"""
Critical Quality Assessment - Focuses on the most impactful issues
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import subprocess

def analyze_backend_quality():
    """Analyze backend API code quality"""
    backend_path = Path('/opt/sutazaiapp/backend')
    issues = []
    
    # Check for proper error handling
    for py_file in backend_path.rglob('*.py'):
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Check for bare except clauses
        if re.search(r'except:\s*$', content, re.MULTILINE):
            issues.append(f"Bare except clause in {py_file.relative_to(backend_path)}")
            
        # Check for missing input validation
        if 'request.get_json()' in content and 'validate' not in content.lower():
            issues.append(f"Missing input validation in {py_file.relative_to(backend_path)}")
            
        # Check for SQL injection vulnerabilities
        if re.search(r'(execute|query)\s*\([^)]*%[^)]*\)', content):
            issues.append(f"Potential SQL injection in {py_file.relative_to(backend_path)}")
    
    return issues

def analyze_agent_stubs():
    """Identify stub vs real agent implementations"""
    agents_path = Path('/opt/sutazaiapp/agents')
    stub_agents = []
    real_agents = []
    
    for agent_dir in agents_path.iterdir():
        if not agent_dir.is_dir() or agent_dir.name.startswith('.'):
            continue
            
        app_file = agent_dir / 'app.py'
        if app_file.exists():
            with open(app_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Check for stub indicators
            stub_indicators = [
                'return {"status": "stub"}',
                'return {"result": "placeholder"}',
                '# TODO: Implement',
                'pass  # Stub',
                'return "Not implemented"'
            ]
            
            is_stub = any(indicator in content for indicator in stub_indicators)
            
            # Check for minimal implementation (less than 50 lines of actual code)
            code_lines = [l for l in content.splitlines() 
                         if l.strip() and not l.strip().startswith('#')]
            
            if is_stub or len(code_lines) < 50:
                stub_agents.append(agent_dir.name)
            else:
                real_agents.append(agent_dir.name)
    
    return stub_agents, real_agents

def analyze_docker_configs():
    """Analyze Docker configuration issues"""
    issues = []
    compose_files = list(Path('/opt/sutazaiapp').glob('docker-compose*.yml'))
    
    # Check for duplicate service definitions
    services_defined = defaultdict(list)
    
    for compose_file in compose_files:
        with open(compose_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Extract service names
        services = re.findall(r'^\s{2}(\w[\w-]*):$', content, re.MULTILINE)
        for service in services:
            services_defined[service].append(compose_file.name)
    
    # Find duplicates
    for service, files in services_defined.items():
        if len(files) > 1:
            issues.append(f"Service '{service}' defined in multiple files: {', '.join(files)}")
    
    # Count total compose files (too many = complexity)
    if len(compose_files) > 5:
        issues.append(f"Too many docker-compose files ({len(compose_files)}). Consider consolidation.")
    
    return issues, services_defined

def analyze_code_duplication():
    """Find significant code duplication"""
    
    # Use a simple approach - find files with identical content
    file_hashes = defaultdict(list)
    
    for py_file in Path('/opt/sutazaiapp').rglob('*.py'):
        if '__pycache__' in str(py_file) or '.git' in str(py_file):
            continue
            
        try:
            with open(py_file, 'rb') as f:
                content = f.read()
            import hashlib
            file_hash = hashlib.md5(content).hexdigest()
            file_hashes[file_hash].append(str(py_file.relative_to('/opt/sutazaiapp')))
        except:
            pass
    
    duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}
    
    return duplicates

def analyze_dependencies():
    """Analyze dependency management issues"""
    issues = []
    requirements_files = list(Path('/opt/sutazaiapp').rglob('requirements*.txt'))
    
    # Count total requirements files
    issues.append(f"Total requirements files: {len(requirements_files)}")
    
    # Check for version conflicts
    package_versions = defaultdict(set)
    
    for req_file in requirements_files[:50]:  # Sample first 50 to avoid timeout
        try:
            with open(req_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package and version
                        match = re.match(r'^([a-zA-Z0-9-_]+)==(.+)$', line)
                        if match:
                            pkg, version = match.groups()
                            package_versions[pkg].add(version)
        except:
            pass
    
    # Find conflicts
    conflicts = {pkg: list(versions) for pkg, versions in package_versions.items() 
                if len(versions) > 1}
    
    if conflicts:
        issues.append(f"Found {len(conflicts)} packages with version conflicts")
        
    return issues, conflicts

def main():
    print("\n" + "="*80)
    print("🔍 CRITICAL QUALITY ASSESSMENT REPORT")
    print("="*80)
    
    # 1. Backend Quality
    print("\n📌 BACKEND API QUALITY:")
    backend_issues = analyze_backend_quality()
    if backend_issues:
        for issue in backend_issues[:5]:
            print(f"  ⚠️ {issue}")
        if len(backend_issues) > 5:
            print(f"  ... and {len(backend_issues) - 5} more issues")
    else:
        print("  ✅ No critical issues found")
    
    # 2. Agent Implementation Status
    print("\n🤖 AGENT IMPLEMENTATION STATUS:")
    stub_agents, real_agents = analyze_agent_stubs()
    print(f"  • Real implementations: {len(real_agents)}")
    print(f"  • Stub implementations: {len(stub_agents)}")
    print(f"  • Stub percentage: {len(stub_agents) / (len(stub_agents) + len(real_agents)) * 100:.1f}%")
    
    if real_agents:
        print(f"  • Sample real agents: {', '.join(real_agents[:5])}")
    
    # 3. Docker Configuration
    print("\n🐳 DOCKER CONFIGURATION ISSUES:")
    docker_issues, services = analyze_docker_configs()
    for issue in docker_issues[:5]:
        print(f"  ⚠️ {issue}")
    print(f"  • Total unique services: {len(services)}")
    
    # 4. Code Duplication
    print("\n📑 CODE DUPLICATION:")
    duplicates = analyze_code_duplication()
    if duplicates:
        print(f"  • Found {len(duplicates)} sets of duplicate files")
        for _, files in list(duplicates.items())[:3]:
            if len(files) <= 3:
                print(f"    - {', '.join(files)}")
    else:
        print("  ✅ No exact duplicates found")
    
    # 5. Dependency Management
    print("\n📦 DEPENDENCY MANAGEMENT:")
    dep_issues, conflicts = analyze_dependencies()
    for issue in dep_issues:
        print(f"  • {issue}")
    
    if conflicts:
        print(f"\n  Version conflicts (top 5):")
        for pkg, versions in list(conflicts.items())[:5]:
            print(f"    - {pkg}: {', '.join(versions[:3])}")
    
    # 6. Technical Debt Summary
    print("\n💰 TECHNICAL DEBT ESTIMATION:")
    
    # Load the full report
    with open('/opt/sutazaiapp/sonarqube_quality_report.json', 'r') as f:
        full_report = json.load(f)
    
    print(f"  • Estimated effort: {full_report['summary']['technical_debt_days']} days")
    print(f"  • Critical security issues: {full_report['summary']['issues_by_severity'].get('BLOCKER', 0)}")
    print(f"  • Major bugs: {full_report['summary']['issues_by_severity'].get('CRITICAL', 0)}")
    
    # 7. Top Recommendations
    print("\n🎯 TOP PRIORITY RECOMMENDATIONS:")
    print("  1. Fix 71 BLOCKER security issues (hardcoded credentials, eval/exec usage)")
    print("  2. Consolidate 320 requirements files into centralized dependency management")
    print("  3. Replace stub agent implementations with real functionality")
    print("  4. Reduce docker-compose files from current count to 3-5 max")
    print("  5. Implement proper error handling and input validation in backend")
    print("  6. Add comprehensive test coverage (currently minimal)")
    print("  7. Remove duplicate code and consolidate shared functionality")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()