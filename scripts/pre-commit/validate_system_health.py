#!/usr/bin/env python3
"""
System Health Validation Script
Purpose: Verify all critical services are healthy before making changes
Author: System Architect
Date: 2025-08-09
"""

import subprocess
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple

def check_docker_services() -> Tuple[bool, List[str]]:
    """Check if critical Docker services are running"""
    critical_services = [
        "sutazai-backend",
        "sutazai-postgres",
        "sutazai-redis",
        "sutazai-neo4j",
        "sutazai-ollama",
        "sutazai-frontend"
    ]
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        running_containers = result.stdout.strip().split('\n')
        
        missing = []
        for service in critical_services:
            if service not in running_containers:
                missing.append(service)
        
        return len(missing) == 0, missing
    except Exception as e:
        return False, [f"Error checking Docker: {str(e)}"]

def check_backend_health() -> Tuple[bool, Dict]:
    """Check backend API health"""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:10010/health"],
            capture_output=True,
            text=True,
            check=True
        )
        health_data = json.loads(result.stdout)
        
        is_healthy = health_data.get("status") == "healthy"
        return is_healthy, health_data
    except Exception as e:
        return False, {"error": str(e)}

def check_ollama_models() -> Tuple[bool, List[str]]:
    """Check Ollama loaded models"""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:10104/api/tags"],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        models = [m["name"] for m in data.get("models", [])]
        
        has_tinyllama = any("tinyllama" in m.lower() for m in models)
        return has_tinyllama, models
    except Exception as e:
        return False, [f"Error: {str(e)}"]

def check_file_structure() -> Tuple[bool, Dict[str, int]]:
    """Check for problematic file patterns"""
    checks = {
        "markdown_files": "find /opt/sutazaiapp -type f -name '*.md' | wc -l",
        "requirements_files": "find /opt/sutazaiapp -type f -name 'requirements*.txt' | wc -l",
        "base_agents": "find /opt/sutazaiapp -type f -name '*base_agent*.py' | wc -l",
        "test_files_root": "find /opt/sutazaiapp -maxdepth 1 -type f -name '*_test.py' | wc -l"
    }
    
    counts = {}
    for name, cmd in checks.items():
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            counts[name] = int(result.stdout.strip())
        except:
            counts[name] = -1
    
    # Check if counts are within acceptable ranges
    is_ok = (
        counts.get("markdown_files", 0) < 2000 and
        counts.get("requirements_files", 0) < 50 and
        counts.get("base_agents", 0) < 10
    )
    
    return is_ok, counts

def main():
    """Run all health checks"""
    print("=" * 60)
    print("SUTAZAI SYSTEM HEALTH VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    all_healthy = True
    
    # Check Docker services
    print("\n1. Docker Services:")
    docker_ok, docker_issues = check_docker_services()
    if docker_ok:
        print("   ✅ All critical services running")
    else:
        print(f"   ❌ Missing services: {', '.join(docker_issues)}")
        all_healthy = False
    
    # Check backend health
    print("\n2. Backend API:")
    backend_ok, backend_data = check_backend_health()
    if backend_ok:
        print(f"   ✅ Backend healthy (v{backend_data.get('version', 'unknown')})")
        services = backend_data.get("services", {})
        print(f"   - Ollama: {services.get('ollama', 'unknown')}")
        print(f"   - Database: {services.get('database', 'unknown')}")
        print(f"   - Redis: {services.get('redis', 'unknown')}")
    else:
        print(f"   ❌ Backend unhealthy: {backend_data.get('error', 'Unknown error')}")
        all_healthy = False
    
    # Check Ollama models
    print("\n3. Ollama Models:")
    ollama_ok, models = check_ollama_models()
    if ollama_ok:
        print(f"   ✅ TinyLlama loaded")
        print(f"   Models: {', '.join(models)}")
    else:
        print(f"   ❌ TinyLlama not found")
        all_healthy = False
    
    # Check file structure
    print("\n4. File Structure Analysis:")
    files_ok, file_counts = check_file_structure()
    print(f"   - Markdown files: {file_counts.get('markdown_files', 'unknown')}")
    print(f"   - Requirements files: {file_counts.get('requirements_files', 'unknown')}")
    print(f"   - BaseAgent implementations: {file_counts.get('base_agents', 'unknown')}")
    print(f"   - Root test files: {file_counts.get('test_files_root', 'unknown')}")
    
    if not files_ok:
        print("   ⚠️  File structure needs cleanup")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_healthy:
        print("✅ SYSTEM HEALTHY - Safe to proceed with changes")
        print("Recommendation: Create backup before making changes")
        return 0
    else:
        print("❌ SYSTEM ISSUES DETECTED - Fix before proceeding")
        print("Recommendation: Review issues and restore health first")
        return 1

if __name__ == "__main__":
    sys.exit(main())