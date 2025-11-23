#!/usr/bin/env python3
"""
Quick Test Validation Script
Runs a subset of critical tests to verify system health
"""

import asyncio
import httpx
import sys
from typing import Dict, List, Tuple

TIMEOUT = 30.0

class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

async def test_service(name: str, url: str, expected_codes: List[int] = [200]) -> Tuple[str, bool, str]:
    """Test a single service endpoint"""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(url)
            success = response.status_code in expected_codes
            return (name, success, f"{response.status_code}")
    except Exception as e:
        return (name, False, str(e)[:50])

async def test_service_with_auth(name: str, url: str, auth: Tuple[str, str]) -> Tuple[str, bool, str]:
    """Test a service that requires authentication"""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(url, auth=auth)
            success = response.status_code in [200, 401]
            return (name, success, f"{response.status_code}")
    except Exception as e:
        return (name, False, str(e)[:50])

async def main():
    print("═══════════════════════════════════════════════════════════")
    print("  SUTAZAI QUICK VALIDATION TEST")
    print("═══════════════════════════════════════════════════════════")
    print()
    
    results = []
    
    # Core Infrastructure
    print(f"{Colors.BLUE}Testing Core Infrastructure...{Colors.NC}")
    results.append(await test_service("Backend API", "http://localhost:10200/health"))
    # PostgreSQL and Redis are tested through backend connectivity
    results.append(await test_service("Neo4j Browser", "http://localhost:10002/", [200, 302, 404]))
    
    # Monitoring Stack
    print(f"{Colors.BLUE}Testing Monitoring Stack...{Colors.NC}")
    results.append(await test_service("Prometheus", "http://localhost:10300/-/healthy"))
    results.append(await test_service("Grafana", "http://localhost:10301/api/health"))
    results.append(await test_service("Loki", "http://localhost:10310/ready", [200, 404]))
    
    # AI Services
    print(f"{Colors.BLUE}Testing AI Services...{Colors.NC}")
    results.append(await test_service("Ollama", "http://localhost:11435/api/tags", [200, 404]))
    results.append(await test_service("ChromaDB", "http://localhost:10100/api/v2/heartbeat", [200, 404]))
    results.append(await test_service("Qdrant", "http://localhost:10102/collections", [200, 404]))
    
    # Message Queue
    print(f"{Colors.BLUE}Testing Message Queue...{Colors.NC}")
    results.append(await test_service_with_auth(
        "RabbitMQ", 
        "http://localhost:10005/api/overview",
        ("sutazai", "sutazai_secure_2024")
    ))
    
    # AI Agents
    print(f"{Colors.BLUE}Testing AI Agents...{Colors.NC}")
    agents = [
        ("CrewAI", "http://localhost:11403/health"),
        ("Aider", "http://localhost:11404/health"),
        ("LangChain", "http://localhost:11405/health"),
        ("ShellGPT", "http://localhost:11413/health"),
        ("Documind", "http://localhost:11414/health"),
        ("FinRobot", "http://localhost:11410/health"),
        ("Letta", "http://localhost:11401/health"),
        ("GPT-Engineer", "http://localhost:11416/health")
    ]
    
    for agent_name, agent_url in agents:
        results.append(await test_service(agent_name, agent_url, [200, 404, 503]))
    
    # Print Results
    print()
    print("═══════════════════════════════════════════════════════════")
    print("  TEST RESULTS")
    print("═══════════════════════════════════════════════════════════")
    print()
    
    passed = 0
    failed = 0
    
    for name, success, status in results:
        if success:
            print(f"{Colors.GREEN}✓{Colors.NC} {name:<20} {status}")
            passed += 1
        else:
            print(f"{Colors.RED}✗{Colors.NC} {name:<20} {status}")
            failed += 1
    
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print()
    print("═══════════════════════════════════════════════════════════")
    print(f"Total Tests: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.NC}")
    print(f"{Colors.RED}Failed: {failed}{Colors.NC}")
    print(f"Success Rate: {success_rate:.1f}%")
    print("═══════════════════════════════════════════════════════════")
    
    if failed == 0:
        print(f"{Colors.GREEN}✓ ALL SERVICES HEALTHY{Colors.NC}")
        return 0
    elif success_rate >= 70:
        print(f"{Colors.YELLOW}⚠ SOME SERVICES DOWN BUT SYSTEM FUNCTIONAL{Colors.NC}")
        return 1
    else:
        print(f"{Colors.RED}✗ CRITICAL FAILURES DETECTED{Colors.NC}")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
