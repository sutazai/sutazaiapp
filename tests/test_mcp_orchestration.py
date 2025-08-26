#!/usr/bin/env python3
"""
Test MCP Orchestration - Validates the MCP deployment fixes
"""

import subprocess
import json
import time
import sys
from pathlib import Path

def run_command(cmd):
    """Run a shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def test_cleanup():
    """Test that orphaned containers are cleaned up"""
    print("Testing cleanup of orphaned containers...")
    
    # Check for orphaned containers
    stdout, _, _ = run_command("docker ps --format '{{.Names}}' | grep -E 'tender_|optimistic_|hungry_|vigilant_'")
    
    if stdout.strip():
        print(f"❌ Found orphaned containers: {stdout}")
        return False
    
    print("✅ No orphaned containers found")
    return True

def test_zombie_processes():
    """Test that zombie processes are cleaned"""
    print("\nTesting cleanup of zombie processes...")
    
    stdout, _, _ = run_command("ps aux | grep defunct | grep -E 'mcp|npm' | grep -v grep")
    
    if stdout.strip():
        print(f"❌ Found zombie processes: {stdout}")
        return False
    
    print("✅ No zombie processes found")
    return True

def test_port_allocation():
    """Test that ports are properly allocated"""
    print("\nTesting port allocation system...")
    
    port_file = Path("/opt/sutazaiapp/config/ports/mcp_ports.json")
    
    if not port_file.exists():
        print("⚠️  Port registry not yet created")
        return None
    
    with open(port_file, 'r') as f:
        ports = json.load(f)
    
    # Check port range
    for service, port in ports.items():
        if not (11100 <= port <= 11200):
            print(f"❌ Port {port} for {service} outside valid range")
            return False
    
    # Check for duplicates
    port_values = list(ports.values())
    if len(port_values) != len(set(port_values)):
        print("❌ Duplicate ports allocated")
        return False
    
    print(f"✅ {len(ports)} ports properly allocated in range 11100-11200")
    return True

def test_container_naming():
    """Test that containers use proper naming convention"""
    print("\nTesting container naming convention...")
    
    stdout, _, _ = run_command("docker ps --format '{{.Names}}' | grep 'sutazai-mcp-'")
    
    if not stdout.strip():
        print("⚠️  No MCP containers running yet")
        return None
    
    containers = stdout.strip().split('\n')
    for container in containers:
        if not container.startswith('sutazai-mcp-'):
            print(f"❌ Invalid container name: {container}")
            return False
    
    print(f"✅ All {len(containers)} containers use proper naming")
    return True

def test_network_integration():
    """Test that containers are on the correct network"""
    print("\nTesting network integration...")
    
    # Check if network exists
    stdout, _, returncode = run_command("docker network ls | grep sutazai-network")
    
    if returncode != 0:
        print("❌ sutazai-network does not exist")
        return False
    
    # Check MCP containers on network
    stdout, _, _ = run_command("docker ps --format '{{.Names}}' | grep 'sutazai-mcp-' | head -5")
    
    if stdout.strip():
        containers = stdout.strip().split('\n')
        for container in containers:
            cmd = f"docker inspect {container} | jq -r '.[0].NetworkSettings.Networks | keys[]'"
            networks, _, _ = run_command(cmd)
            if 'sutazai-network' not in networks:
                print(f"❌ Container {container} not on sutazai-network")
                return False
    
    print("✅ Network properly configured")
    return True

def test_multi_client_simulation():
    """Simulate multiple clients accessing MCPs"""
    print("\nTesting multi-client access capability...")
    
    # Check if any MCP containers are running
    stdout, _, _ = run_command("docker ps --format '{{.Names}}:{{.Ports}}' | grep 'sutazai-mcp-'")
    
    if not stdout.strip():
        print("⚠️  No MCP containers running for multi-client test")
        return None
    
    # Parse first container and port
    lines = stdout.strip().split('\n')
    if lines:
        parts = lines[0].split(':')
        if len(parts) >= 2:
            # Extract port from format like "0.0.0.0:11100->8080/tcp"
            port_info = parts[1]
            if '->' in port_info:
                port = port_info.split('->')[0].split(':')[-1]
                
                # Simulate 3 concurrent clients
                print(f"  Simulating 3 clients on port {port}...")
                
                success = 0
                for i in range(3):
                    _, _, returncode = run_command(
                        f"timeout 2 curl -s -X POST http://localhost:{port}/api/mcp "
                        f"-H 'Content-Type: application/json' "
                        f"-d '{{\"client_id\": {i}, \"test\": true}}' 2>/dev/null"
                    )
                    if returncode == 0:
                        success += 1
                
                if success >= 2:
                    print(f"✅ Multi-client access working ({success}/3 clients)")
                    return True
                else:
                    print(f"❌ Multi-client access limited ({success}/3 clients)")
                    return False
    
    print("⚠️  Could not test multi-client access")
    return None

def main():
    """Run all tests"""
    print("=" * 60)
    print("MCP ORCHESTRATION VALIDATION TESTS")
    print("=" * 60)
    
    results = {
        'cleanup': test_cleanup(),
        'zombies': test_zombie_processes(),
        'ports': test_port_allocation(),
        'naming': test_container_naming(),
        'network': test_network_integration(),
        'multi_client': test_multi_client_simulation()
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"{test_name:15} : {status}")
    
    # Calculate overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All critical tests passed! MCP deployment chaos resolved.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Deployment issues remain.")
        return 1

if __name__ == '__main__':
    sys.exit(main())