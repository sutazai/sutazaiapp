#!/usr/bin/env python3
"""
Quick MCP Server Validation Script

Fast health check and basic functionality test for all MCP servers.
Designed for CI/CD pipelines and quick validation.

Usage:
    python3 quick_test.py                    # Basic health checks
    python3 quick_test.py --full            # Include protocol tests
    python3 quick_test.py --server <name>   # Test specific server
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Import our test utilities
from test_mcp_servers import load_mcp_servers, MCPServerProcess, MCPProtocolTester


def run_health_checks() -> Dict[str, bool]:
    """Run health checks on all wrapper scripts"""
    print("🏥 Running Health Checks...")
    results = {}
    servers = load_mcp_servers()
    
    for server in servers:
        if server.command.endswith('.sh'):
            try:
                result = subprocess.run(
                    ['/bin/bash', server.command, '--selfcheck'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                results[server.name] = result.returncode == 0
                status = "✅ PASS" if result.returncode == 0 else "❌ FAIL"
                print(f"  {status} {server.name}")
                
            except subprocess.TimeoutExpired:
                results[server.name] = False
                print(f"  ⏰ TIMEOUT {server.name}")
            except Exception as e:
                results[server.name] = False
                print(f"  🔥 ERROR {server.name}: {e}")
    
    return results


async def test_server_startup(server_names: List[str] = None) -> Dict[str, bool]:
    """Test server startup for specified servers"""
    print("\n🚀 Testing Server Startup...")
    servers = load_mcp_servers()
    results = {}
    
    if server_names:
        test_servers = [s for s in servers if s.name in server_names]
    else:
        # Test first 3 enabled servers for quick validation
        test_servers = [s for s in servers[:5] if s.enabled][:3]
    
    for server in test_servers:
        print(f"  Testing {server.name}...")
        server_process = MCPServerProcess(server)
        
        try:
            startup_success = await server_process.start()
            results[server.name] = startup_success
            
            if startup_success:
                print(f"    ✅ {server.name} started successfully")
                # Give it a moment then stop
                await asyncio.sleep(0.5)
            else:
                print(f"    ❌ {server.name} failed to start")
        except Exception as e:
            results[server.name] = False
            print(f"    🔥 {server.name} error: {e}")
        finally:
            server_process.stop()
    
    return results


async def test_mcp_protocol(server_names: List[str] = None) -> Dict[str, bool]:
    """Test MCP protocol compliance"""
    print("\n📡 Testing MCP Protocol...")
    servers = load_mcp_servers()
    results = {}
    protocol_tester = MCPProtocolTester()
    
    if server_names:
        test_servers = [s for s in servers if s.name in server_names]
    else:
        # Test one known working server
        test_servers = []
        for server in servers:
            if 'extended-memory' in server.name or 'claude-flow' in server.name:
                test_servers.append(server)
                break
    
    for server in test_servers[:2]:  # Max 2 for quick test
        print(f"  Testing {server.name}...")
        server_process = MCPServerProcess(server)
        
        try:
            if not await server_process.start():
                results[server.name] = False
                print(f"    ❌ {server.name} failed to start")
                continue
            
            # Test initialize
            init_request = protocol_tester.create_initialize_request()
            response = await server_process.send_message(init_request, timeout=5)
            
            if response and protocol_tester.validate_mcp_response(response):
                results[server.name] = True
                print(f"    ✅ {server.name} MCP protocol OK")
            else:
                results[server.name] = False
                print(f"    ❌ {server.name} invalid MCP response")
                
        except Exception as e:
            results[server.name] = False
            print(f"    🔥 {server.name} protocol error: {e}")
        finally:
            server_process.stop()
    
    return results


def print_summary(health_results: Dict[str, bool], 
                 startup_results: Dict[str, bool] = None,
                 protocol_results: Dict[str, bool] = None):
    """Print test summary"""
    print("\n" + "="*50)
    print("📊 QUICK TEST SUMMARY")
    print("="*50)
    
    # Health checks
    passed_health = sum(1 for r in health_results.values() if r)
    total_health = len(health_results)
    health_pct = (passed_health / total_health * 100) if total_health > 0 else 0
    print(f"Health Checks: {passed_health}/{total_health} ({health_pct:.1f}%)")
    
    # Startup tests
    if startup_results:
        passed_startup = sum(1 for r in startup_results.values() if r)
        total_startup = len(startup_results)
        startup_pct = (passed_startup / total_startup * 100) if total_startup > 0 else 0
        print(f"Startup Tests: {passed_startup}/{total_startup} ({startup_pct:.1f}%)")
    
    # Protocol tests
    if protocol_results:
        passed_protocol = sum(1 for r in protocol_results.values() if r)
        total_protocol = len(protocol_results)
        protocol_pct = (passed_protocol / total_protocol * 100) if total_protocol > 0 else 0
        print(f"Protocol Tests: {passed_protocol}/{total_protocol} ({protocol_pct:.1f}%)")
    
    # Overall status
    all_results = [health_results]
    if startup_results:
        all_results.append(startup_results)
    if protocol_results:
        all_results.append(protocol_results)
    
    total_passed = sum(sum(1 for r in results.values() if r) for results in all_results)
    total_tests = sum(len(results) for results in all_results)
    overall_pct = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    if overall_pct >= 80:
        status = "🟢 EXCELLENT"
    elif overall_pct >= 60:
        status = "🟡 GOOD"
    elif overall_pct >= 40:
        status = "🟠 FAIR"
    else:
        status = "🔴 POOR"
    
    print(f"\nOverall Score: {total_passed}/{total_tests} ({overall_pct:.1f}%) {status}")
    
    # Return code for CI/CD
    return 0 if overall_pct >= 70 else 1


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick MCP Server Validation')
    parser.add_argument('--full', action='store_true', help='Include protocol tests')
    parser.add_argument('--server', type=str, help='Test specific server only')
    parser.add_argument('--no-health', action='store_true', help='Skip health checks')
    parser.add_argument('--timeout', type=int, default=30, help='Overall timeout in seconds')
    
    args = parser.parse_args()
    
    print("⚡ Quick MCP Server Test Suite")
    print("=" * 30)
    
    start_time = time.time()
    
    try:
        # Health checks (fast)
        health_results = {}
        if not args.no_health:
            health_results = run_health_checks()
        
        # Startup tests
        server_list = [args.server] if args.server else None
        startup_results = await test_server_startup(server_list)
        
        # Protocol tests (if requested)
        protocol_results = {}
        if args.full:
            protocol_results = await test_mcp_protocol(server_list)
        
        # Summary
        duration = time.time() - start_time
        print(f"\n⏱️  Tests completed in {duration:.1f}s")
        
        exit_code = print_summary(health_results, startup_results, protocol_results)
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n🔥 Test suite error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)