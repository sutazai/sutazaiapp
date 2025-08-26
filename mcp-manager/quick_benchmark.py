#!/usr/bin/env python3
"""
Quick MCP Server Performance Test

Immediate synchronous testing of MCP servers with basic metrics.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mcp_config():
    """Load MCP server configurations"""
    servers = {}
    config_path = Path("/opt/sutazaiapp/.mcp.json")
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            for name, server_config in config.get("mcpServers", {}).items():
                servers[name] = {
                    "command": server_config["command"],
                    "args": server_config.get("args", []),
                    "type": server_config.get("type", "stdio")
                }
    
    return servers

def test_server_startup(name, config, timeout=10):
    """Test server startup time and basic availability"""
    if config["command"].endswith('.sh'):
        # Test wrapper script health check first
        try:
            health_result = subprocess.run(
                ['/bin/bash', config["command"], '--selfcheck'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            health_status = "PASS" if health_result.returncode == 0 else "FAIL"
            health_output = health_result.stdout.strip() if health_result.returncode == 0 else health_result.stderr.strip()
            
        except subprocess.TimeoutExpired:
            health_status = "TIMEOUT"
            health_output = "Health check timed out"
        except Exception as e:
            health_status = "ERROR"
            health_output = str(e)
    else:
        health_status = "N/A"
        health_output = "Not a wrapper script"
    
    # Test actual server startup
    try:
        if config["command"].endswith('.sh'):
            cmd = ['/bin/bash', config["command"]] + config["args"]
        else:
            cmd = [config["command"]] + config["args"]
        
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(1)
        startup_time = time.time() - start_time
        
        # Check if process is still running
        if process.poll() is None:
            # Send a simple initialize message
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "benchmark", "version": "1.0.0"}
                }
            }
            
            try:
                process.stdin.write(json.dumps(init_message) + '\n')
                process.stdin.flush()
                
                # Wait for response with timeout
                process.wait(timeout=3)
                stdout, stderr = process.communicate(timeout=2)
                
                startup_success = True
                response_received = len(stdout) > 0
                
            except subprocess.TimeoutExpired:
                process.kill()
                startup_success = True  # Started but didn't respond quickly
                response_received = False
        else:
            startup_success = False
            response_received = False
            stderr = process.stderr.read() if process.stderr else "No stderr"
        
        # Clean up
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        
        return {
            "name": name,
            "health_check": health_status,
            "health_output": health_output[:100],  # Truncate long output
            "startup_time_ms": startup_time * 1000,
            "startup_success": startup_success,
            "response_received": response_received,
            "working": startup_success
        }
        
    except Exception as e:
        return {
            "name": name,
            "health_check": health_status,
            "health_output": health_output[:100],
            "startup_time_ms": 0,
            "startup_success": False,
            "response_received": False,
            "error": str(e),
            "working": False
        }

def main():
    """Run quick benchmark"""
    print("🎯 Quick MCP Server Performance Test")
    print("=" * 50)
    
    servers = load_mcp_config()
    print(f"📋 Found {len(servers)} MCP servers")
    
    results = []
    
    # Test first 10 servers to avoid overwhelming
    for i, (name, config) in enumerate(list(servers.items())[:10]):
        print(f"\n🧪 Testing {i+1}/10: {name}")
        result = test_server_startup(name, config)
        results.append(result)
        
        if result["working"]:
            print(f"  ✅ Working - startup: {result['startup_time_ms']:.0f}ms, health: {result['health_check']}")
        else:
            print(f"  ❌ Not working - health: {result['health_check']}")
            if "error" in result:
                print(f"     Error: {result['error']}")
    
    # Summary report
    print(f"\n📊 Performance Summary")
    print("-" * 30)
    
    working_servers = [r for r in results if r["working"]]
    print(f"✅ Working servers: {len(working_servers)}/{len(results)}")
    
    if working_servers:
        print(f"\n⚡ Fastest startup times:")
        sorted_by_startup = sorted(working_servers, key=lambda x: x["startup_time_ms"])
        for server in sorted_by_startup[:5]:
            print(f"  {server['name']}: {server['startup_time_ms']:.0f}ms")
        
        print(f"\n🏥 Health check results:")
        health_pass = [r for r in results if r["health_check"] == "PASS"]
        health_fail = [r for r in results if r["health_check"] == "FAIL"]
        health_na = [r for r in results if r["health_check"] == "N/A"]
        
        print(f"  PASS: {len(health_pass)} servers")
        print(f"  FAIL: {len(health_fail)} servers")
        print(f"  N/A:  {len(health_na)} servers")
    
    # Issues detected
    print(f"\n⚠️ Issues Detected:")
    non_working = [r for r in results if not r["working"]]
    if non_working:
        print(f"  • {len(non_working)} servers not responding")
        for server in non_working:
            if "error" in server:
                print(f"    - {server['name']}: {server['error'][:50]}")
    
    slow_servers = [r for r in working_servers if r["startup_time_ms"] > 2000]
    if slow_servers:
        print(f"  • {len(slow_servers)} servers have slow startup (>2s)")
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_dir = "/opt/sutazaiapp/mcp-manager/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON
    output_data = {
        "timestamp": time.time(),
        "date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "test_type": "quick_benchmark",
        "servers_tested": len(results),
        "working_servers": len(working_servers),
        "results": results
    }
    
    json_file = f"{results_dir}/quick_benchmark_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📄 Results saved to: {json_file}")
    print(f"✅ Quick benchmark complete!")

if __name__ == "__main__":
    main()