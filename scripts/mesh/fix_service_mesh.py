#!/usr/bin/env python3
"""
Service Mesh Fix Script
Repairs Kong-Consul-MCP integration issues
"""

import subprocess
import json
import time
import sys
import requests
from typing import Dict, List, Any

# Configuration
CONSUL_URL = "http://localhost:10006"
KONG_ADMIN_URL = "http://localhost:10015"
BACKEND_URL = "http://localhost:10010"
MCP_ORCHESTRATOR = "sutazai-mcp-orchestrator"

# MCP service port mappings - actual DinD container ports
MCP_SERVICES = {
    "mcp-claude-flow": {"dind_port": 3001, "host_port": 11100, "health_path": "/health"},
    "mcp-files": {"dind_port": 3003, "host_port": 11108, "health_path": "/health"},
    "mcp-context7": {"dind_port": 3004, "host_port": 11107, "health_path": "/health"},
}

def run_cmd(cmd: str) -> str:
    """Execute shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return ""

def get_dind_container_ip() -> str:
    """Get IP address of DinD orchestrator container"""
    cmd = f"docker inspect {MCP_ORCHESTRATOR} --format '{{{{.NetworkSettings.Networks.sutazai-network.IPAddress}}}}'"
    ip = run_cmd(cmd)
    if not ip:
        # Try alternative network
        cmd = f"docker inspect {MCP_ORCHESTRATOR} --format '{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}'"
        ip = run_cmd(cmd).split()[0] if run_cmd(cmd) else ""
    return ip

def get_mcp_containers() -> List[Dict[str, Any]]:
    """Get list of MCP containers running in DinD"""
    cmd = f"docker exec {MCP_ORCHESTRATOR} docker ps --format '{{{{json .}}}}'"
    output = run_cmd(cmd)
    containers = []
    for line in output.split('\n'):
        if line.strip():
            try:
                container = json.loads(line)
                if 'mcp-' in container.get('Names', ''):
                    containers.append(container)
            except json.JSONDecodeError:
                continue
    return containers

def setup_port_forwarding():
    """Setup port forwarding from host to DinD containers"""
    print("\n1. Setting up port forwarding from host to DinD containers...")
    
    dind_ip = get_dind_container_ip()
    if not dind_ip:
        print("ERROR: Could not get DinD orchestrator IP")
        return False
    
    print(f"   DinD orchestrator IP: {dind_ip}")
    
    # Setup iptables rules for port forwarding
    for service_name, config in MCP_SERVICES.items():
        host_port = config["host_port"]
        dind_port = config["dind_port"]
        
        # Remove existing rules
        run_cmd(f"sudo iptables -t nat -D PREROUTING -p tcp --dport {host_port} -j DNAT --to-destination {dind_ip}:{dind_port} 2>/dev/null")
        run_cmd(f"sudo iptables -t nat -D POSTROUTING -p tcp -d {dind_ip} --dport {dind_port} -j MASQUERADE 2>/dev/null")
        
        # Add new rules
        cmd1 = f"sudo iptables -t nat -A PREROUTING -p tcp --dport {host_port} -j DNAT --to-destination {dind_ip}:{dind_port}"
        cmd2 = f"sudo iptables -t nat -A POSTROUTING -p tcp -d {dind_ip} --dport {dind_port} -j MASQUERADE"
        
        run_cmd(cmd1)
        run_cmd(cmd2)
        print(f"   ✓ Port forwarding setup: {host_port} -> {dind_ip}:{dind_port} ({service_name})")
    
    # Enable IP forwarding
    run_cmd("sudo sysctl -w net.ipv4.ip_forward=1")
    
    return True

def deregister_old_services():
    """Deregister stale MCP services from Consul"""
    print("\n2. Cleaning up stale Consul registrations...")
    
    try:
        response = requests.get(f"{CONSUL_URL}/v1/agent/services")
        services = response.json()
        
        for service_id, service_info in services.items():
            if service_id.startswith("mcp-"):
                requests.put(f"{CONSUL_URL}/v1/agent/service/deregister/{service_id}")
                print(f"   ✓ Deregistered: {service_id}")
    except Exception as e:
        print(f"   WARNING: Could not clean Consul services: {e}")

def register_mcp_services():
    """Register MCP services in Consul with correct endpoints"""
    print("\n3. Registering MCP services in Consul...")
    
    dind_ip = get_dind_container_ip()
    if not dind_ip:
        print("ERROR: Could not get DinD IP")
        return False
    
    for service_name, config in MCP_SERVICES.items():
        service_def = {
            "ID": service_name,
            "Name": service_name,
            "Tags": ["mcp", "ai", "dind"],
            "Address": dind_ip,
            "Port": config["dind_port"],
            "Check": {
                "HTTP": f"http://{dind_ip}:{config['dind_port']}{config['health_path']}",
                "Interval": "30s",
                "Timeout": "5s"
            }
        }
        
        try:
            response = requests.put(
                f"{CONSUL_URL}/v1/agent/service/register",
                json=service_def
            )
            if response.status_code == 200:
                print(f"   ✓ Registered: {service_name} at {dind_ip}:{config['dind_port']}")
            else:
                print(f"   ✗ Failed to register {service_name}: {response.text}")
        except Exception as e:
            print(f"   ✗ Error registering {service_name}: {e}")
    
    return True

def update_kong_services():
    """Update Kong gateway configuration for MCP services"""
    print("\n4. Configuring Kong gateway routes...")
    
    dind_ip = get_dind_container_ip()
    if not dind_ip:
        print("ERROR: Could not get DinD IP")
        return False
    
    for service_name, config in MCP_SERVICES.items():
        # Create Kong service
        service_data = {
            "name": service_name,
            "url": f"http://{dind_ip}:{config['dind_port']}"
        }
        
        try:
            # Check if service exists
            response = requests.get(f"{KONG_ADMIN_URL}/services/{service_name}")
            if response.status_code == 200:
                # Update existing service
                response = requests.patch(
                    f"{KONG_ADMIN_URL}/services/{service_name}",
                    json=service_data
                )
            else:
                # Create new service
                response = requests.post(
                    f"{KONG_ADMIN_URL}/services",
                    json=service_data
                )
            
            if response.status_code in [200, 201]:
                print(f"   ✓ Kong service configured: {service_name}")
                
                # Create route for the service
                route_data = {
                    "name": f"{service_name}-route",
                    "paths": [f"/{service_name.replace('mcp-', '')}"],
                    "service": {"name": service_name}
                }
                
                # Check if route exists
                route_response = requests.get(f"{KONG_ADMIN_URL}/routes/{service_name}-route")
                if route_response.status_code == 200:
                    # Update existing route
                    requests.patch(
                        f"{KONG_ADMIN_URL}/routes/{service_name}-route",
                        json=route_data
                    )
                else:
                    # Create new route
                    requests.post(
                        f"{KONG_ADMIN_URL}/routes",
                        json=route_data
                    )
                print(f"   ✓ Kong route configured: /{service_name.replace('mcp-', '')}")
            else:
                print(f"   ✗ Failed to configure Kong service {service_name}: {response.text}")
        except Exception as e:
            print(f"   ✗ Error configuring Kong for {service_name}: {e}")
    
    return True

def verify_connectivity():
    """Test connectivity to MCP services"""
    print("\n5. Verifying service connectivity...")
    
    dind_ip = get_dind_container_ip()
    results = {}
    
    # Test direct DinD connectivity
    print("\n   Testing direct DinD connectivity:")
    for service_name, config in MCP_SERVICES.items():
        try:
            url = f"http://{dind_ip}:{config['dind_port']}{config['health_path']}"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"   ✓ {service_name}: Direct connection OK ({dind_ip}:{config['dind_port']})")
                results[service_name] = "OK"
            else:
                print(f"   ✗ {service_name}: Direct connection failed (status {response.status_code})")
                results[service_name] = f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            print(f"   ✗ {service_name}: Direct connection failed ({str(e)[:50]})")
            results[service_name] = "Connection failed"
    
    # Test through Kong gateway
    print("\n   Testing through Kong gateway:")
    for service_name in MCP_SERVICES.keys():
        try:
            path = service_name.replace('mcp-', '')
            url = f"http://localhost:10005/{path}/health"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"   ✓ {service_name}: Kong proxy OK")
            else:
                print(f"   ✗ {service_name}: Kong proxy failed (status {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"   ✗ {service_name}: Kong proxy failed ({str(e)[:50]})")
    
    # Test Consul health checks
    print("\n   Checking Consul service health:")
    try:
        response = requests.get(f"{CONSUL_URL}/v1/health/state/any")
        health_checks = response.json()
        for check in health_checks:
            if check.get("ServiceID", "").startswith("mcp-"):
                status = check.get("Status", "unknown")
                service = check.get("ServiceID", "unknown")
                symbol = "✓" if status == "passing" else "✗"
                print(f"   {symbol} {service}: {status}")
    except Exception as e:
        print(f"   ✗ Could not check Consul health: {e}")
    
    return results

def main():
    """Main execution flow"""
    print("=" * 60)
    print("SERVICE MESH FIX SCRIPT")
    print("Repairing Kong-Consul-MCP Integration")
    print("=" * 60)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    # Check if DinD orchestrator is running
    result = run_cmd(f"docker ps -q -f name={MCP_ORCHESTRATOR}")
    if not result:
        print(f"ERROR: {MCP_ORCHESTRATOR} is not running!")
        print("Please start the MCP orchestrator first.")
        sys.exit(1)
    
    # Check MCP containers in DinD
    mcp_containers = get_mcp_containers()
    print(f"Found {len(mcp_containers)} MCP containers in DinD")
    for container in mcp_containers:
        print(f"  - {container.get('Names', 'unknown')}: {container.get('Status', 'unknown')}")
    
    # Execute fixes
    steps = [
        ("Port Forwarding", setup_port_forwarding),
        ("Consul Cleanup", deregister_old_services),
        ("Service Registration", register_mcp_services),
        ("Kong Configuration", update_kong_services),
    ]
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            if result is False:
                print(f"\n✗ {step_name} failed!")
                sys.exit(1)
        except Exception as e:
            print(f"\n✗ {step_name} error: {e}")
            sys.exit(1)
    
    # Verify everything works
    print("\n" + "=" * 60)
    results = verify_connectivity()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = sum(1 for v in results.values() if v == "OK")
    total = len(results)
    
    if working == total:
        print(f"✓ SUCCESS: All {total} MCP services are accessible")
        print("\nService mesh is now operational!")
        print("\nAccess MCP services through Kong:")
        for service_name in MCP_SERVICES.keys():
            path = service_name.replace('mcp-', '')
            print(f"  - http://localhost:10005/{path}")
    else:
        print(f"⚠ PARTIAL SUCCESS: {working}/{total} services are accessible")
        print("\nSome issues remain. Please check the logs above.")
    
    print("\nConsul UI: http://localhost:10006")
    print("Kong Admin: http://localhost:10015")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()