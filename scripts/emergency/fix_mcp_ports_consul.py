#!/usr/bin/env python3
"""
Fix MCP port configuration in Consul
Updates service registrations to use actual ports (3001-3019) instead of fictional ports (11100+)
"""
import requests
import json
import time

# Actual MCP container port mappings (verified from docker ps)
MCP_PORT_MAPPING = {
    "mcp-claude-flow": 3001,
    "mcp-ruv-swarm": 3002,
    "mcp-files": 3003,
    "mcp-context7": 3004,
    "mcp-http-fetch": 3005,
    "mcp-ddg": 3006,
    "mcp-sequentialthinking": 3007,
    "mcp-nx-mcp": 3008,
    "mcp-extended-memory": 3009,
    "mcp-mcp-ssh": 3010,
    "mcp-ultimatecoder": 3011,
    "mcp-playwright-mcp": 3012,
    "mcp-memory-bank-mcp": 3013,
    "mcp-knowledge-graph-mcp": 3014,
    "mcp-compass-mcp": 3015,
    "mcp-github": 3016,
    "mcp-http": 3017,
    "mcp-language-server": 3018,
    "mcp-claude-task-runner": 3019
}

CONSUL_URL = "http://localhost:10006"

def fix_mcp_consul_ports():
    """Fix MCP service port registrations in Consul"""
    print("🔧 Fixing MCP port configurations in Consul...")
    
    # First, get current services
    try:
        response = requests.get(f"{CONSUL_URL}/v1/agent/services")
        current_services = response.json()
        print(f"📋 Found {len(current_services)} services in Consul")
    except Exception as e:
        print(f"❌ Failed to connect to Consul: {e}")
        return False
    
    # Deregister old services with wrong ports
    for service_id, service_info in current_services.items():
        if service_id.startswith("mcp-"):
            print(f"🗑️ Deregistering old service: {service_id}")
            try:
                requests.put(f"{CONSUL_URL}/v1/agent/service/deregister/{service_id}")
            except Exception as e:
                print(f"⚠️ Failed to deregister {service_id}: {e}")
    
    time.sleep(2)
    
    # Register services with correct ports
    success_count = 0
    for service_name, port in MCP_PORT_MAPPING.items():
        service_config = {
            "ID": service_name,
            "Name": service_name,
            "Address": "172.20.0.22",  # DinD container IP
            "Port": port,
            "Tags": ["mcp", "ai", "protocol"],
            "Meta": {
                "version": "1.0",
                "type": "mcp-server",
                "capabilities": ["ai", "automation"]
            },
            "Check": {
                "HTTP": f"http://172.20.0.22:{port}/health",
                "Interval": "10s",
                "Timeout": "5s",
                "DeregisterCriticalServiceAfter": "1m"
            }
        }
        
        try:
            response = requests.put(
                f"{CONSUL_URL}/v1/agent/service/register",
                json=service_config
            )
            if response.status_code == 200:
                print(f"✅ Registered {service_name} on port {port}")
                success_count += 1
            else:
                print(f"❌ Failed to register {service_name}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error registering {service_name}: {e}")
    
    print(f"\n🎯 Successfully registered {success_count}/{len(MCP_PORT_MAPPING)} MCP services")
    
    # Verify registrations
    time.sleep(3)
    try:
        response = requests.get(f"{CONSUL_URL}/v1/agent/services")
        updated_services = response.json()
        mcp_services = {k: v for k, v in updated_services.items() if k.startswith("mcp-")}
        print(f"✅ Verification: {len(mcp_services)} MCP services now registered")
        
        # Show service health
        print("\n📊 Service Health Status:")
        for service_id, service_info in mcp_services.items():
            port = service_info.get('Port', 'unknown')
            print(f"  {service_id}: Port {port}")
            
    except Exception as e:
        print(f"⚠️ Failed to verify registrations: {e}")
    
    return success_count == len(MCP_PORT_MAPPING)

if __name__ == "__main__":
    success = fix_mcp_consul_ports()
    if success:
        print("\n🎉 MCP port configuration fixed successfully!")
        print("All services now registered with correct ports (3001-3019)")
    else:
        print("\n⚠️ Some issues occurred during port configuration fix")