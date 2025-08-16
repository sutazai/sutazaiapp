#!/usr/bin/env python3
"""
Test MCP service initialization and registration
"""
import asyncio
import json
import sys
import os
sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.mesh.mcp_bridge import MCPMeshBridge
from app.mesh.service_mesh import ServiceMesh

async def test_mcp_initialization():
    """Test MCP service initialization from .mcp.json"""
    try:
        print("=== Testing MCP Service Initialization ===\n")
        
        # Initialize service mesh (optional)
        mesh = ServiceMesh()
        print(f"✓ Service mesh initialized")
        
        # Initialize MCP bridge
        bridge = MCPMeshBridge(mesh)
        print(f"✓ MCP bridge created")
        
        # Load and display MCP configuration
        config_file = "/opt/sutazaiapp/.mcp.json"
        with open(config_file, 'r') as f:
            mcp_config = json.load(f)
        
        mcp_servers = mcp_config.get("mcpServers", {})
        print(f"\n📋 Found {len(mcp_servers)} MCP servers in configuration:")
        for name in mcp_servers.keys():
            wrapper_path = f"/opt/sutazaiapp/scripts/mcp/wrappers/{name}.sh"
            exists = "✓" if os.path.exists(wrapper_path) else "✗"
            print(f"  {exists} {name}: {wrapper_path}")
        
        # Initialize services
        print(f"\n🚀 Initializing MCP services...")
        init_result = await bridge.initialize_services()
        print(f"  Initialization result: {init_result}")
        
        # Get health status
        print(f"\n💊 Checking health status...")
        health = await bridge.health_check_all()
        
        print(f"\n=== Health Check Results ===")
        print(f"Total services: {health['summary']['total']}")
        print(f"Healthy: {health['summary']['healthy']}")
        print(f"Unhealthy: {health['summary']['unhealthy']}")
        print(f"Health percentage: {health['summary']['percentage_healthy']:.1f}%")
        
        if health['services']:
            print(f"\n=== Service Details ===")
            for name, status in health['services'].items():
                health_icon = "✓" if status['healthy'] else "✗"
                print(f"{health_icon} {name}:")
                print(f"  - Available: {status['available']}")
                print(f"  - Process running: {status['process_running']}")
                print(f"  - Retry count: {status['retry_count']}")
        else:
            print("\n⚠️ No services registered")
        
        # List available services
        services = await bridge.list_services()
        print(f"\n📜 Available MCP services: {services}")
        
        return health
        
    except Exception as e:
        print(f"\n❌ Error during MCP initialization test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_mcp_initialization())