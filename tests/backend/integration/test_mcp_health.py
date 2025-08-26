#!/usr/bin/env python3
"""
Test script to investigate MCP health endpoint response format mismatch
"""
import asyncio
import json
import sys
import os
sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.mesh.mcp_bridge import MCPMeshBridge
from app.mesh.service_mesh import ServiceMesh

async def test_mcp_health():
    """Test what the MCP bridge health_check_all actually returns"""
    try:
        # Initialize service mesh
        mesh = ServiceMesh()
        
        # Initialize MCP bridge
        bridge = MCPMeshBridge(mesh)
        
        # Get health status
        health = await bridge.health_check_all()
        
        print("=== Raw health_check_all() response ===")
        print(json.dumps(health, indent=2, default=str))
        print("\n=== Response structure ===")
        print(f"Keys: {list(health.keys())}")
        
        if 'services' in health:
            print(f"Services count: {len(health['services'])}")
            if health['services']:
                first_service = list(health['services'].keys())[0]
                print(f"Example service '{first_service}': {health['services'][first_service]}")
        
        if 'summary' in health:
            print(f"Summary: {health['summary']}")
            
        return health
        
    except Exception as e:
        print(f"Error testing MCP health: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_mcp_health())