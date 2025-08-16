#!/usr/bin/env python3
"""
Test script to verify MCP-Mesh integration is working
"""
import asyncio
import httpx
import json
import sys

async def test_mesh_discovery():
    """Test that MCPs appear in mesh service discovery"""
    print("\n🔍 Testing MCP visibility in service mesh...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Get all services from mesh
            response = await client.get("http://localhost:10010/api/v1/mesh/v2/services")
            
            if response.status_code == 200:
                services = response.json()
                
                # Count MCP services
                mcp_services = [s for s in services if s.get("service_name", "").startswith("mcp-")]
                
                print(f"\n📊 Service Mesh Status:")
                print(f"  Total services: {len(services)}")
                print(f"  MCP services: {len(mcp_services)}")
                
                if mcp_services:
                    print("\n✅ MCP Services Found in Mesh:")
                    for svc in mcp_services:
                        print(f"  - {svc['service_name']} on port {svc.get('port', 'unknown')}")
                    return True
                else:
                    print("\n❌ No MCP services found in mesh!")
                    return False
            else:
                print(f"❌ Failed to query mesh: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error querying mesh: {e}")
            return False

async def test_mcp_health():
    """Test MCP health endpoints through mesh"""
    print("\n🏥 Testing MCP health checks...")
    
    mcp_ports = range(11100, 11117)  # 17 MCP services
    healthy = 0
    unhealthy = 0
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for port in mcp_ports:
            try:
                response = await client.get(f"http://localhost:{port}/health")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        healthy += 1
                        print(f"  ✅ Port {port}: {data.get('service', 'unknown')} - HEALTHY")
                    else:
                        unhealthy += 1
                        print(f"  ⚠️ Port {port}: {data.get('service', 'unknown')} - UNHEALTHY")
                else:
                    unhealthy += 1
                    print(f"  ❌ Port {port}: HTTP {response.status_code}")
            except Exception as e:
                unhealthy += 1
                print(f"  ❌ Port {port}: Not reachable")
    
    print(f"\n📊 Health Check Summary:")
    print(f"  Healthy: {healthy}/17")
    print(f"  Unhealthy: {unhealthy}/17")
    
    return healthy > 0

async def main():
    """Run all integration tests"""
    print("=" * 60)
    print("MCP-MESH INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Service Discovery
    discovery_ok = await test_mesh_discovery()
    
    # Test 2: Health Checks
    health_ok = await test_mcp_health()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    if discovery_ok and health_ok:
        print("✅ MCP-Mesh integration is WORKING!")
        print("   - MCPs are visible in service mesh")
        print("   - Health checks are responding")
        return 0
    elif discovery_ok:
        print("⚠️ Partial integration:")
        print("   - MCPs visible in mesh but health checks failing")
        return 1
    elif health_ok:
        print("⚠️ Partial integration:")
        print("   - MCP health checks work but not in mesh")
        return 1
    else:
        print("❌ MCP-Mesh integration is NOT WORKING!")
        print("   - MCPs not visible in service mesh")
        print("   - Health checks not responding")
        return 2

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
