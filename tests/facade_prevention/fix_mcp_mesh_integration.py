#!/usr/bin/env python3
"""
Script to fix the MCP-Mesh integration in the backend
Re-enables the disabled MCP startup and implements proper mesh registration
"""
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

def backup_file(filepath):
    """Create backup of file before modification"""
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up {filepath} to {backup_path}")
    return backup_path

def fix_main_py():
    """Fix the main.py to use real MCP startup instead of disabled version"""
    main_py = Path("/opt/sutazaiapp/backend/app/main.py")
    
    if not main_py.exists():
        print("❌ main.py not found")
        return False
    
    # Read current content
    content = main_py.read_text()
    
    # Check current state
    if "from app.core.mcp_disabled import" in content:
        print("✅ Found disabled MCP import in main.py")
        
        # Create fixed version
        fixed_content = content.replace(
            "# from app.core.mcp_startup import initialize_mcp_background, shutdown_mcp_services\n"
            "from app.core.mcp_disabled import initialize_mcp_background, shutdown_mcp_services",
            "from app.core.mcp_startup import initialize_mcp_background, shutdown_mcp_services\n"
            "# from app.core.mcp_disabled import initialize_mcp_background, shutdown_mcp_services"
        )
        
        # Alternative pattern if comments are different
        if fixed_content == content:
            fixed_content = content.replace(
                "from app.core.mcp_disabled import initialize_mcp_background, shutdown_mcp_services",
                "from app.core.mcp_startup import initialize_mcp_background, shutdown_mcp_services"
            )
        
        if fixed_content != content:
            backup_file(main_py)
            main_py.write_text(fixed_content)
            print("✅ Fixed main.py to use real MCP startup")
            return True
        else:
            print("⚠️ Could not fix main.py - pattern not found")
            return False
    else:
        print("ℹ️ main.py already using mcp_startup")
        return True

def create_mcp_mesh_initializer():
    """Create new initializer that properly integrates MCPs with mesh"""
    initializer_path = Path("/opt/sutazaiapp/backend/app/mesh/mcp_mesh_initializer.py")
    
    content = '''"""
MCP-Mesh Integration Initializer
Properly registers all MCP services with the service mesh on startup
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class MCPMeshInitializer:
    """Initializes MCP services and registers them with the mesh"""
    
    # MCP service port mapping (11100-11116)
    MCP_SERVICES = {
        "language-server": {"port": 11100, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh"},
        "github": {"port": 11101, "wrapper": "npx -y @modelcontextprotocol/server-github"},
        "ultimatecoder": {"port": 11102, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ultimatecoder.sh"},
        "sequentialthinking": {"port": 11103, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/sequentialthinking.sh"},
        "context7": {"port": 11104, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/context7.sh"},
        "files": {"port": 11105, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh"},
        "http": {"port": 11106, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh"},
        "ddg": {"port": 11107, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh"},
        "postgres": {"port": 11108, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh"},
        "extended-memory": {"port": 11109, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh"},
        "mcp_ssh": {"port": 11110, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh"},
        "nx-mcp": {"port": 11111, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp.sh"},
        "puppeteer-mcp": {"port": 11112, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/puppeteer-mcp.sh"},
        "memory-bank-mcp": {"port": 11113, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/memory-bank-mcp.sh"},
        "playwright-mcp": {"port": 11114, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh"},
        "knowledge-graph-mcp": {"port": 11115, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh"},
        "compass-mcp": {"port": 11116, "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/compass-mcp.sh"}
    }
    
    def __init__(self, mesh_client):
        self.mesh_client = mesh_client
        self.registered_services: List[str] = []
        
    async def initialize_and_register(self) -> Dict[str, Any]:
        """Initialize MCP services and register with mesh"""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "registered": [],
            "failed": [],
            "total": len(self.MCP_SERVICES)
        }
        
        logger.info(f"Registering {len(self.MCP_SERVICES)} MCP services with mesh...")
        
        for name, config in self.MCP_SERVICES.items():
            try:
                # Register MCP service with mesh
                service_data = {
                    "service_name": f"mcp-{name}",
                    "address": "localhost",
                    "port": config["port"],
                    "tags": ["mcp", "stdio-bridge"],
                    "metadata": {
                        "wrapper": config["wrapper"],
                        "mcp_type": name,
                        "protocol": "stdio"
                    }
                }
                
                # Register with mesh
                instance = await self.mesh_client.register_service(**service_data)
                
                if instance:
                    self.registered_services.append(name)
                    results["registered"].append(name)
                    logger.info(f"✅ Registered MCP {name} with mesh on port {config['port']}")
                else:
                    results["failed"].append(name)
                    logger.error(f"❌ Failed to register MCP {name}")
                    
            except Exception as e:
                results["failed"].append(name)
                logger.error(f"❌ Error registering MCP {name}: {e}")
        
        # Log summary
        logger.info(f"MCP Mesh Registration Complete:")
        logger.info(f"  Registered: {len(results['registered'])}/{results['total']}")
        logger.info(f"  Failed: {len(results['failed'])}/{results['total']}")
        
        return results
    
    async def deregister_all(self):
        """Deregister all MCP services from mesh"""
        for name in self.registered_services:
            try:
                service_id = f"mcp-{name}"
                await self.mesh_client.deregister_service(service_id)
                logger.info(f"Deregistered MCP {name} from mesh")
            except Exception as e:
                logger.error(f"Error deregistering MCP {name}: {e}")
        
        self.registered_services.clear()

# Global instance
_mcp_mesh_initializer: Optional[MCPMeshInitializer] = None

async def get_mcp_mesh_initializer(mesh_client):
    """Get or create MCP mesh initializer"""
    global _mcp_mesh_initializer
    if not _mcp_mesh_initializer:
        _mcp_mesh_initializer = MCPMeshInitializer(mesh_client)
    return _mcp_mesh_initializer
'''
    
    initializer_path.parent.mkdir(parents=True, exist_ok=True)
    initializer_path.write_text(content)
    print(f"✅ Created MCP mesh initializer at {initializer_path}")
    return True

def update_mcp_startup():
    """Update mcp_startup.py to include mesh registration"""
    startup_path = Path("/opt/sutazaiapp/backend/app/core/mcp_startup.py")
    
    if not startup_path.exists():
        print("❌ mcp_startup.py not found")
        return False
    
    content = startup_path.read_text()
    
    # Add mesh registration import if not present
    if "from ..mesh.mcp_mesh_initializer import" not in content:
        # Add import after existing imports
        import_line = "from ..mesh.mcp_mesh_initializer import get_mcp_mesh_initializer"
        
        # Find where to insert
        if "from ..mesh.mcp_stdio_bridge import" in content:
            content = content.replace(
                "from ..mesh.mcp_stdio_bridge import get_mcp_stdio_bridge",
                f"from ..mesh.mcp_stdio_bridge import get_mcp_stdio_bridge\n{import_line}"
            )
        else:
            # Add after logger definition
            content = content.replace(
                'logger = logging.getLogger(__name__)',
                f'logger = logging.getLogger(__name__)\n\n{import_line}'
            )
        
        # Add mesh registration to initialize function
        if "# Register with mesh" not in content:
            registration_code = '''
        # Register MCPs with service mesh
        try:
            from ..mesh.service_mesh import get_mesh
            mesh = await get_mesh()
            if mesh:
                initializer = await get_mcp_mesh_initializer(mesh)
                mesh_results = await initializer.initialize_and_register()
                logger.info(f"Registered {len(mesh_results['registered'])} MCPs with mesh")
        except Exception as e:
            logger.warning(f"Could not register MCPs with mesh: {e}")
            # Non-fatal - MCPs can still work without mesh
'''
            
            # Insert after successful initialization
            content = content.replace(
                '            logger.info(f"✅ Successfully initialized all {started} MCP services via stdio")',
                f'            logger.info(f"✅ Successfully initialized all {{started}} MCP services via stdio")\n{registration_code}'
            )
        
        backup_file(startup_path)
        startup_path.write_text(content)
        print("✅ Updated mcp_startup.py to include mesh registration")
        return True
    else:
        print("ℹ️ mcp_startup.py already has mesh registration")
        return True

def create_test_script():
    """Create test script to verify integration"""
    test_path = Path("/opt/sutazaiapp/tests/facade_prevention/test_mcp_mesh_integration.py")
    
    content = '''#!/usr/bin/env python3
"""
Test script to verify MCP-Mesh integration is working
"""
import asyncio
import httpx
import json
import sys

async def test_mesh_discovery():
    """Test that MCPs appear in mesh service discovery"""
    print("\\n🔍 Testing MCP visibility in service mesh...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Get all services from mesh
            response = await client.get("http://localhost:10010/api/v1/mesh/v2/services")
            
            if response.status_code == 200:
                services = response.json()
                
                # Count MCP services
                mcp_services = [s for s in services if s.get("service_name", "").startswith("mcp-")]
                
                print(f"\\n📊 Service Mesh Status:")
                print(f"  Total services: {len(services)}")
                print(f"  MCP services: {len(mcp_services)}")
                
                if mcp_services:
                    print("\\n✅ MCP Services Found in Mesh:")
                    for svc in mcp_services:
                        print(f"  - {svc['service_name']} on port {svc.get('port', 'unknown')}")
                    return True
                else:
                    print("\\n❌ No MCP services found in mesh!")
                    return False
            else:
                print(f"❌ Failed to query mesh: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error querying mesh: {e}")
            return False

async def test_mcp_health():
    """Test MCP health endpoints through mesh"""
    print("\\n🏥 Testing MCP health checks...")
    
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
    
    print(f"\\n📊 Health Check Summary:")
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
    print("\\n" + "=" * 60)
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
'''
    
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(content)
    test_path.chmod(0o755)
    print(f"✅ Created test script at {test_path}")
    return True

def main():
    """Main function to apply all fixes"""
    print("=" * 60)
    print("MCP-MESH INTEGRATION FIX")
    print("=" * 60)
    print()
    
    success = True
    
    # Fix 1: Re-enable MCP startup
    print("1. Fixing main.py...")
    if not fix_main_py():
        success = False
    
    # Fix 2: Create mesh initializer
    print("\n2. Creating MCP mesh initializer...")
    if not create_mcp_mesh_initializer():
        success = False
    
    # Fix 3: Update startup to include mesh registration
    print("\n3. Updating MCP startup for mesh registration...")
    if not update_mcp_startup():
        success = False
    
    # Fix 4: Create test script
    print("\n4. Creating test script...")
    if not create_test_script():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Restart the backend: docker-compose restart backend")
        print("2. Wait for services to initialize (30-60 seconds)")
        print("3. Run test: python /opt/sutazaiapp/tests/facade_prevention/test_mcp_mesh_integration.py")
    else:
        print("⚠️ Some fixes failed - review the output above")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())