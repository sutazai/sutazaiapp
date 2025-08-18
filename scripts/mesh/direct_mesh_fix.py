#!/usr/bin/env python3
"""
Direct Service Mesh Fix - Focused on Real Issues
Establishes proper connectivity between DinD MCP containers and the service mesh
"""
import subprocess
import json
import requests
import logging
import sys
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectMeshFix:
    """Direct fixes for the service mesh issues"""
    
    def __init__(self):
        self.consul_url = "http://localhost:10006"
        self.kong_url = "http://localhost:10015"
        self.backend_url = "http://localhost:10010"
        
    def get_mcp_containers_in_dind(self) -> List[Dict]:
        """Get actual MCP containers running in DinD"""
        cmd = "docker exec sutazai-mcp-orchestrator docker ps --format json"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                container = json.loads(line)
                if container['Names'].startswith('mcp-'):
                    # Extract port mapping
                    ports = container.get('Ports', '')
                    port = None
                    if '->' in ports:
                        port = int(ports.split('->')[0].split(':')[-1])
                    
                    containers.append({
                        'name': container['Names'],
                        'status': container['Status'],
                        'port': port
                    })
        
        return containers
    
    def fix_consul_services(self):
        """Register real MCP services in Consul"""
        logger.info("FIXING CONSUL SERVICES")
        logger.info("-" * 40)
        
        # Get MCP containers
        mcp_containers = self.get_mcp_containers_in_dind()
        logger.info(f"Found {len(mcp_containers)} MCP containers in DinD")
        
        # Deregister all existing MCP services
        try:
            response = requests.get(f"{self.consul_url}/v1/agent/services")
            services = response.json()
            
            for service_id in services.keys():
                if 'mcp' in service_id.lower():
                    logger.info(f"Deregistering old service: {service_id}")
                    requests.put(f"{self.consul_url}/v1/agent/service/deregister/{service_id}")
        except Exception as e:
            logger.error(f"Failed to deregister services: {e}")
        
        # Register real MCP services
        registered = 0
        for container in mcp_containers:
            if container['port']:
                service_name = container['name'].replace('mcp-', '')
                service_def = {
                    "ID": f"mcp-{service_name}-{container['port']}",
                    "Name": f"mcp-{service_name}",
                    "Tags": ["mcp", "dind", service_name],
                    "Address": "localhost",  # Services are exposed on localhost via port mapping
                    "Port": container['port'],
                    "Check": {
                        "TCP": f"localhost:{container['port']}",
                        "Interval": "10s",
                        "Timeout": "5s"
                    }
                }
                
                try:
                    response = requests.put(
                        f"{self.consul_url}/v1/agent/service/register",
                        json=service_def
                    )
                    if response.status_code == 200:
                        registered += 1
                        logger.info(f"✓ Registered: {service_name} on port {container['port']}")
                except Exception as e:
                    logger.error(f"✗ Failed to register {service_name}: {e}")
        
        logger.info(f"Registered {registered}/{len(mcp_containers)} MCP services")
        return registered
    
    def fix_kong_routes(self):
        """Create working Kong routes for MCP services"""
        logger.info("\nFIXING KONG ROUTES")
        logger.info("-" * 40)
        
        mcp_containers = self.get_mcp_containers_in_dind()
        routes_created = 0
        
        for container in mcp_containers:
            if container['port']:
                service_name = container['name'].replace('mcp-', '')
                
                # Create Kong service
                service_data = {
                    "name": f"mcp-{service_name}",
                    "host": "host.docker.internal",  # Use Docker's host alias
                    "port": container['port'],
                    "protocol": "http"
                }
                
                try:
                    # Create or update service
                    response = requests.put(
                        f"{self.kong_url}/services/mcp-{service_name}",
                        json=service_data
                    )
                    
                    if response.status_code in [200, 201]:
                        # Create route
                        route_data = {
                            "name": f"mcp-{service_name}-route",
                            "paths": [f"/mcp/{service_name}"],
                            "service": {"name": f"mcp-{service_name}"}
                        }
                        
                        route_response = requests.put(
                            f"{self.kong_url}/routes/mcp-{service_name}-route",
                            json=route_data
                        )
                        
                        if route_response.status_code in [200, 201]:
                            routes_created += 1
                            logger.info(f"✓ Created route: /mcp/{service_name} -> port {container['port']}")
                except Exception as e:
                    logger.error(f"✗ Failed to create route for {service_name}: {e}")
        
        logger.info(f"Created {routes_created}/{len(mcp_containers)} Kong routes")
        return routes_created
    
    def test_connectivity(self):
        """Test actual connectivity to MCP services"""
        logger.info("\nTESTING CONNECTIVITY")
        logger.info("-" * 40)
        
        # Test direct port access
        test_ports = [3001, 3002, 3003, 3004, 3005]
        working_ports = []
        
        for port in test_ports:
            try:
                # Test if port is open
                cmd = f"nc -zv localhost {port} 2>&1"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
                if "succeeded" in result.stderr or "open" in result.stderr:
                    working_ports.append(port)
                    logger.info(f"✓ Port {port} is accessible")
                else:
                    logger.info(f"✗ Port {port} is not accessible")
            except:
                logger.info(f"✗ Port {port} test failed")
        
        # Test via Kong proxy
        try:
            response = requests.get(f"http://localhost:10005/mcp/claude-flow/health", timeout=2)
            logger.info(f"Kong proxy test: {response.status_code}")
        except Exception as e:
            logger.info(f"Kong proxy test failed: {e}")
        
        # Test via backend API
        try:
            response = requests.get(f"{self.backend_url}/api/v1/mcp/services")
            services = response.json()
            logger.info(f"Backend API reports {len(services)} MCP services")
        except Exception as e:
            logger.info(f"Backend API test failed: {e}")
        
        return len(working_ports)
    
    def create_network_bridge(self):
        """Create proper network bridge between DinD and host"""
        logger.info("\nCREATING NETWORK BRIDGE")
        logger.info("-" * 40)
        
        # Check if containers are on same network
        cmd = "docker inspect sutazai-mcp-orchestrator --format '{{json .NetworkSettings.Networks}}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        networks = json.loads(result.stdout)
        
        logger.info(f"Orchestrator networks: {list(networks.keys())}")
        
        # Ensure backend can reach orchestrator
        cmd = "docker inspect sutazai-backend --format '{{json .NetworkSettings.Networks}}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        try:
            backend_networks = json.loads(result.stdout) if result.stdout else {}
        except:
            backend_networks = {}
        
        logger.info(f"Backend networks: {list(backend_networks.keys())}")
        
        # Find common network
        common_networks = set(networks.keys()) & set(backend_networks.keys())
        if common_networks:
            logger.info(f"✓ Common network found: {common_networks}")
        else:
            logger.info("✗ No common network - services cannot communicate")
            
            # Connect orchestrator to sutazai-network if not connected
            if "sutazai-network" not in networks:
                cmd = "docker network connect sutazai-network sutazai-mcp-orchestrator"
                subprocess.run(cmd, shell=True)
                logger.info("Connected orchestrator to sutazai-network")
        
        return bool(common_networks)
    
    def generate_summary(self):
        """Generate summary report"""
        logger.info("\n" + "=" * 60)
        logger.info("SERVICE MESH FIX SUMMARY")
        logger.info("=" * 60)
        
        # Get final status
        mcp_containers = self.get_mcp_containers_in_dind()
        
        # Check Consul
        try:
            response = requests.get(f"{self.consul_url}/v1/agent/services")
            consul_services = len([s for s in response.json().keys() if 'mcp' in s.lower()])
        except:
            consul_services = 0
        
        # Check Kong
        try:
            response = requests.get(f"{self.kong_url}/routes")
            kong_routes = len([r for r in response.json()['data'] if 'mcp' in r.get('name', '').lower()])
        except:
            kong_routes = 0
        
        logger.info(f"MCP Containers in DinD: {len(mcp_containers)}")
        logger.info(f"Consul MCP Services: {consul_services}")
        logger.info(f"Kong MCP Routes: {kong_routes}")
        
        # Test end-to-end
        try:
            response = requests.get(f"{self.backend_url}/api/v1/mcp/services")
            backend_services = len(response.json())
            logger.info(f"Backend MCP Services: {backend_services}")
            
            if backend_services > 0:
                logger.info("\n✅ SERVICE MESH IS OPERATIONAL")
            else:
                logger.info("\n⚠️ SERVICE MESH PARTIALLY WORKING")
        except:
            logger.info("\n❌ SERVICE MESH NOT WORKING")
    
    def run(self):
        """Execute all fixes"""
        logger.info("STARTING DIRECT SERVICE MESH FIX")
        logger.info("=" * 60)
        
        # 1. Create network bridge
        self.create_network_bridge()
        
        # 2. Fix Consul services
        self.fix_consul_services()
        
        # 3. Fix Kong routes
        self.fix_kong_routes()
        
        # 4. Test connectivity
        self.test_connectivity()
        
        # 5. Generate summary
        self.generate_summary()

if __name__ == "__main__":
    fixer = DirectMeshFix()
    fixer.run()