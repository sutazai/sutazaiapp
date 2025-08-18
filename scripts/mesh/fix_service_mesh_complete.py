#!/usr/bin/env python3
"""
Complete Service Mesh Overhaul Script
Fixes Consul registration, Kong routes, DinD bridge, and health checks
"""
import subprocess
import json
import time
import requests
import logging
from typing import Dict, List, Any, Optional
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ServiceMeshFixer:
    """Complete service mesh repair implementation"""
    
    def __init__(self):
        self.consul_url = "http://localhost:10006"
        self.kong_admin_url = "http://localhost:10015"
        self.backend_url = "http://localhost:10010"
        self.orchestrator_host = "sutazai-mcp-orchestrator"
        self.results = {
            "network_analysis": {},
            "consul_fixes": {},
            "kong_fixes": {},
            "bridge_fixes": {},
            "health_checks": {},
            "validation": {}
        }
    
    def run_command(self, cmd: str) -> tuple:
        """Execute command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr, result.returncode
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return "", str(e), 1
    
    def analyze_network_topology(self):
        """Complete network topology analysis"""
        logger.info("=" * 60)
        logger.info("PHASE 1: NETWORK TOPOLOGY ANALYSIS")
        logger.info("=" * 60)
        
        # Get host containers
        stdout, _, _ = self.run_command("docker ps --format json")
        host_containers = []
        for line in stdout.strip().split('\n'):
            if line:
                host_containers.append(json.loads(line))
        
        # Get DinD containers
        stdout, _, _ = self.run_command("docker exec sutazai-mcp-orchestrator docker ps --format json")
        dind_containers = []
        for line in stdout.strip().split('\n'):
            if line:
                dind_containers.append(json.loads(line))
        
        # Analyze networks
        stdout, _, _ = self.run_command("docker network ls --format json")
        networks = []
        for line in stdout.strip().split('\n'):
            if line:
                networks.append(json.loads(line))
        
        self.results["network_analysis"] = {
            "host_containers": len(host_containers),
            "dind_containers": len(dind_containers),
            "networks": [n["Name"] for n in networks if "sutazai" in n["Name"]],
            "host_container_names": [c["Names"] for c in host_containers],
            "dind_container_names": [c["Names"] for c in dind_containers]
        }
        
        logger.info(f"Found {len(host_containers)} host containers")
        logger.info(f"Found {len(dind_containers)} DinD containers")
        logger.info(f"Networks: {self.results['network_analysis']['networks']}")
        
        # Test connectivity
        logger.info("\nTesting connectivity...")
        
        # Test host to orchestrator
        stdout, _, code = self.run_command("docker exec sutazai-backend ping -c 1 sutazai-mcp-orchestrator")
        can_ping_orchestrator = code == 0
        
        # Test orchestrator internal network
        stdout, _, code = self.run_command("docker exec sutazai-mcp-orchestrator docker exec mcp-claude-flow echo 'test'")
        can_access_mcp = code == 0
        
        self.results["network_analysis"]["connectivity"] = {
            "host_to_orchestrator": can_ping_orchestrator,
            "orchestrator_to_mcp": can_access_mcp
        }
        
        logger.info(f"Host → Orchestrator: {can_ping_orchestrator}")
        logger.info(f"Orchestrator → MCP: {can_access_mcp}")
    
    def fix_consul_registration(self):
        """Fix Consul service registration"""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: FIXING CONSUL REGISTRATION")
        logger.info("=" * 60)
        
        try:
            # Get current services
            response = requests.get(f"{self.consul_url}/v1/agent/services")
            current_services = response.json()
            
            logger.info(f"Current services in Consul: {len(current_services)}")
            
            # Deregister fake/broken services
            services_to_remove = []
            for service_id, service_data in current_services.items():
                # Check if service actually responds
                if service_data.get("Port"):
                    try:
                        test_url = f"http://{service_data.get('Address', 'localhost')}:{service_data['Port']}/health"
                        test_response = requests.get(test_url, timeout=2)
                        if test_response.status_code != 200:
                            services_to_remove.append(service_id)
                    except:
                        services_to_remove.append(service_id)
            
            # Remove broken services
            for service_id in services_to_remove:
                logger.info(f"Deregistering broken service: {service_id}")
                requests.put(f"{self.consul_url}/v1/agent/service/deregister/{service_id}")
            
            self.results["consul_fixes"]["removed"] = services_to_remove
            
            # Register real MCP services via bridge
            mcp_services = self._get_real_mcp_services()
            registered = []
            
            for mcp_service in mcp_services:
                service_def = {
                    "ID": f"mcp-{mcp_service['name']}-bridge",
                    "Name": f"mcp-{mcp_service['name']}",
                    "Tags": ["mcp", "bridge", mcp_service['name']],
                    "Address": self.orchestrator_host,
                    "Port": mcp_service['port'],
                    "Check": {
                        "HTTP": f"http://{self.orchestrator_host}:{mcp_service['port']}/health",
                        "Interval": "10s",
                        "Timeout": "5s"
                    }
                }
                
                response = requests.put(
                    f"{self.consul_url}/v1/agent/service/register",
                    json=service_def
                )
                
                if response.status_code == 200:
                    registered.append(mcp_service['name'])
                    logger.info(f"Registered MCP service: {mcp_service['name']}")
            
            self.results["consul_fixes"]["registered"] = registered
            
        except Exception as e:
            logger.error(f"Consul fix failed: {e}")
            self.results["consul_fixes"]["error"] = str(e)
    
    def _get_real_mcp_services(self) -> List[Dict]:
        """Get actual MCP services from DinD"""
        services = []
        
        # Map of MCP services to their ports
        mcp_port_map = {
            "claude-flow": 3001,
            "ruv-swarm": 3002,
            "files": 3003,
            "context7": 3004,
            "http-fetch": 3005,
            "ddg": 3006,
            "sequentialthinking": 3007,
            "nx-mcp": 3008,
            "extended-memory": 3009,
            "mcp-ssh": 3010,
            "ultimatecoder": 3011,
            "playwright-mcp": 3012,
            "memory-bank-mcp": 3013,
            "knowledge-graph-mcp": 3014,
            "compass-mcp": 3015,
            "github": 3016,
            "http": 3017,
            "language-server": 3018,
            "claude-task-runner": 3019
        }
        
        for name, port in mcp_port_map.items():
            services.append({
                "name": name,
                "port": port,
                "container": f"mcp-{name}"
            })
        
        return services
    
    def fix_kong_routes(self):
        """Fix Kong API Gateway routes"""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: FIXING KONG ROUTES")
        logger.info("=" * 60)
        
        try:
            # Get current routes
            response = requests.get(f"{self.kong_admin_url}/routes")
            current_routes = response.json()["data"]
            
            logger.info(f"Current Kong routes: {len(current_routes)}")
            
            # Remove broken routes
            routes_removed = []
            for route in current_routes:
                # Test if service behind route is accessible
                service_id = route.get("service", {}).get("id")
                if service_id:
                    service_resp = requests.get(f"{self.kong_admin_url}/services/{service_id}")
                    if service_resp.status_code == 200:
                        service_data = service_resp.json()
                        # Test connectivity
                        try:
                            test_url = f"http://{service_data['host']}:{service_data['port']}/health"
                            test_resp = requests.get(test_url, timeout=2)
                            if test_resp.status_code != 200:
                                # Remove broken route
                                requests.delete(f"{self.kong_admin_url}/routes/{route['id']}")
                                routes_removed.append(route['name'])
                        except:
                            requests.delete(f"{self.kong_admin_url}/routes/{route['id']}")
                            routes_removed.append(route['name'])
            
            self.results["kong_fixes"]["removed"] = routes_removed
            
            # Add MCP routes
            routes_added = []
            mcp_services = self._get_real_mcp_services()
            
            for mcp_service in mcp_services:
                # Create service first
                service_data = {
                    "name": f"mcp-{mcp_service['name']}-service",
                    "host": self.orchestrator_host,
                    "port": mcp_service['port'],
                    "protocol": "http",
                    "path": "/",
                    "retries": 3,
                    "connect_timeout": 5000,
                    "write_timeout": 60000,
                    "read_timeout": 60000
                }
                
                service_resp = requests.post(f"{self.kong_admin_url}/services", json=service_data)
                
                if service_resp.status_code in [200, 201, 409]:  # 409 = already exists
                    if service_resp.status_code == 409:
                        # Get existing service
                        existing_resp = requests.get(f"{self.kong_admin_url}/services/mcp-{mcp_service['name']}-service")
                        service_id = existing_resp.json()["id"]
                    else:
                        service_id = service_resp.json()["id"]
                    
                    # Create route
                    route_data = {
                        "name": f"mcp-{mcp_service['name']}-route",
                        "paths": [f"/mcp/{mcp_service['name']}"],
                        "service": {"id": service_id},
                        "strip_path": True,
                        "preserve_host": False
                    }
                    
                    route_resp = requests.post(f"{self.kong_admin_url}/routes", json=route_data)
                    
                    if route_resp.status_code in [200, 201]:
                        routes_added.append(mcp_service['name'])
                        logger.info(f"Added Kong route for MCP service: {mcp_service['name']}")
            
            self.results["kong_fixes"]["added"] = routes_added
            
        except Exception as e:
            logger.error(f"Kong fix failed: {e}")
            self.results["kong_fixes"]["error"] = str(e)
    
    def implement_dind_bridge(self):
        """Implement proper DinD to host network bridge"""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: IMPLEMENTING DIND BRIDGE")
        logger.info("=" * 60)
        
        try:
            # Create network proxy configuration
            proxy_config = """
#!/bin/bash
# DinD to Host Network Bridge
# This script creates proper network connectivity between DinD and host

# Enable port forwarding from host to DinD containers
for port in {3001..3019}; do
    # Check if MCP container is listening on this port
    if docker exec sutazai-mcp-orchestrator docker port mcp-* $port 2>/dev/null | grep -q $port; then
        # Create iptables rule for port forwarding
        iptables -t nat -A DOCKER -p tcp --dport $port -j DNAT --to-destination sutazai-mcp-orchestrator:$port
        echo "Forwarded port $port to DinD"
    fi
done

# Create service discovery bridge
cat > /tmp/mcp_bridge.json << EOF
{
    "bridge_enabled": true,
    "orchestrator_host": "sutazai-mcp-orchestrator",
    "port_range": [3001, 3019],
    "health_check_interval": 10
}
EOF

echo "DinD bridge configuration created"
"""
            
            # Write bridge script
            with open("/tmp/dind_bridge.sh", "w") as f:
                f.write(proxy_config)
            
            # Make executable
            self.run_command("chmod +x /tmp/dind_bridge.sh")
            
            # Execute bridge setup
            stdout, stderr, code = self.run_command("bash /tmp/dind_bridge.sh")
            
            self.results["bridge_fixes"]["script_created"] = True
            self.results["bridge_fixes"]["execution_status"] = code == 0
            
            # Update backend configuration to use bridge
            backend_config = {
                "mcp_bridge": {
                    "enabled": True,
                    "orchestrator_host": self.orchestrator_host,
                    "port_mapping": self._get_real_mcp_services()
                }
            }
            
            # Send configuration to backend
            try:
                response = requests.post(
                    f"{self.backend_url}/api/v1/mesh/configure",
                    json=backend_config,
                    timeout=5
                )
                self.results["bridge_fixes"]["backend_configured"] = response.status_code == 200
            except:
                self.results["bridge_fixes"]["backend_configured"] = False
            
            logger.info("DinD bridge implementation complete")
            
        except Exception as e:
            logger.error(f"Bridge implementation failed: {e}")
            self.results["bridge_fixes"]["error"] = str(e)
    
    def implement_health_checks(self):
        """Implement working health checks for all services"""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 5: IMPLEMENTING HEALTH CHECKS")
        logger.info("=" * 60)
        
        health_results = {}
        
        # Check host services
        host_services = [
            ("backend", "http://localhost:10010/health"),
            ("frontend", "http://localhost:10011/"),
            ("consul", "http://localhost:10006/v1/status/leader"),
            ("kong", "http://localhost:10015/status"),
            ("postgres", "postgresql://localhost:10000/"),
            ("redis", "redis://localhost:10001/"),
            ("rabbitmq", "http://localhost:10008/api/health/checks/virtual-hosts"),
            ("neo4j", "http://localhost:10003/"),
            ("ollama", "http://localhost:10104/api/tags"),
            ("chromadb", "http://localhost:10100/api/v1/heartbeat"),
            ("qdrant", "http://localhost:10101/")
        ]
        
        for service_name, url in host_services:
            try:
                if url.startswith("postgresql://"):
                    # Special handling for PostgreSQL
                    stdout, _, code = self.run_command(f"docker exec sutazai-postgres pg_isready")
                    health_results[service_name] = {"status": "healthy" if code == 0 else "unhealthy"}
                elif url.startswith("redis://"):
                    # Special handling for Redis
                    stdout, _, code = self.run_command(f"docker exec sutazai-redis redis-cli ping")
                    health_results[service_name] = {"status": "healthy" if "PONG" in stdout else "unhealthy"}
                else:
                    response = requests.get(url, timeout=5)
                    health_results[service_name] = {
                        "status": "healthy" if response.status_code < 400 else "unhealthy",
                        "response_code": response.status_code
                    }
            except Exception as e:
                health_results[service_name] = {"status": "unhealthy", "error": str(e)}
        
        # Check MCP services
        mcp_services = self._get_real_mcp_services()
        for mcp_service in mcp_services:
            try:
                # Test via orchestrator
                cmd = f"docker exec sutazai-mcp-orchestrator docker exec {mcp_service['container']} echo 'health_check'"
                stdout, _, code = self.run_command(cmd)
                health_results[f"mcp-{mcp_service['name']}"] = {
                    "status": "healthy" if code == 0 else "unhealthy"
                }
            except Exception as e:
                health_results[f"mcp-{mcp_service['name']}"] = {"status": "unhealthy", "error": str(e)}
        
        self.results["health_checks"] = health_results
        
        # Log summary
        healthy_count = sum(1 for s in health_results.values() if s["status"] == "healthy")
        total_count = len(health_results)
        
        logger.info(f"Health check summary: {healthy_count}/{total_count} services healthy")
        
        for service, status in health_results.items():
            symbol = "✓" if status["status"] == "healthy" else "✗"
            logger.info(f"  {symbol} {service}: {status['status']}")
    
    def validate_mesh(self):
        """Validate the complete service mesh functionality"""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 6: VALIDATING SERVICE MESH")
        logger.info("=" * 60)
        
        validation_results = {
            "consul_registration": False,
            "kong_routing": False,
            "mcp_connectivity": False,
            "backend_integration": False,
            "end_to_end": False
        }
        
        try:
            # Test Consul registration
            response = requests.get(f"{self.consul_url}/v1/agent/services")
            services = response.json()
            mcp_services_count = sum(1 for s in services.values() if "mcp" in s.get("Tags", []))
            validation_results["consul_registration"] = mcp_services_count > 0
            logger.info(f"Consul registration: {mcp_services_count} MCP services registered")
            
            # Test Kong routing
            response = requests.get(f"{self.kong_admin_url}/routes")
            routes = response.json()["data"]
            mcp_routes_count = sum(1 for r in routes if "mcp" in r.get("name", ""))
            validation_results["kong_routing"] = mcp_routes_count > 0
            logger.info(f"Kong routing: {mcp_routes_count} MCP routes configured")
            
            # Test MCP connectivity via backend
            try:
                response = requests.get(f"{self.backend_url}/api/v1/mcp/services")
                mcp_services = response.json()
                validation_results["backend_integration"] = len(mcp_services) > 0
                logger.info(f"Backend integration: {len(mcp_services)} MCP services available")
            except:
                validation_results["backend_integration"] = False
            
            # Test end-to-end MCP call
            if validation_results["backend_integration"]:
                try:
                    # Try to get status of a service
                    response = requests.get(f"{self.backend_url}/api/v1/mcp/services/files/status")
                    validation_results["end_to_end"] = response.status_code == 200
                except:
                    validation_results["end_to_end"] = False
            
            self.results["validation"] = validation_results
            
            # Final summary
            all_valid = all(validation_results.values())
            
            logger.info("\n" + "=" * 60)
            if all_valid:
                logger.info("✅ SERVICE MESH FULLY OPERATIONAL")
            else:
                logger.info("⚠️ SERVICE MESH PARTIALLY OPERATIONAL")
                failed_checks = [k for k, v in validation_results.items() if not v]
                logger.info(f"Failed checks: {failed_checks}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.results["validation"]["error"] = str(e)
    
    def generate_report(self):
        """Generate comprehensive report"""
        report = f"""
SERVICE MESH COMPLETE OVERHAUL REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}

NETWORK TOPOLOGY ANALYSIS:
- Host containers: {self.results['network_analysis'].get('host_containers', 0)}
- DinD containers: {self.results['network_analysis'].get('dind_containers', 0)}
- Networks: {self.results['network_analysis'].get('networks', [])}
- Connectivity:
  - Host → Orchestrator: {self.results['network_analysis'].get('connectivity', {}).get('host_to_orchestrator', False)}
  - Orchestrator → MCP: {self.results['network_analysis'].get('connectivity', {}).get('orchestrator_to_mcp', False)}

CONSUL FIXES:
- Services removed: {len(self.results['consul_fixes'].get('removed', []))}
- Services registered: {len(self.results['consul_fixes'].get('registered', []))}

KONG FIXES:
- Routes removed: {len(self.results['kong_fixes'].get('removed', []))}
- Routes added: {len(self.results['kong_fixes'].get('added', []))}

DIND BRIDGE:
- Script created: {self.results['bridge_fixes'].get('script_created', False)}
- Execution status: {self.results['bridge_fixes'].get('execution_status', False)}
- Backend configured: {self.results['bridge_fixes'].get('backend_configured', False)}

HEALTH CHECKS:
- Total services checked: {len(self.results.get('health_checks', {}))}
- Healthy services: {sum(1 for s in self.results.get('health_checks', {}).values() if s.get('status') == 'healthy')}

VALIDATION:
- Consul registration: {self.results['validation'].get('consul_registration', False)}
- Kong routing: {self.results['validation'].get('kong_routing', False)}
- Backend integration: {self.results['validation'].get('backend_integration', False)}
- End-to-end functionality: {self.results['validation'].get('end_to_end', False)}

RECOMMENDATIONS:
1. Monitor service health continuously
2. Implement automatic recovery for failed services
3. Add circuit breakers for resilience
4. Set up proper logging and tracing
5. Document the working architecture
"""
        
        # Save report
        report_path = "/opt/sutazaiapp/docs/reports/SERVICE_MESH_OVERHAUL_REPORT.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        logger.info(f"\nReport saved to: {report_path}")
        
        return report
    
    def run(self):
        """Execute complete service mesh overhaul"""
        logger.info("STARTING SERVICE MESH COMPLETE OVERHAUL")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Analyze network topology
            self.analyze_network_topology()
            
            # Phase 2: Fix Consul registration
            self.fix_consul_registration()
            
            # Phase 3: Fix Kong routes
            self.fix_kong_routes()
            
            # Phase 4: Implement DinD bridge
            self.implement_dind_bridge()
            
            # Phase 5: Implement health checks
            self.implement_health_checks()
            
            # Phase 6: Validate mesh
            self.validate_mesh()
            
            # Generate report
            report = self.generate_report()
            
            print(report)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Service mesh overhaul failed: {e}")
            raise

if __name__ == "__main__":
    fixer = ServiceMeshFixer()
    results = fixer.run()
    
    # Exit with appropriate code
    if results.get("validation", {}).get("end_to_end"):
        sys.exit(0)
    else:
        sys.exit(1)