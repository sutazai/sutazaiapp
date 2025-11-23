"""
Comprehensive System Validation
Tests all services, ports, connections, and endpoints as documented in PortRegistry.md
Compliant with Professional Project Standards - Rule 1 & Rule 2
"""

import asyncio
import httpx
import socket
import time
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class SystemValidator:
    """Validates the entire SutazaiApp system architecture"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "core_infrastructure": {},
            "ai_services": {},
            "agents": {},
            "mcp_services": {},
            "monitoring": {},
            "summary": {}
        }
        
    def check_port_open(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if a TCP port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            print(f"Error checking port {port}: {e}")
            return False
    
    async def check_http_endpoint(
        self, 
        url: str, 
        timeout: float = 5.0,
        expected_status: List[int] = [200]
    ) -> Tuple[bool, Optional[str]]:
        """Check if an HTTP endpoint is reachable"""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                is_ok = response.status_code in expected_status
                return (is_ok, f"HTTP {response.status_code}")
        except httpx.ConnectError:
            return (False, "Connection refused")
        except httpx.TimeoutException:
            return (False, "Timeout")
        except Exception as e:
            return (False, f"Error: {str(e)}")
    
    def test_core_infrastructure(self):
        """Test Core Infrastructure (10000-10099)"""
        print("\n" + "="*80)
        print("TESTING CORE INFRASTRUCTURE (Ports 10000-10099)")
        print("="*80)
        
        services = {
            "PostgreSQL": ("localhost", 10000),
            "Redis": ("localhost", 10001),
            "Neo4j HTTP": ("localhost", 10002),
            "Neo4j Bolt": ("localhost", 10003),
            "Kong Proxy": ("localhost", 10008),
            "Kong Admin": ("localhost", 10009),
            "RabbitMQ AMQP": ("localhost", 10004),
            "RabbitMQ Management": ("localhost", 10005),
            "Consul": ("localhost", 10006),
            "Backend API": ("localhost", 10200),
            "Frontend": ("localhost", 11000),
        }
        
        for service_name, (host, port) in services.items():
            is_open = self.check_port_open(host, port)
            status = "✓ ONLINE" if is_open else "✗ OFFLINE"
            print(f"{service_name:25} Port {port:5} {status}")
            
            self.results["core_infrastructure"][service_name] = {
                "port": port,
                "status": "online" if is_open else "offline",
                "host": host
            }
    
    async def test_ai_services(self):
        """Test AI & Vector Services (10100-10199)"""
        print("\n" + "="*80)
        print("TESTING AI & VECTOR SERVICES (Ports 10100-10199)")
        print("="*80)
        
        services = {
            "ChromaDB": {
                "port": 10100,
                "url": "http://localhost:10100/api/v1/heartbeat"
            },
            "Qdrant HTTP": {
                "port": 10101,
                "url": "http://localhost:10101"
            },
            "Qdrant gRPC": {
                "port": 10102,
                "url": None  # gRPC, not HTTP
            },
            "FAISS": {
                "port": 10103,
                "url": "http://localhost:10103/health"
            },
            "Ollama": {
                "port": 11434,  # Note: Ollama uses 11434
                "url": "http://localhost:11434/api/tags"
            }
        }
        
        for service_name, config in services.items():
            port = config["port"]
            url = config["url"]
            
            # Check TCP port
            is_open = self.check_port_open("localhost", port)
            
            # Check HTTP endpoint if available
            http_status = None
            if url:
                is_http_ok, http_msg = await self.check_http_endpoint(url)
                http_status = http_msg
            
            status = "✓ ONLINE" if is_open else "✗ OFFLINE"
            http_info = f" | {http_status}" if http_status else ""
            print(f"{service_name:25} Port {port:5} {status}{http_info}")
            
            self.results["ai_services"][service_name] = {
                "port": port,
                "tcp_status": "online" if is_open else "offline",
                "http_status": http_status
            }
    
    def test_agents(self):
        """Test Agent Services (11000-11999)"""
        print("\n" + "="*80)
        print("TESTING AGENT SERVICES (Ports 11000-11999)")
        print("="*80)
        
        # Based on docker-compose files discovered
        agents = {
            "Letta": 11101,
            "LangChain": 11201,
            "Aider": 11301,
            "GPT-Engineer": 11302,
            "CrewAI": 11401,
            "Documind": 11502,
            "FinRobot": 11601,
            "ShellGPT": 11701,
        }
        
        for agent_name, port in agents.items():
            is_open = self.check_port_open("localhost", port)
            status = "✓ ONLINE" if is_open else "✗ OFFLINE"
            print(f"{agent_name:25} Port {port:5} {status}")
            
            self.results["agents"][agent_name] = {
                "port": port,
                "status": "online" if is_open else "offline"
            }
    
    async def test_mcp_services(self):
        """Test MCP Bridge Services (11100-11199)"""
        print("\n" + "="*80)
        print("TESTING MCP BRIDGE SERVICES (Ports 11100-11199)")
        print("="*80)
        
        # MCP Bridge should be on port 11100
        port = 11100
        is_open = self.check_port_open("localhost", port)
        
        # Try to check HTTP endpoint
        url = "http://localhost:11100/health"
        is_http_ok, http_msg = await self.check_http_endpoint(url)
        
        status = "✓ ONLINE" if is_open else "✗ OFFLINE"
        print(f"{'MCP Bridge':25} Port {port:5} {status} | {http_msg}")
        
        self.results["mcp_services"]["MCP Bridge"] = {
            "port": port,
            "tcp_status": "online" if is_open else "offline",
            "http_status": http_msg
        }
    
    def test_monitoring(self):
        """Test Monitoring Stack (10200-10299)"""
        print("\n" + "="*80)
        print("TESTING MONITORING STACK (Ports 10200-10299)")
        print("="*80)
        
        services = {
            "Prometheus": 10200,
            "Grafana": 10201,
            "Loki": 10202,
            "AlertManager": 10203,
            "Blackbox Exporter": 10204,
            "Node Exporter": 10205,
            "Jaeger": 10211,
        }
        
        for service_name, port in services.items():
            is_open = self.check_port_open("localhost", port)
            status = "✓ ONLINE" if is_open else "✗ OFFLINE"
            print(f"{service_name:25} Port {port:5} {status}")
            
            self.results["monitoring"][service_name] = {
                "port": port,
                "status": "online" if is_open else "offline"
            }
    
    async def test_backend_endpoints(self):
        """Test Backend API Endpoints"""
        print("\n" + "="*80)
        print("TESTING BACKEND API ENDPOINTS")
        print("="*80)
        
        base_url = "http://localhost:10200"
        
        endpoints = {
            "Root": "/",
            "Health Check": "/health",
            "Detailed Health": "/health/detailed",
            "API Docs": "/docs",
            "OpenAPI Schema": "/api/v1/openapi.json",
        }
        
        for endpoint_name, path in endpoints.items():
            url = f"{base_url}{path}"
            is_ok, msg = await self.check_http_endpoint(url, expected_status=[200, 307])
            status = "✓" if is_ok else "✗"
            print(f"{status} {endpoint_name:25} {path:30} {msg}")
    
    async def test_websocket(self):
        """Test WebSocket connection"""
        print("\n" + "="*80)
        print("TESTING WEBSOCKET CONNECTION")
        print("="*80)
        
        # WebSocket endpoint: ws://localhost:10200/ws
        # For now, just check if the port is open and HTTP endpoint responds
        url = "http://localhost:10200/"
        is_ok, msg = await self.check_http_endpoint(url)
        print(f"WebSocket Endpoint Check: {msg}")
    
    def generate_summary(self):
        """Generate summary of test results"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        summary = {}
        
        for category, services in self.results.items():
            if category in ["timestamp", "summary"]:
                continue
                
            total = len(services)
            online = sum(
                1 for s in services.values() 
                if isinstance(s, dict) and (
                    s.get("status") == "online" or 
                    s.get("tcp_status") == "online"
                )
            )
            
            summary[category] = {
                "total": total,
                "online": online,
                "offline": total - online,
                "percentage": (online / total * 100) if total > 0 else 0
            }
            
            status_icon = "✓" if online == total else "⚠" if online > 0 else "✗"
            print(f"{status_icon} {category:25} {online:2}/{total:2} online ({summary[category]['percentage']:.0f}%)")
        
        self.results["summary"] = summary
        
        # Overall system health
        total_services = sum(s["total"] for s in summary.values())
        total_online = sum(s["online"] for s in summary.values())
        overall_health = (total_online / total_services * 100) if total_services > 0 else 0
        
        print(f"\n{'Overall System Health':25} {total_online:2}/{total_services:2} ({overall_health:.1f}%)")
        
        return overall_health
    
    def save_results(self, filename: str = "system_validation_results.json"):
        """Save validation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {filename}")
    
    async def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "="*80)
        print(" SUTAZAIAPP COMPREHENSIVE SYSTEM VALIDATION")
        print(" Validating against PortRegistry.md specifications")
        print("="*80)
        
        # Run all tests
        self.test_core_infrastructure()
        await self.test_ai_services()
        self.test_agents()
        await self.test_mcp_services()
        self.test_monitoring()
        await self.test_backend_endpoints()
        await self.test_websocket()
        
        # Generate summary
        overall_health = self.generate_summary()
        
        # Save results
        self.save_results()
        
        return overall_health


async def main():
    """Main execution function"""
    validator = SystemValidator()
    health = await validator.run_all_tests()
    
    # Exit with appropriate code
    if health >= 80:
        print("\n✓ System validation PASSED (>= 80% healthy)")
        exit(0)
    elif health >= 50:
        print(f"\n⚠ System validation WARNING ({health:.1f}% healthy)")
        exit(1)
    else:
        print(f"\n✗ System validation FAILED ({health:.1f}% healthy)")
        exit(2)


if __name__ == "__main__":
    asyncio.run(main())
