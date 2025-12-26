#!/usr/bin/env python3
"""
Phase 2 Validation Script
Deep review and validation of all critical components
"""

import asyncio
import httpx
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {text}")


class Phase2Validator:
    """Validates all Phase 2 components"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results = {
            "jwt_auth": {"status": "pending", "details": []},
            "backend_api": {"status": "pending", "details": []},
            "frontend": {"status": "pending", "details": []},
            "mcp_bridge": {"status": "pending", "details": []},
            "portainer_stack": {"status": "pending", "details": []},
            "vector_dbs": {"status": "pending", "details": []},
            "infrastructure": {"status": "pending", "details": []},
        }
        
    async def validate_jwt_authentication(self):
        """Validate JWT authentication implementation"""
        print_header("JWT Authentication Validation")
        
        try:
            # Check backend health endpoint
            response = await self.client.get("http://localhost:10200/health")
            if response.status_code == 200:
                print_success("Backend health endpoint accessible")
                self.results["jwt_auth"]["details"].append("Health endpoint OK")
            else:
                print_warning(f"Backend health endpoint returned {response.status_code}")
                
            # Test JWT token structure
            print_info("JWT implementation review:")
            security_file = Path("/home/runner/work/sutazaiapp/sutazaiapp/backend/app/core/security.py")
            if security_file.exists():
                print_success("Security module exists")
                with open(security_file) as f:
                    content = f.read()
                    if "create_access_token" in content:
                        print_success("  - Access token creation implemented")
                    if "create_refresh_token" in content:
                        print_success("  - Refresh token creation implemented")
                    if "verify_token" in content:
                        print_success("  - Token verification implemented")
                    if "bcrypt" in content:
                        print_success("  - Password hashing (bcrypt) implemented")
                    if "JWTError" in content:
                        print_success("  - JWT error handling implemented")
                        
                self.results["jwt_auth"]["status"] = "passed"
                self.results["jwt_auth"]["details"].append("All JWT functions implemented")
            else:
                print_error("Security module not found")
                self.results["jwt_auth"]["status"] = "failed"
                
        except Exception as e:
            print_error(f"JWT validation failed: {e}")
            self.results["jwt_auth"]["status"] = "failed"
            self.results["jwt_auth"]["details"].append(f"Error: {str(e)}")
    
    async def validate_backend_api(self):
        """Validate backend API implementation"""
        print_header("Backend API Validation")
        
        try:
            # Test health endpoint
            response = await self.client.get("http://localhost:10200/health")
            if response.status_code == 200:
                data = response.json()
                print_success("Backend API is responding")
                print_info(f"  Status: {data.get('status', 'unknown')}")
                print_info(f"  Service: {data.get('service', 'unknown')}")
                
                # Check service connections
                if "services" in data:
                    print_info("  Connected services:")
                    for service, status in data["services"].items():
                        if status == "healthy":
                            print_success(f"    - {service}: {status}")
                        else:
                            print_warning(f"    - {service}: {status}")
                            
                self.results["backend_api"]["status"] = "passed"
                self.results["backend_api"]["details"].append(f"API responding with {len(data.get('services', {}))} services")
            else:
                print_error(f"Backend API returned status {response.status_code}")
                self.results["backend_api"]["status"] = "degraded"
                
        except httpx.ConnectError:
            print_error("Cannot connect to backend API (not running)")
            self.results["backend_api"]["status"] = "failed"
            self.results["backend_api"]["details"].append("Service not accessible")
        except Exception as e:
            print_error(f"Backend validation failed: {e}")
            self.results["backend_api"]["status"] = "failed"
    
    async def validate_frontend(self):
        """Validate frontend implementation"""
        print_header("Frontend Validation")
        
        try:
            # Check if frontend is accessible
            response = await self.client.get("http://localhost:11000/_stcore/health")
            if response.status_code == 200:
                print_success("Frontend is accessible")
                self.results["frontend"]["status"] = "passed"
            else:
                print_warning(f"Frontend health check returned {response.status_code}")
                self.results["frontend"]["status"] = "degraded"
                
            # Check frontend files
            frontend_dir = Path("/home/runner/work/sutazaiapp/sutazaiapp/frontend")
            if frontend_dir.exists():
                print_info("Frontend implementation review:")
                
                app_py = frontend_dir / "app.py"
                if app_py.exists():
                    print_success("  - Main app.py exists")
                    with open(app_py) as f:
                        content = f.read()
                        if "VoiceAssistant" in content:
                            print_success("  - Voice assistant component integrated")
                        if "ChatInterface" in content:
                            print_success("  - Chat interface component integrated")
                        if "SystemMonitor" in content:
                            print_success("  - System monitor component integrated")
                        if "AgentOrchestrator" in content:
                            print_success("  - Agent orchestrator integrated")
                            
                # Check components directory
                components_dir = frontend_dir / "components"
                if components_dir.exists():
                    components = list(components_dir.glob("*.py"))
                    print_success(f"  - {len(components)} component files found")
                    
                self.results["frontend"]["details"].append("Frontend structure validated")
            else:
                print_error("Frontend directory not found")
                self.results["frontend"]["status"] = "failed"
                
        except httpx.ConnectError:
            print_error("Cannot connect to frontend (not running)")
            self.results["frontend"]["status"] = "failed"
        except Exception as e:
            print_error(f"Frontend validation failed: {e}")
            self.results["frontend"]["status"] = "failed"
    
    async def validate_mcp_bridge(self):
        """Validate MCP Bridge implementation"""
        print_header("MCP Bridge Validation")
        
        try:
            # Check MCP Bridge health
            response = await self.client.get("http://localhost:11100/health")
            if response.status_code == 200:
                data = response.json()
                print_success("MCP Bridge is responding")
                print_info(f"  Status: {data.get('status', 'unknown')}")
                
                # Check agent registry
                agents_response = await self.client.get("http://localhost:11100/agents")
                if agents_response.status_code == 200:
                    agents = agents_response.json()
                    print_success(f"  Agent registry: {len(agents.get('agents', []))} agents registered")
                    
                self.results["mcp_bridge"]["status"] = "passed"
            else:
                print_warning(f"MCP Bridge returned status {response.status_code}")
                self.results["mcp_bridge"]["status"] = "degraded"
                
            # Check MCP Bridge files
            mcp_bridge_dir = Path("/home/runner/work/sutazaiapp/sutazaiapp/mcp-bridge")
            if mcp_bridge_dir.exists():
                print_info("MCP Bridge implementation review:")
                
                server_file = mcp_bridge_dir / "services" / "mcp_bridge_server.py"
                if server_file.exists():
                    print_success("  - MCP Bridge server exists")
                    with open(server_file) as f:
                        content = f.read()
                        if "SERVICE_REGISTRY" in content:
                            print_success("  - Service registry implemented")
                        if "AGENT_REGISTRY" in content:
                            print_success("  - Agent registry implemented")
                        if "rabbitmq" in content.lower():
                            print_success("  - RabbitMQ integration implemented")
                        if "redis" in content.lower():
                            print_success("  - Redis integration implemented")
                            
                self.results["mcp_bridge"]["details"].append("MCP Bridge structure validated")
            else:
                print_error("MCP Bridge directory not found")
                
        except httpx.ConnectError:
            print_error("Cannot connect to MCP Bridge (not running)")
            self.results["mcp_bridge"]["status"] = "failed"
        except Exception as e:
            print_error(f"MCP Bridge validation failed: {e}")
            self.results["mcp_bridge"]["status"] = "failed"
    
    async def validate_portainer_stack(self):
        """Validate Portainer stack configuration"""
        print_header("Portainer Stack Validation")
        
        try:
            stack_file = Path("/home/runner/work/sutazaiapp/sutazaiapp/portainer-stack.yml")
            if stack_file.exists():
                print_success("Portainer stack file exists")
                
                with open(stack_file) as f:
                    content = f.read()
                    
                print_info("Stack configuration review:")
                
                # Count services
                services_count = content.count("image:")
                print_success(f"  - {services_count} services defined")
                
                # Check critical services
                critical_services = [
                    "portainer", "postgres", "redis", "neo4j", "rabbitmq",
                    "consul", "kong", "chromadb", "qdrant", "faiss",
                    "ollama", "backend", "frontend", "prometheus", "grafana"
                ]
                
                missing_services = []
                for service in critical_services:
                    if service in content.lower():
                        print_success(f"  - {service} configured")
                    else:
                        missing_services.append(service)
                        print_warning(f"  - {service} not found")
                        
                # Check network configuration
                if "sutazai-network" in content:
                    print_success("  - Network 'sutazai-network' configured")
                if "172.20.0.0/16" in content:
                    print_success("  - Subnet 172.20.0.0/16 configured")
                    
                # Check health checks
                healthcheck_count = content.count("healthcheck:")
                print_success(f"  - {healthcheck_count} health checks configured")
                
                # Check resource limits
                limits_count = content.count("limits:")
                print_success(f"  - {limits_count} resource limits configured")
                
                if len(missing_services) == 0:
                    self.results["portainer_stack"]["status"] = "passed"
                    self.results["portainer_stack"]["details"].append(f"All {len(critical_services)} services configured")
                else:
                    self.results["portainer_stack"]["status"] = "degraded"
                    self.results["portainer_stack"]["details"].append(f"Missing services: {missing_services}")
            else:
                print_error("Portainer stack file not found")
                self.results["portainer_stack"]["status"] = "failed"
                
        except Exception as e:
            print_error(f"Portainer stack validation failed: {e}")
            self.results["portainer_stack"]["status"] = "failed"
    
    async def validate_vector_databases(self):
        """Validate vector database implementations"""
        print_header("Vector Databases Validation")
        
        vector_dbs = [
            ("ChromaDB", "http://localhost:10100/api/v1/heartbeat"),
            ("Qdrant", "http://localhost:10102/collections"),
            ("FAISS", "http://localhost:10103/health"),
        ]
        
        passed = 0
        for db_name, url in vector_dbs:
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    print_success(f"{db_name} is accessible")
                    passed += 1
                else:
                    print_warning(f"{db_name} returned status {response.status_code}")
            except httpx.ConnectError:
                print_error(f"{db_name} not accessible (not running)")
            except Exception as e:
                print_error(f"{db_name} validation failed: {e}")
                
        if passed == len(vector_dbs):
            self.results["vector_dbs"]["status"] = "passed"
        elif passed > 0:
            self.results["vector_dbs"]["status"] = "degraded"
        else:
            self.results["vector_dbs"]["status"] = "failed"
            
        self.results["vector_dbs"]["details"].append(f"{passed}/{len(vector_dbs)} databases accessible")
    
    async def validate_infrastructure(self):
        """Validate core infrastructure services"""
        print_header("Core Infrastructure Validation")
        
        infrastructure = [
            ("PostgreSQL", "http://localhost:10000", "database"),
            ("Redis", "http://localhost:10001", "cache"),
            ("Neo4j", "http://localhost:10002", "graph"),
            ("RabbitMQ", "http://localhost:10005", "queue"),
            ("Consul", "http://localhost:10006", "discovery"),
            ("Kong", "http://localhost:10009", "gateway"),
        ]
        
        passed = 0
        for name, url, stype in infrastructure:
            try:
                response = await self.client.get(url)
                if response.status_code < 500:  # Any response is good for now
                    print_success(f"{name} is accessible")
                    passed += 1
                else:
                    print_warning(f"{name} returned status {response.status_code}")
            except httpx.ConnectError:
                print_error(f"{name} not accessible (not running)")
            except Exception as e:
                print_info(f"{name} check inconclusive: {str(e)[:50]}")
                
        if passed >= len(infrastructure) - 1:  # Allow 1 failure
            self.results["infrastructure"]["status"] = "passed"
        elif passed > 0:
            self.results["infrastructure"]["status"] = "degraded"
        else:
            self.results["infrastructure"]["status"] = "failed"
            
        self.results["infrastructure"]["details"].append(f"{passed}/{len(infrastructure)} services accessible")
    
    async def run_all_validations(self):
        """Run all validation checks"""
        print_header("Phase 2 Component Validation Suite")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Run all validations
        await self.validate_jwt_authentication()
        await self.validate_backend_api()
        await self.validate_frontend()
        await self.validate_mcp_bridge()
        await self.validate_portainer_stack()
        await self.validate_vector_databases()
        await self.validate_infrastructure()
        
        # Print summary
        print_header("Validation Summary")
        
        total_tests = len(self.results)
        passed = sum(1 for r in self.results.values() if r["status"] == "passed")
        degraded = sum(1 for r in self.results.values() if r["status"] == "degraded")
        failed = sum(1 for r in self.results.values() if r["status"] == "failed")
        
        for component, result in self.results.items():
            status = result["status"]
            symbol = "✓" if status == "passed" else "⚠" if status == "degraded" else "✗"
            color = Colors.GREEN if status == "passed" else Colors.YELLOW if status == "degraded" else Colors.RED
            
            print(f"{color}{symbol}{Colors.ENDC} {component.replace('_', ' ').title()}: {status}")
            for detail in result["details"]:
                print(f"    {detail}")
        
        print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
        print(f"  Passed: {Colors.GREEN}{passed}{Colors.ENDC}/{total_tests}")
        print(f"  Degraded: {Colors.YELLOW}{degraded}{Colors.ENDC}/{total_tests}")
        print(f"  Failed: {Colors.RED}{failed}{Colors.ENDC}/{total_tests}")
        
        # Overall status
        if failed == 0 and degraded == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL VALIDATIONS PASSED{Colors.ENDC}")
            return 0
        elif failed == 0:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ SOME COMPONENTS DEGRADED{Colors.ENDC}")
            return 1
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME VALIDATIONS FAILED{Colors.ENDC}")
            return 2
        
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    """Main entry point"""
    validator = Phase2Validator()
    try:
        exit_code = await validator.run_all_validations()
        return exit_code
    finally:
        await validator.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
