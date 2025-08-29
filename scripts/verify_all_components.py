#!/usr/bin/env python3
"""
Comprehensive System Component Verification Script
Validates all services, connections, and integrations in the Sutazai ecosystem
"""

import sys
import json
import time
import asyncio
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import httpx
import psycopg2
import redis
from neo4j import GraphDatabase
import pika
from pathlib import Path

# ANSI color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class ComponentVerifier:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "pending",
            "components": {},
            "errors": [],
            "warnings": []
        }
        self.critical_failures = []
        
    def print_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{title.center(60)}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")
        
    def print_status(self, component: str, status: str, details: str = ""):
        """Print component status with color coding"""
        if status == "healthy":
            color = Colors.GREEN
            symbol = "✓"
        elif status == "warning":
            color = Colors.YELLOW
            symbol = "⚠"
        else:
            color = Colors.RED
            symbol = "✗"
            
        print(f"{color}{symbol} {component:<30} {status:<10}{Colors.RESET} {details}")
        
    async def check_docker_services(self) -> Dict:
        """Check all Docker containers status"""
        self.print_header("Docker Services Health Check")
        services = {}
        
        try:
            # Get all containers
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "json"],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    container = json.loads(line)
                    name = container['Names']
                    state = container['State']
                    status = container['Status']
                    
                    health = "healthy" if state == "running" else "unhealthy"
                    services[name] = {
                        "state": state,
                        "status": status,
                        "health": health
                    }
                    
                    self.print_status(name, health, status)
                    
        except Exception as e:
            self.results["errors"].append(f"Docker check failed: {str(e)}")
            self.critical_failures.append("Docker services")
            
        self.results["components"]["docker"] = services
        return services
        
    async def check_postgresql(self) -> bool:
        """Verify PostgreSQL connectivity and basic operations"""
        self.print_header("PostgreSQL Database Check")
        
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=10000,
                database="jarvis_ai",
                user="jarvis",
                password="sutazai_secure_2024"
            )
            
            with conn.cursor() as cur:
                # Check version
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                
                # Check tables
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public';
                """)
                table_count = cur.fetchone()[0]
                
                # Check users table
                cur.execute("SELECT COUNT(*) FROM users;")
                user_count = cur.fetchone()[0]
                
            conn.close()
            
            self.print_status("PostgreSQL", "healthy", 
                            f"Tables: {table_count}, Users: {user_count}")
            
            self.results["components"]["postgresql"] = {
                "status": "healthy",
                "version": version,
                "tables": table_count,
                "users": user_count
            }
            return True
            
        except Exception as e:
            self.print_status("PostgreSQL", "unhealthy", str(e))
            self.results["components"]["postgresql"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            self.critical_failures.append("PostgreSQL")
            return False
            
    async def check_redis(self) -> bool:
        """Verify Redis connectivity and operations"""
        self.print_header("Redis Cache Check")
        
        try:
            r = redis.Redis(host='localhost', port=10001, decode_responses=True)
            
            # Ping test
            r.ping()
            
            # Get info
            info = r.info()
            memory = info.get('used_memory_human', 'N/A')
            clients = info.get('connected_clients', 0)
            
            # Test set/get
            r.set('test_key', 'test_value', ex=10)
            value = r.get('test_key')
            
            self.print_status("Redis", "healthy", 
                            f"Memory: {memory}, Clients: {clients}")
            
            self.results["components"]["redis"] = {
                "status": "healthy",
                "memory": memory,
                "clients": clients
            }
            return True
            
        except Exception as e:
            self.print_status("Redis", "unhealthy", str(e))
            self.results["components"]["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            self.critical_failures.append("Redis")
            return False
            
    async def check_neo4j(self) -> bool:
        """Verify Neo4j connectivity"""
        self.print_header("Neo4j Graph Database Check")
        
        try:
            driver = GraphDatabase.driver(
                "bolt://localhost:10003",
                auth=("neo4j", "sutazai_secure_2024")
            )
            
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()['count']
                
            driver.close()
            
            self.print_status("Neo4j", "healthy", f"Nodes: {node_count}")
            
            self.results["components"]["neo4j"] = {
                "status": "healthy",
                "nodes": node_count
            }
            return True
            
        except Exception as e:
            self.print_status("Neo4j", "unhealthy", str(e))
            self.results["components"]["neo4j"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            return False
            
    async def check_rabbitmq(self) -> bool:
        """Verify RabbitMQ connectivity"""
        self.print_header("RabbitMQ Message Queue Check")
        
        try:
            credentials = pika.PlainCredentials('sutazai', 'sutazai_secure_2024')
            connection = pika.BlockingConnection(
                pika.ConnectionParameters('localhost', 10004, '/', credentials)
            )
            channel = connection.channel()
            
            # Declare test queue
            channel.queue_declare(queue='test_queue', durable=False)
            
            # Get queue info
            method = channel.queue_declare(queue='test_queue', passive=True)
            message_count = method.method.message_count
            
            connection.close()
            
            self.print_status("RabbitMQ", "healthy", f"Messages: {message_count}")
            
            self.results["components"]["rabbitmq"] = {
                "status": "healthy",
                "messages": message_count
            }
            return True
            
        except Exception as e:
            self.print_status("RabbitMQ", "unhealthy", str(e))
            self.results["components"]["rabbitmq"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            return False
            
    async def check_backend_api(self) -> bool:
        """Check Backend FastAPI service"""
        self.print_header("Backend API Check")
        
        try:
            async with httpx.AsyncClient() as client:
                # Health check
                response = await client.get("http://localhost:10200/health")
                health_data = response.json()
                
                # API docs
                docs_response = await client.get("http://localhost:10200/docs")
                docs_available = docs_response.status_code == 200
                
                # Test auth endpoint
                auth_response = await client.post(
                    "http://localhost:10200/api/v1/auth/login",
                    json={"username": "test", "password": "test"}
                )
                auth_working = auth_response.status_code in [200, 401, 422]
                
                self.print_status("Backend API", "healthy", 
                                f"Docs: {docs_available}, Auth: {auth_working}")
                
                self.results["components"]["backend"] = {
                    "status": "healthy",
                    "health": health_data,
                    "docs": docs_available,
                    "auth": auth_working
                }
                return True
                
        except Exception as e:
            self.print_status("Backend API", "unhealthy", str(e))
            self.results["components"]["backend"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            self.critical_failures.append("Backend API")
            return False
            
    async def check_frontend(self) -> bool:
        """Check Streamlit frontend"""
        self.print_header("Frontend (Streamlit) Check")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11000/_stcore/health")
                
                if response.status_code == 200 and response.text.strip() == "ok":
                    self.print_status("Streamlit Frontend", "healthy", "UI accessible")
                    
                    self.results["components"]["frontend"] = {
                        "status": "healthy",
                        "accessible": True
                    }
                    return True
                else:
                    raise Exception(f"Health check returned {response.status_code}: {response.text}")
                    
        except Exception as e:
            self.print_status("Streamlit Frontend", "unhealthy", str(e))
            self.results["components"]["frontend"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            self.critical_failures.append("Frontend")
            return False
            
    async def check_mcp_bridge(self) -> bool:
        """Check MCP Bridge service"""
        self.print_header("MCP Bridge Check")
        
        try:
            async with httpx.AsyncClient() as client:
                # Health check
                response = await client.get("http://localhost:11100/health")
                health_data = response.json()
                
                # Get agents
                agents_response = await client.get("http://localhost:11100/agents")
                agents = agents_response.json()
                
                # Handle both dict and list formats
                if isinstance(agents, dict):
                    agent_list = list(agents.values())
                else:
                    agent_list = agents
                    
                online_count = sum(1 for a in agent_list if a.get('status') == 'online')
                total_count = len(agent_list)
                
                self.print_status("MCP Bridge", "healthy", 
                                f"Agents: {online_count}/{total_count} online")
                
                self.results["components"]["mcp_bridge"] = {
                    "status": "healthy",
                    "health": health_data,
                    "agents_online": online_count,
                    "agents_total": total_count
                }
                return True
                
        except Exception as e:
            self.print_status("MCP Bridge", "unhealthy", str(e))
            self.results["components"]["mcp_bridge"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            self.critical_failures.append("MCP Bridge")
            return False
            
    async def check_vector_databases(self) -> bool:
        """Check vector database services"""
        self.print_header("Vector Databases Check")
        
        vector_dbs = {
            "ChromaDB": "http://localhost:10100/api/v1/heartbeat",
            "Qdrant": "http://localhost:10101/health",
            "FAISS": "http://localhost:10103/health"
        }
        
        all_healthy = True
        for name, url in vector_dbs.items():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        self.print_status(name, "healthy", "Service responsive")
                        self.results["components"][name.lower()] = {"status": "healthy"}
                    else:
                        raise Exception(f"Status code: {response.status_code}")
                        
            except Exception as e:
                self.print_status(name, "unhealthy", str(e))
                self.results["components"][name.lower()] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                all_healthy = False
                
        return all_healthy
        
    async def check_mcp_servers(self) -> bool:
        """Check individual MCP server wrappers"""
        self.print_header("MCP Server Wrappers Check")
        
        wrapper_dir = Path("/opt/sutazaiapp/scripts/mcp/wrappers")
        all_healthy = True
        
        for wrapper in wrapper_dir.glob("*.sh"):
            name = wrapper.stem
            try:
                result = subprocess.run(
                    [str(wrapper), "--selfcheck"],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0:
                    self.print_status(f"MCP:{name}", "healthy", "Self-check passed")
                else:
                    self.print_status(f"MCP:{name}", "unhealthy", result.stderr[:50])
                    all_healthy = False
                    
            except subprocess.TimeoutExpired:
                self.print_status(f"MCP:{name}", "unhealthy", "Timeout")
                all_healthy = False
            except Exception as e:
                self.print_status(f"MCP:{name}", "unhealthy", str(e)[:50])
                all_healthy = False
                
        return all_healthy
        
    async def generate_report(self):
        """Generate final verification report"""
        self.print_header("Verification Summary")
        
        # Calculate overall status
        if self.critical_failures:
            self.results["overall_status"] = "critical"
            print(f"{Colors.RED}{Colors.BOLD}CRITICAL FAILURES DETECTED:{Colors.RESET}")
            for failure in self.critical_failures:
                print(f"  {Colors.RED}✗ {failure}{Colors.RESET}")
        else:
            healthy_count = sum(
                1 for c in self.results["components"].values()
                if isinstance(c, dict) and c.get("status") == "healthy"
            )
            total_count = len(self.results["components"])
            
            if healthy_count == total_count:
                self.results["overall_status"] = "healthy"
                print(f"{Colors.GREEN}{Colors.BOLD}ALL SYSTEMS OPERATIONAL{Colors.RESET}")
            else:
                self.results["overall_status"] = "degraded"
                print(f"{Colors.YELLOW}{Colors.BOLD}SYSTEM DEGRADED{Colors.RESET}")
                
        print(f"\n{Colors.CYAN}Components Checked: {len(self.results['components'])}{Colors.RESET}")
        print(f"{Colors.CYAN}Errors: {len(self.results['errors'])}{Colors.RESET}")
        print(f"{Colors.CYAN}Warnings: {len(self.results['warnings'])}{Colors.RESET}")
        
        # Save results to file
        report_file = Path("/opt/sutazaiapp/verification_report.json")
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\n{Colors.BLUE}Full report saved to: {report_file}{Colors.RESET}")
        
        return self.results["overall_status"] == "healthy"
        
    async def run_verification(self):
        """Run all verification checks"""
        print(f"{Colors.BOLD}{Colors.MAGENTA}")
        print("╔══════════════════════════════════════════════════════════╗")
        print("║     SUTAZAI COMPREHENSIVE SYSTEM VERIFICATION v1.0      ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"{Colors.RESET}")
        
        # Run all checks
        await self.check_docker_services()
        await self.check_postgresql()
        await self.check_redis()
        await self.check_neo4j()
        await self.check_rabbitmq()
        await self.check_backend_api()
        await self.check_frontend()
        await self.check_mcp_bridge()
        await self.check_vector_databases()
        await self.check_mcp_servers()
        
        # Generate report
        success = await self.generate_report()
        
        return 0 if success else 1

async def main():
    verifier = ComponentVerifier()
    return await verifier.run_verification()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))