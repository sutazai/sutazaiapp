#!/usr/bin/env python3
"""
SutazAI Comprehensive Deployment Verification Script
========================================================

This script provides thorough verification of the SutazAI system deployment including:
1. Health checks for all deployed services
2. API endpoint validation 
3. Agent communication testing
4. Database connectivity verification
5. Ollama model validation
6. Resource usage monitoring
7. Comprehensive deployment reporting

Created by: Testing QA Validator Agent
Version: 1.0.0
"""

import asyncio
import sys
import time
import json
import logging
import subprocess
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Try to import optional dependencies with graceful fallbacks
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not available - HTTP checks will be skipped")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available - system metrics will be limited")

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    print("Warning: asyncpg not available - PostgreSQL checks will be skipped")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis not available - Redis checks will be skipped")

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    print("Warning: docker not available - Docker checks will be limited")

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j not available - Neo4j checks will be skipped")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: pyyaml not available - configuration loading will be limited")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'/opt/sutazaiapp/logs/deployment_verification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Color codes for console output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


class DeploymentVerifier:
    """Comprehensive deployment verification system"""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {
            'services': {},
            'apis': {},
            'agents': {},
            'databases': {},
            'models': {},
            'resources': {},
            'overall': {
                'status': 'UNKNOWN',
                'score': 0,
                'timestamp': datetime.now().isoformat()
            }
        }
        self.docker_client = None
        self.total_checks = 0
        self.passed_checks = 0
        
        # Service configurations
        self.services = {
            'postgres': {'container': 'sutazai-postgres', 'port': 5432, 'health_url': None},
            'redis': {'container': 'sutazai-redis', 'port': 6379, 'health_url': None},
            'neo4j': {'container': 'sutazai-neo4j', 'port': 7687, 'health_url': 'http://localhost:7474'},
            'chromadb': {'container': 'sutazai-chromadb', 'port': 8001, 'health_url': 'http://localhost:8001/api/v1/heartbeat'},
            'qdrant': {'container': 'sutazai-qdrant', 'port': 6333, 'health_url': 'http://localhost:6333/healthz'},
            'ollama': {'container': 'sutazai-ollama', 'port': 11434, 'health_url': 'http://localhost:11434/api/tags'},
            'backend': {'container': 'sutazai-backend', 'port': 8000, 'health_url': 'http://localhost:8000/health'},
            'frontend': {'container': 'sutazai-frontend', 'port': 8501, 'health_url': 'http://localhost:8501/healthz'},
            'litellm': {'container': 'sutazai-litellm', 'port': 4000, 'health_url': 'http://localhost:4000/health'},
            'prometheus': {'container': 'sutazai-prometheus', 'port': 9090, 'health_url': 'http://localhost:9090/-/healthy'},
            'grafana': {'container': 'sutazai-grafana', 'port': 3000, 'health_url': 'http://localhost:3000/api/health'},
            'langflow': {'container': 'sutazai-langflow', 'port': 8090, 'health_url': 'http://localhost:8090/health'},
            'flowise': {'container': 'sutazai-flowise', 'port': 8099, 'health_url': 'http://localhost:8099/api/v1/ping'},
            'dify': {'container': 'sutazai-dify', 'port': 8107, 'health_url': 'http://localhost:8107'},
            'n8n': {'container': 'sutazai-n8n', 'port': 5678, 'health_url': 'http://localhost:5678/healthz'},
        }
        
        # Agent configurations
        self.agents = {
            'autogpt': {'container': 'sutazai-autogpt', 'port': 8080, 'health_url': 'http://localhost:8080/health'},
            'crewai': {'container': 'sutazai-crewai', 'port': 8096, 'health_url': 'http://localhost:8096/health'},
            'aider': {'container': 'sutazai-aider', 'port': 8095, 'health_url': 'http://localhost:8095/health'},
            'gpt-engineer': {'container': 'sutazai-gpt-engineer', 'port': 8097, 'health_url': 'http://localhost:8097/health'},
            'letta': {'container': 'sutazai-letta', 'port': None, 'health_url': None},
            'agentgpt': {'container': 'sutazai-agentgpt', 'port': 8091, 'health_url': 'http://localhost:8091'},
            'privategpt': {'container': 'sutazai-privategpt', 'port': 8092, 'health_url': 'http://localhost:8092'},
            'agentzero': {'container': 'sutazai-agentzero', 'port': 8105, 'health_url': 'http://localhost:8105/health'},
        }
        
        # API endpoints to test
        self.api_endpoints = {
            'health': {'url': 'http://localhost:8000/health', 'method': 'GET', 'expected_status': 200},
            'agents': {'url': 'http://localhost:8000/agents', 'method': 'GET', 'expected_status': 200},
            'models': {'url': 'http://localhost:8000/models', 'method': 'GET', 'expected_status': 200},
            'metrics': {'url': 'http://localhost:8000/public/metrics', 'method': 'GET', 'expected_status': 200},
            'chat': {'url': 'http://localhost:8000/chat', 'method': 'POST', 'expected_status': 200,
                    'payload': {'message': 'Hello, test message', 'model': 'test'}},
            'think': {'url': 'http://localhost:8000/public/think', 'method': 'POST', 'expected_status': 200,
                     'payload': {'query': 'Test reasoning query', 'reasoning_type': 'general'}},
            'ollama_tags': {'url': 'http://localhost:11434/api/tags', 'method': 'GET', 'expected_status': 200},
            'chromadb_heartbeat': {'url': 'http://localhost:8001/api/v1/heartbeat', 'method': 'GET', 'expected_status': 200},
            'qdrant_cluster': {'url': 'http://localhost:6333/cluster', 'method': 'GET', 'expected_status': 200},
        }
    
    def print_banner(self):
        """Print verification banner"""
        print(f"\n{Colors.CYAN}{'='*80}{Colors.NC}")
        print(f"{Colors.CYAN}{Colors.BOLD}🔍 SutazAI Comprehensive Deployment Verification{Colors.NC}")
        print(f"{Colors.CYAN}Version: 1.0.0 | Testing QA Validator Agent{Colors.NC}")
        print(f"{Colors.CYAN}{'='*80}{Colors.NC}\n")
    
    def print_section(self, title: str, icon: str = "📋"):
        """Print section header"""
        print(f"\n{Colors.YELLOW}{Colors.BOLD}{icon} {title}{Colors.NC}")
        print(f"{Colors.YELLOW}{'-'*50}{Colors.NC}")
    
    async def check_port(self, host: str, port: int) -> bool:
        """Check if a port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    async def check_http_endpoint(self, url: str, method: str = 'GET', 
                                 payload: Optional[Dict] = None, 
                                 expected_status: int = 200,
                                 timeout: int = 10) -> Tuple[bool, Optional[Dict]]:
        """Check HTTP endpoint"""
        if not AIOHTTP_AVAILABLE:
            logger.debug(f"aiohttp not available, skipping HTTP check for {url}")
            return False, None
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                if method.upper() == 'GET':
                    async with session.get(url) as response:
                        data = await response.text()
                        try:
                            json_data = await response.json()
                        except:
                            json_data = None
                        return response.status == expected_status, json_data
                elif method.upper() == 'POST':
                    async with session.post(url, json=payload) as response:
                        try:
                            json_data = await response.json()
                        except:
                            json_data = None
                        return response.status == expected_status, json_data
        except Exception as e:
            logger.debug(f"HTTP check failed for {url}: {e}")
            return False, None
    
    def init_docker(self):
        """Initialize Docker client"""
        if not DOCKER_AVAILABLE:
            logger.warning("Docker module not available")
            return False
        try:
            self.docker_client = docker.from_env()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            return False
    
    async def check_docker_containers(self):
        """Check Docker container status"""
        self.print_section("Docker Container Status", "🐳")
        
        if not self.init_docker():
            print(f"  {Colors.RED}❌ Docker client initialization failed{Colors.NC}")
            return
        
        try:
            containers = self.docker_client.containers.list(all=True)
            sutazai_containers = [c for c in containers if c.name.startswith('sutazai-')]
            
            running_count = 0
            total_count = len(sutazai_containers)
            
            for container in sutazai_containers:
                status = container.status
                if status == 'running':
                    print(f"  {Colors.GREEN}✅ {container.name:<25} {status}{Colors.NC}")
                    running_count += 1
                    self.passed_checks += 1
                elif status == 'exited':
                    print(f"  {Colors.RED}❌ {container.name:<25} {status}{Colors.NC}")
                else:
                    print(f"  {Colors.YELLOW}⚠️ {container.name:<25} {status}{Colors.NC}")
                
                self.total_checks += 1
            
            print(f"\n  {Colors.BLUE}📊 Container Summary: {running_count}/{total_count} running{Colors.NC}")
            self.results['services']['docker'] = {
                'status': 'healthy' if running_count > total_count * 0.8 else 'degraded',
                'running': running_count,
                'total': total_count,
                'details': {c.name: c.status for c in sutazai_containers}
            }
            
        except Exception as e:
            logger.error(f"Docker container check failed: {e}")
            print(f"  {Colors.RED}❌ Docker container check failed: {e}{Colors.NC}")
    
    async def check_service_health(self):
        """Check service health"""
        self.print_section("Service Health Checks", "🏥")
        
        for service_name, config in self.services.items():
            try:
                port_open = await self.check_port('localhost', config['port']) if config['port'] else False
                
                if config['health_url']:
                    health_ok, health_data = await self.check_http_endpoint(config['health_url'])
                else:
                    health_ok = port_open
                    health_data = None
                
                if health_ok:
                    print(f"  {Colors.GREEN}✅ {service_name:<20} healthy{Colors.NC}")
                    status = 'healthy'
                    self.passed_checks += 1
                elif port_open:
                    print(f"  {Colors.YELLOW}⚠️ {service_name:<20} port open, health check failed{Colors.NC}")
                    status = 'degraded'
                else:
                    print(f"  {Colors.RED}❌ {service_name:<20} unavailable{Colors.NC}")
                    status = 'unhealthy'
                
                self.results['services'][service_name] = {
                    'status': status,
                    'port_open': port_open,
                    'health_check': health_ok,
                    'health_data': health_data
                }
                self.total_checks += 1
                
            except Exception as e:
                logger.error(f"Service check failed for {service_name}: {e}")
                print(f"  {Colors.RED}❌ {service_name:<20} check failed: {e}{Colors.NC}")
                self.results['services'][service_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                self.total_checks += 1
    
    async def validate_api_endpoints(self):
        """Validate API endpoints"""
        self.print_section("API Endpoint Validation", "🔗")
        
        for endpoint_name, config in self.api_endpoints.items():
            try:
                success, response_data = await self.check_http_endpoint(
                    config['url'],
                    config['method'],
                    config.get('payload'),
                    config['expected_status']
                )
                
                if success:
                    print(f"  {Colors.GREEN}✅ {endpoint_name:<25} responding correctly{Colors.NC}")
                    status = 'healthy'
                    self.passed_checks += 1
                else:
                    print(f"  {Colors.RED}❌ {endpoint_name:<25} failed validation{Colors.NC}")
                    status = 'unhealthy'
                
                self.results['apis'][endpoint_name] = {
                    'status': status,
                    'url': config['url'],
                    'method': config['method'],
                    'response_data': response_data if success else None
                }
                self.total_checks += 1
                
            except Exception as e:
                logger.error(f"API validation failed for {endpoint_name}: {e}")
                print(f"  {Colors.RED}❌ {endpoint_name:<25} validation error: {e}{Colors.NC}")
                self.results['apis'][endpoint_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                self.total_checks += 1
    
    async def test_agent_communication(self):
        """Test agent communication"""
        self.print_section("Agent Communication Testing", "🤖")
        
        for agent_name, config in self.agents.items():
            try:
                # Check container status
                container_running = False
                if self.docker_client:
                    try:
                        container = self.docker_client.containers.get(config['container'])
                        container_running = container.status == 'running'
                    except:
                        pass
                
                # Check health endpoint if available
                health_ok = False
                if config['health_url'] and container_running:
                    health_ok, _ = await self.check_http_endpoint(config['health_url'], timeout=5)
                elif config['port'] and container_running:
                    health_ok = await self.check_port('localhost', config['port'])
                
                if container_running and health_ok:
                    print(f"  {Colors.GREEN}✅ {agent_name:<20} communicating{Colors.NC}")
                    status = 'healthy'
                    self.passed_checks += 1
                elif container_running:
                    print(f"  {Colors.YELLOW}⚠️ {agent_name:<20} running but not responding{Colors.NC}")
                    status = 'degraded'
                else:
                    print(f"  {Colors.RED}❌ {agent_name:<20} not running{Colors.NC}")
                    status = 'unhealthy'
                
                self.results['agents'][agent_name] = {
                    'status': status,
                    'container_running': container_running,
                    'health_check': health_ok
                }
                self.total_checks += 1
                
            except Exception as e:
                logger.error(f"Agent communication test failed for {agent_name}: {e}")
                print(f"  {Colors.RED}❌ {agent_name:<20} test error: {e}{Colors.NC}")
                self.results['agents'][agent_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                self.total_checks += 1
    
    async def verify_database_connectivity(self):
        """Verify database connectivity"""
        self.print_section("Database Connectivity", "🗄️")
        
        # PostgreSQL
        if ASYNCPG_AVAILABLE:
            try:
                conn = await asyncpg.connect(
                    'postgresql://sutazai:sutazai_password@localhost:5432/sutazai',
                    timeout=5
                )
                result = await conn.fetchval('SELECT version()')
                await conn.close()
                print(f"  {Colors.GREEN}✅ PostgreSQL connected{Colors.NC}")
                self.results['databases']['postgres'] = {
                    'status': 'healthy',
                    'version': result.split()[1] if result else 'unknown'
                }
                self.passed_checks += 1
            except Exception as e:
                print(f"  {Colors.RED}❌ PostgreSQL connection failed: {e}{Colors.NC}")
                self.results['databases']['postgres'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            print(f"  {Colors.YELLOW}⚠️ PostgreSQL check skipped (asyncpg not available){Colors.NC}")
            self.results['databases']['postgres'] = {
                'status': 'skipped',
                'reason': 'asyncpg not available'
            }
        self.total_checks += 1
        
        # Redis
        if REDIS_AVAILABLE:
            try:
                r = redis.Redis(host='localhost', port=6379, password='redis_password', 
                              socket_timeout=5, socket_connect_timeout=5)
                info = r.info()
                print(f"  {Colors.GREEN}✅ Redis connected{Colors.NC}")
                self.results['databases']['redis'] = {
                    'status': 'healthy',
                    'version': info.get('redis_version', 'unknown')
                }
                self.passed_checks += 1
            except Exception as e:
                print(f"  {Colors.RED}❌ Redis connection failed: {e}{Colors.NC}")
                self.results['databases']['redis'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            print(f"  {Colors.YELLOW}⚠️ Redis check skipped (redis module not available){Colors.NC}")
            self.results['databases']['redis'] = {
                'status': 'skipped',
                'reason': 'redis module not available'
            }
        self.total_checks += 1
        
        # Neo4j
        if NEO4J_AVAILABLE:
            try:
                driver = GraphDatabase.driver(
                    'bolt://localhost:7687',
                    auth=('neo4j', 'sutazai_neo4j_password')
                )
                with driver.session() as session:
                    result = session.run('CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version')
                    record = result.single()
                    driver.close()
                print(f"  {Colors.GREEN}✅ Neo4j connected{Colors.NC}")
                self.results['databases']['neo4j'] = {
                    'status': 'healthy',
                    'version': record['version'] if record else 'unknown'
                }
                self.passed_checks += 1
            except Exception as e:
                print(f"  {Colors.RED}❌ Neo4j connection failed: {e}{Colors.NC}")
                self.results['databases']['neo4j'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            print(f"  {Colors.YELLOW}⚠️ Neo4j check skipped (neo4j module not available){Colors.NC}")
            self.results['databases']['neo4j'] = {
                'status': 'skipped',
                'reason': 'neo4j module not available'
            }
        self.total_checks += 1
    
    async def confirm_ollama_models(self):
        """Confirm Ollama models are loaded"""
        self.print_section("Ollama Model Verification", "🧠")
        
        try:
            success, models_data = await self.check_http_endpoint('http://localhost:11434/api/tags')
            
            if success and models_data:
                models = models_data.get('models', [])
                model_count = len(models)
                
                print(f"  {Colors.BLUE}📊 Total models loaded: {model_count}{Colors.NC}")
                
                if model_count > 0:
                    print(f"  {Colors.GREEN}✅ Ollama models available{Colors.NC}")
                    for i, model in enumerate(models[:5]):  # Show first 5 models
                        model_name = model.get('name', 'unknown')
                        model_size = model.get('size', 0)
                        size_gb = round(model_size / (1024**3), 2) if model_size else 0
                        print(f"    {Colors.CYAN}• {model_name} ({size_gb}GB){Colors.NC}")
                    
                    if model_count > 5:
                        print(f"    {Colors.CYAN}... and {model_count - 5} more models{Colors.NC}")
                    
                    # Test model inference
                    test_payload = {
                        'model': models[0]['name'],
                        'prompt': 'Hello, this is a test.',
                        'stream': False
                    }
                    
                    inference_success, inference_data = await self.check_http_endpoint(
                        'http://localhost:11434/api/generate',
                        'POST',
                        test_payload,
                        timeout=30
                    )
                    
                    if inference_success:
                        print(f"  {Colors.GREEN}✅ Model inference working{Colors.NC}")
                        self.results['models']['inference_test'] = 'passed'
                        self.passed_checks += 1
                    else:
                        print(f"  {Colors.YELLOW}⚠️ Model inference test failed{Colors.NC}")
                        self.results['models']['inference_test'] = 'failed'
                    
                    self.results['models']['ollama'] = {
                        'status': 'healthy',
                        'model_count': model_count,
                        'models': [m['name'] for m in models]
                    }
                    self.passed_checks += 1
                else:
                    print(f"  {Colors.RED}❌ No models loaded in Ollama{Colors.NC}")
                    self.results['models']['ollama'] = {
                        'status': 'unhealthy',
                        'model_count': 0,
                        'error': 'No models loaded'
                    }
            else:
                print(f"  {Colors.RED}❌ Could not retrieve Ollama models{Colors.NC}")
                self.results['models']['ollama'] = {
                    'status': 'unhealthy',
                    'error': 'API call failed'
                }
                
        except Exception as e:
            logger.error(f"Ollama model verification failed: {e}")
            print(f"  {Colors.RED}❌ Ollama model check error: {e}{Colors.NC}")
            self.results['models']['ollama'] = {
                'status': 'error',
                'error': str(e)
            }
        
        self.total_checks += 1
    
    async def check_resource_usage(self):
        """Check resource usage"""
        self.print_section("Resource Usage Monitoring", "📊")
        
        if not PSUTIL_AVAILABLE:
            print(f"  {Colors.YELLOW}⚠️ psutil not available - using basic resource checks{Colors.NC}")
            # Basic fallback resource checks
            try:
                # Try to get basic info from /proc if available
                with open('/proc/loadavg', 'r') as f:
                    load_avg = f.read().strip().split()[:3]
                    print(f"  {Colors.BLUE}⚖️  Load Average: {', '.join(load_avg)}{Colors.NC}")
                    
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            mem_total = int(line.split()[1]) // 1024  # MB
                            print(f"  {Colors.BLUE}🧠 Total Memory: {mem_total}MB{Colors.NC}")
                            break
            except:
                print(f"  {Colors.YELLOW}⚠️ Basic resource info not available{Colors.NC}")
            
            self.results['resources'] = {
                'status': 'limited',
                'message': 'psutil not available - limited resource monitoring'
            }
            self.total_checks += 1
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_gb = round(memory.total / (1024**3), 2)
            memory_used_gb = round(memory.used / (1024**3), 2)
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_total_gb = round(disk.total / (1024**3), 2)
            disk_used_gb = round(disk.used / (1024**3), 2)
            disk_percent = round((disk.used / disk.total) * 100, 2)
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
            except:
                load_avg = (0, 0, 0)
            
            print(f"  {Colors.BLUE}🖥️  CPU Usage: {cpu_percent}% ({cpu_count} cores){Colors.NC}")
            print(f"  {Colors.BLUE}🧠 Memory: {memory_used_gb}GB / {memory_gb}GB ({memory_percent}%){Colors.NC}")
            print(f"  {Colors.BLUE}💽 Disk: {disk_used_gb}GB / {disk_total_gb}GB ({disk_percent}%){Colors.NC}")
            print(f"  {Colors.BLUE}⚖️  Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}{Colors.NC}")
            
            # Resource status assessment
            resource_issues = []
            if cpu_percent > 90:
                resource_issues.append("High CPU usage")
            if memory_percent > 90:
                resource_issues.append("High memory usage")
            if disk_percent > 90:
                resource_issues.append("High disk usage")
            
            if not resource_issues:
                print(f"  {Colors.GREEN}✅ Resource usage within limits{Colors.NC}")
                resource_status = 'healthy'
                self.passed_checks += 1
            else:
                print(f"  {Colors.YELLOW}⚠️ Resource concerns: {', '.join(resource_issues)}{Colors.NC}")
                resource_status = 'warning'
            
            self.results['resources'] = {
                'status': resource_status,
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'memory_total_gb': memory_gb,
                'memory_used_gb': memory_used_gb,
                'disk_percent': disk_percent,
                'disk_total_gb': disk_total_gb,
                'disk_used_gb': disk_used_gb,
                'load_average': load_avg,
                'issues': resource_issues
            }
            
        except Exception as e:
            logger.error(f"Resource usage check failed: {e}")
            print(f"  {Colors.RED}❌ Resource check error: {e}{Colors.NC}")
            self.results['resources'] = {
                'status': 'error',
                'error': str(e)
            }
        
        self.total_checks += 1
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        self.print_section("Deployment Report Generation", "📋")
        
        # Calculate overall score
        if self.total_checks > 0:
            score = round((self.passed_checks / self.total_checks) * 100, 2)
        else:
            score = 0
        
        # Determine overall status
        if score >= 90:
            overall_status = 'EXCELLENT'
            status_color = Colors.GREEN
        elif score >= 80:
            overall_status = 'GOOD'
            status_color = Colors.GREEN
        elif score >= 70:
            overall_status = 'ACCEPTABLE'
            status_color = Colors.YELLOW
        elif score >= 50:
            overall_status = 'DEGRADED'
            status_color = Colors.YELLOW
        else:
            overall_status = 'CRITICAL'
            status_color = Colors.RED
        
        self.results['overall'] = {
            'status': overall_status,
            'score': score,
            'passed_checks': self.passed_checks,
            'total_checks': self.total_checks,
            'timestamp': datetime.now().isoformat(),
            'verification_time': round(time.time() - self.start_time, 2)
        }
        
        # Print summary
        print(f"\n{Colors.CYAN}{'='*80}{Colors.NC}")
        print(f"{Colors.CYAN}{Colors.BOLD}📊 DEPLOYMENT VERIFICATION SUMMARY{Colors.NC}")
        print(f"{Colors.CYAN}{'='*80}{Colors.NC}")
        
        print(f"\n{status_color}{Colors.BOLD}Overall Status: {overall_status} ({score}%){Colors.NC}")
        print(f"{Colors.BLUE}Verification Time: {self.results['overall']['verification_time']}s{Colors.NC}")
        print(f"{Colors.BLUE}Checks Passed: {self.passed_checks}/{self.total_checks}{Colors.NC}")
        
        # Service status summary
        healthy_services = sum(1 for s in self.results['services'].values() 
                             if s.get('status') == 'healthy')
        total_services = len(self.results['services'])
        print(f"{Colors.BLUE}Healthy Services: {healthy_services}/{total_services}{Colors.NC}")
        
        # Agent status summary
        healthy_agents = sum(1 for a in self.results['agents'].values() 
                           if a.get('status') == 'healthy')
        total_agents = len(self.results['agents'])
        print(f"{Colors.BLUE}Communicating Agents: {healthy_agents}/{total_agents}{Colors.NC}")
        
        # Database status summary
        healthy_dbs = sum(1 for d in self.results['databases'].values() 
                         if d.get('status') == 'healthy')
        total_dbs = len(self.results['databases'])
        print(f"{Colors.BLUE}Connected Databases: {healthy_dbs}/{total_dbs}{Colors.NC}")
        
        # Model status
        model_status = self.results.get('models', {}).get('ollama', {}).get('status', 'unknown')
        model_count = self.results.get('models', {}).get('ollama', {}).get('model_count', 0)
        print(f"{Colors.BLUE}Ollama Models: {model_count} loaded ({model_status}){Colors.NC}")
        
        # Access points
        if score >= 70:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}🌐 Key Access Points:{Colors.NC}")
            print(f"  • Frontend UI: http://localhost:8501")
            print(f"  • Backend API: http://localhost:8000")
            print(f"  • API Docs: http://localhost:8000/docs")
            print(f"  • Grafana: http://localhost:3000")
            print(f"  • LangFlow: http://localhost:8090")
            print(f"  • Flowise: http://localhost:8099")
            print(f"  • Dify: http://localhost:8107")
            print(f"  • n8n: http://localhost:5678")
        
        # Recommendations
        print(f"\n{Colors.YELLOW}{Colors.BOLD}📌 Recommendations:{Colors.NC}")
        if score >= 90:
            print(f"  • System is ready for production use")
            print(f"  • Monitor resource usage regularly")
            print(f"  • Consider setting up automated monitoring")
        elif score >= 80:
            print(f"  • System is functional with minor issues")
            print(f"  • Review failed checks and address issues")
            print(f"  • Monitor degraded services")
        elif score >= 70:
            print(f"  • System has acceptable functionality")
            print(f"  • Address failed health checks")
            print(f"  • Consider scaling resources if needed")
        else:
            print(f"  • System requires immediate attention")
            print(f"  • Review logs for critical services")
            print(f"  • Check docker-compose configurations")
            print(f"  • Verify network connectivity")
        
        # Save report to file
        report_file = f'/opt/sutazaiapp/logs/deployment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n{Colors.GREEN}📄 Detailed report saved: {report_file}{Colors.NC}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ Failed to save report: {e}{Colors.NC}")
        
        print(f"\n{Colors.CYAN}{'='*80}{Colors.NC}")
        
        return overall_status, score
    
    async def run_verification(self):
        """Run complete verification process"""
        self.print_banner()
        
        try:
            # Run all verification steps
            await self.check_docker_containers()
            await self.check_service_health()
            await self.validate_api_endpoints()
            await self.test_agent_communication()
            await self.verify_database_connectivity()
            await self.confirm_ollama_models()
            await self.check_resource_usage()
            
            # Generate final report
            status, score = self.generate_deployment_report()
            
            # Return appropriate exit code
            if score >= 80:
                return 0  # Success
            elif score >= 60:
                return 1  # Warning
            else:
                return 2  # Critical
                
        except Exception as e:
            logger.error(f"Verification process failed: {e}")
            print(f"\n{Colors.RED}❌ Verification process failed: {e}{Colors.NC}")
            return 3  # Error


async def main():
    """Main entry point"""
    verifier = DeploymentVerifier()
    exit_code = await verifier.run_verification()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())