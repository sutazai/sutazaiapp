#!/usr/bin/env python3
"""
SutazAI Service Health Checker
Purpose: Dedicated health checking for core services including Ollama, databases, and AI components
Usage: python service_health_checker.py [--service SERVICE_NAME] [--all]
Requirements: requests, psycopg2, redis, neo4j
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import argparse

# Third-party imports with graceful degradation
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/service_health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ServiceHealthChecker')

class ServiceHealthChecker:
    def __init__(self):
        self.health_results = {}
        
    async def check_ollama_health(self) -> Dict[str, Any]:
        """Check Ollama service health and available models"""
        if not HAS_REQUESTS:
            return {'healthy': False, 'error': 'requests library not available'}
            
        try:
            # Check if Ollama is running by looking for process or container
            import subprocess
            
            # Try to find Ollama process
            ollama_running = False
            try:
                result = subprocess.run(['pgrep', '-f', 'ollama'], capture_output=True, text=True)
                if result.returncode == 0:
                    ollama_running = True
            except Exception as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
            
            # Also check for Ollama container
            try:
                result = subprocess.run(['docker', 'ps', '--filter', 'name=ollama', '--format', '{{.Names}}'], 
                                      capture_output=True, text=True)
                if 'ollama' in result.stdout:
                    ollama_running = True
            except Exception as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
            
            if not ollama_running:
                return {
                    'healthy': False,
                    'error': 'Ollama process/container not found',
                    'models': [],
                    'version': None
                }
            
            # Try different Ollama endpoints
            endpoints = [
                'http://localhost:11434',
                'http://127.0.0.1:11434',
                'http://ollama:11434'
            ]
            
            for base_url in endpoints:
                try:
                    # Check health endpoint
                    health_response = requests.get(f"{base_url}/api/tags", timeout=10)
                    if health_response.status_code == 200:
                        models_data = health_response.json()
                        
                        # Get version info
                        try:
                            version_response = requests.get(f"{base_url}/api/version", timeout=5)
                            version_info = version_response.json() if version_response.status_code == 200 else {}
                        except Exception as e:
                            logger.error(f"Unexpected exception: {e}", exc_info=True)
                            version_info = {}
                        
                        return {
                            'healthy': True,
                            'base_url': base_url,
                            'models': [model['name'] for model in models_data.get('models', [])],
                            'model_count': len(models_data.get('models', [])),
                            'version': version_info.get('version', 'unknown'),
                            'response_time': health_response.elapsed.total_seconds()
                        }
                except requests.exceptions.RequestException as e:
                    continue
            
            return {
                'healthy': False,
                'error': 'Could not connect to any Ollama endpoint',
                'endpoints_tried': endpoints
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': f'Unexpected error checking Ollama: {str(e)}'
            }
    
    async def check_postgres_health(self, host: str = 'localhost', port: int = 5432, 
                                  database: str = 'postgres', user: str = 'postgres', 
                                  password: str = None) -> Dict[str, Any]:
        """Check PostgreSQL database health"""
        if not HAS_PSYCOPG2:
            return {'healthy': False, 'error': 'psycopg2 library not available'}
        
        try:
            # Try to get password from environment or secrets
            if not password:
                import os
                password = os.environ.get('POSTGRES_PASSWORD', 'sutazai')
                
                # Try to read from secrets file
                try:
                    with open('/opt/sutazaiapp/secrets/postgres_password.txt', 'r') as f:
                        password = f.read().strip()
                except Exception as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
            
            start_time = time.time()
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                connect_timeout=10
            )
            
            cursor = conn.cursor()
            
            # Get database version
            cursor.execute('SELECT version();')
            version = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute(f"SELECT pg_size_pretty(pg_database_size('{database}'));")
            db_size = cursor.fetchone()[0]
            
            # Get connection count
            cursor.execute('SELECT count(*) FROM pg_stat_activity;')
            connection_count = cursor.fetchone()[0]
            
            # Get uptime
            cursor.execute('SELECT extract(epoch from now() - pg_postmaster_start_time());')
            uptime_seconds = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            return {
                'healthy': True,
                'version': version,
                'database_size': db_size,
                'connection_count': connection_count,
                'uptime_seconds': float(uptime_seconds),
                'response_time': response_time,
                'host': host,
                'port': port,
                'database': database
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'host': host,
                'port': port,
                'database': database
            }
    
    async def check_redis_health(self, host: str = 'localhost', port: int = 6379, 
                               password: str = None) -> Dict[str, Any]:
        """Check Redis cache health"""
        if not HAS_REDIS:
            return {'healthy': False, 'error': 'redis library not available'}
        
        try:
            start_time = time.time()
            
            r = redis.Redis(host=host, port=port, password=password, socket_timeout=10)
            
            # Ping Redis
            ping_result = r.ping()
            
            # Get Redis info
            info = r.info()
            
            # Test set/get operation
            test_key = f"health_check_{int(time.time())}"
            r.set(test_key, "test_value", ex=60)  # Expire in 60 seconds
            test_value = r.get(test_key)
            r.delete(test_key)
            
            response_time = time.time() - start_time
            
            return {
                'healthy': True and ping_result and test_value == b"test_value",
                'version': info.get('redis_version'),
                'uptime_seconds': info.get('uptime_in_seconds'),
                'connected_clients': info.get('connected_clients'),
                'used_memory': info.get('used_memory_human'),
                'used_memory_peak': info.get('used_memory_peak_human'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses'),
                'response_time': response_time,
                'host': host,
                'port': port
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'host': host,
                'port': port
            }
    
    async def check_neo4j_health(self, uri: str = "bolt://localhost:7687", 
                               user: str = "neo4j", password: str = None) -> Dict[str, Any]:
        """Check Neo4j graph database health"""
        if not HAS_NEO4J:
            return {'healthy': False, 'error': 'neo4j library not available'}
        
        try:
            # Try to get password from environment or secrets
            if not password:
                import os
                password = os.environ.get('NEO4J_PASSWORD', 'sutazai')
                
                try:
                    with open('/opt/sutazaiapp/secrets/neo4j_password.txt', 'r') as f:
                        password = f.read().strip()
                except Exception as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
            
            start_time = time.time()
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Test basic connectivity
                result = session.run("RETURN 'Hello, Neo4j!' as message")
                message = result.single()["message"]
                
                # Get database info
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                components = [dict(record) for record in result]
                
                # Get node and relationship counts
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()["node_count"]
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()["rel_count"]
                
                # Get database size
                try:
                    result = session.run("CALL apoc.meta.stats() YIELD nodeCount, relCount, labelCount, propertyKeyCount")
                    stats = dict(result.single()) if result else {}
                except Exception as e:
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    stats = {}
            
            driver.close()
            response_time = time.time() - start_time
            
            return {
                'healthy': True and message == 'Hello, Neo4j!',
                'components': components,
                'node_count': node_count,
                'relationship_count': rel_count,
                'statistics': stats,
                'response_time': response_time,
                'uri': uri,
                'user': user
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'uri': uri,
                'user': user
            }
    
    async def check_http_service(self, name: str, url: str, expected_status: int = 200,
                               timeout: int = 10, headers: Dict = None) -> Dict[str, Any]:
        """Check HTTP-based service health"""
        if not HAS_REQUESTS:
            return {'healthy': False, 'error': 'requests library not available'}
        
        try:
            start_time = time.time()
            
            response = requests.get(url, timeout=timeout, headers=headers or {})
            response_time = time.time() - start_time
            
            # Try to parse JSON response if possible
            try:
                response_data = response.json()
            except Exception as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                response_data = None
            
            return {
                'healthy': response.status_code == expected_status,
                'status_code': response.status_code,
                'response_time': response_time,
                'content_length': len(response.content),
                'headers': dict(response.headers),
                'json_data': response_data,
                'url': url
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'url': url
            }
    
    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all configured services"""
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'services': {}
        }
        
        # Check Ollama
        logger.info("Checking Ollama health...")
        results['services']['ollama'] = await self.check_ollama_health()
        
        # Check databases
        logger.info("Checking PostgreSQL databases...")
        results['services']['postgres_main'] = await self.check_postgres_health(
            host='localhost', port=5432, database='postgres'
        )
        results['services']['postgres_hygiene'] = await self.check_postgres_health(
            host='localhost', port=10020, database='postgres'
        )
        
        # Check Redis
        logger.info("Checking Redis...")
        results['services']['redis_hygiene'] = await self.check_redis_health(
            host='localhost', port=10021
        )
        
        # Check Neo4j
        logger.info("Checking Neo4j...")
        results['services']['neo4j'] = await self.check_neo4j_health(
            uri="bolt://localhost:10003"
        )
        
        # Check HTTP services
        logger.info("Checking HTTP services...")
        http_services = {
            'consul': 'http://localhost:10006/v1/status/leader',
            'kong_admin': 'http://localhost:10007/status',
            'rabbitmq_mgmt': 'http://localhost:10042/api/whoami',
            'grafana_dashboard': 'http://localhost:10050/api/health',
            'loki': 'http://localhost:10202/ready',
            'alertmanager': 'http://localhost:10203/-/healthy',
            'hygiene_backend': 'http://localhost:10420/health',
            'hygiene_dashboard': 'http://localhost:10422/',
            'rule_control_api': 'http://localhost:10421/health',
            'faiss_vector': 'http://localhost:10103/health',
            'hardware_optimizer': 'http://localhost:8116/health',
        }
        
        for service_name, service_url in http_services.items():
            results['services'][service_name] = await self.check_http_service(
                service_name, service_url
            )
        
        # Calculate overall health
        healthy_count = sum(1 for service in results['services'].values() if service.get('healthy', False))
        total_count = len(results['services'])
        results['overall_health'] = {
            'healthy_services': healthy_count,
            'total_services': total_count,
            'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0
        }
        
        return results
    
    def print_health_report(self, results: Dict[str, Any]):
        """Print a formatted health report"""
        print("\n" + "="*80)
        print(f"🏥 SutazAI Service Health Report - {results['timestamp']}")
        print("="*80)
        
        # Overall health
        overall = results['overall_health']
        health_icon = "🟢" if overall['health_percentage'] >= 90 else "🟡" if overall['health_percentage'] >= 70 else "🔴"
        print(f"{health_icon} Overall Health: {overall['health_percentage']:.1f}% ({overall['healthy_services']}/{overall['total_services']} services healthy)")
        
        # Service details
        print("\n📋 Service Details:")
        for service_name, service_data in results['services'].items():
            icon = "✅" if service_data.get('healthy', False) else "❌"
            
            if 'response_time' in service_data:
                rt = service_data['response_time'] * 1000
                rt_str = f"({rt:.1f}ms)"
            else:
                rt_str = ""
            
            error_msg = service_data.get('error', '')
            if error_msg and len(error_msg) > 50:
                error_msg = error_msg[:47] + "..."
            
            status_msg = error_msg if error_msg else "OK"
            print(f"   {icon} {service_name:<25} {rt_str:<10} - {status_msg}")
        
        # Special service details
        print("\n🔍 Detailed Service Information:")
        
        # Ollama details
        if 'ollama' in results['services']:
            ollama = results['services']['ollama']
            if ollama.get('healthy'):
                print(f"   🧠 Ollama: {ollama.get('model_count', 0)} models loaded")
                if ollama.get('models'):
                    for model in ollama['models'][:3]:  # Show first 3 models
                        print(f"      - {model}")
                    if len(ollama['models']) > 3:
                        print(f"      ... and {len(ollama['models']) - 3} more")
        
        # Database details
        for db_name in ['postgres_main', 'postgres_hygiene']:
            if db_name in results['services']:
                db = results['services'][db_name]
                if db.get('healthy'):
                    print(f"   🗄️  {db_name}: {db.get('connection_count', 0)} connections, {db.get('database_size', 'unknown')} size")
        
        # Redis details
        if 'redis_hygiene' in results['services']:
            redis_data = results['services']['redis_hygiene']
            if redis_data.get('healthy'):
                print(f"   🔄 Redis: {redis_data.get('connected_clients', 0)} clients, {redis_data.get('used_memory', 'unknown')} memory")
        
        # Neo4j details
        if 'neo4j' in results['services']:
            neo4j_data = results['services']['neo4j']
            if neo4j_data.get('healthy'):
                print(f"   🕸️  Neo4j: {neo4j_data.get('node_count', 0)} nodes, {neo4j_data.get('relationship_count', 0)} relationships")
        
        print("="*80)
    
    def save_results(self, results: Dict[str, Any]):
        """Save health check results to file"""
        try:
            results_file = Path('/opt/sutazaiapp/logs/service_health_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Health check results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    async def check_specific_service(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        if service_name == 'ollama':
            return await self.check_ollama_health()
        elif service_name in ['postgres', 'postgres_main']:
            return await self.check_postgres_health()
        elif service_name == 'postgres_hygiene':
            return await self.check_postgres_health(port=10020)
        elif service_name in ['redis', 'redis_hygiene']:
            return await self.check_redis_health(port=10021)
        elif service_name == 'neo4j':
            return await self.check_neo4j_health(uri="bolt://localhost:10003")
        else:
            # Try as HTTP service
            service_urls = {
                'consul': 'http://localhost:10006/v1/status/leader',
                'kong': 'http://localhost:10007/status',
                'rabbitmq': 'http://localhost:10042/api/whoami',
                'grafana': 'http://localhost:10050/api/health',
                'loki': 'http://localhost:10202/ready',
                'alertmanager': 'http://localhost:10203/-/healthy',
                'hygiene_backend': 'http://localhost:10420/health',
                'hygiene_dashboard': 'http://localhost:10422/',
                'rule_control_api': 'http://localhost:10421/health',
                'faiss_vector': 'http://localhost:10103/health',
                'hardware_optimizer': 'http://localhost:8116/health',
            }
            
            if service_name in service_urls:
                return await self.check_http_service(service_name, service_urls[service_name])
            else:
                return {'healthy': False, 'error': f'Unknown service: {service_name}'}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SutazAI Service Health Checker')
    parser.add_argument('--service', type=str, help='Check specific service only')
    parser.add_argument('--all', action='store_true', default=True, help='Check all services (default)')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    checker = ServiceHealthChecker()
    
    if args.service:
        logger.info(f"Checking health of service: {args.service}")
        result = await checker.check_specific_service(args.service)
        print(f"\n{args.service} Health Check Result:")
        print(json.dumps(result, indent=2, default=str))
    else:
        logger.info("Checking health of all services...")
        results = await checker.check_all_services()
        checker.print_health_report(results)
        
        if args.save:
            checker.save_results(results)

if __name__ == "__main__":
