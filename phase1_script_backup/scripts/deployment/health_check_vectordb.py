#!/usr/bin/env python3
"""
DEPLOYMENT HEALTH CHECKER
Specialized health checking for deployment scenarios

Consolidated from:
- check_services_health.py
- infrastructure_health_check.py
- health_check_gateway.py
- health_check_ollama.py
- health_check_dataservices.py
- health_check_monitoring.py
- health_check_vectordb.py

Purpose: Validate services are ready for production traffic during deployment
Author: ULTRA SCRIPT CONSOLIDATION MASTER
"""

import sys
import time
import json
import requests
from pathlib import Path

# Import the master health controller from scripts.lib
sys.path.append('/opt/sutazaiapp')
from scripts.lib.master_health_controller import HealthMaster

class DeploymentHealthChecker(HealthMaster):
    """Specialized health checker for deployment scenarios"""
    
    def __init__(self):
        super().__init__()
        
        # Deployment-specific checks with stricter requirements
        self.deployment_checks = {
            'database_connectivity': self.check_database_connectivity,
            'ai_model_availability': self.check_ai_model_availability,
            'service_mesh_connectivity': self.check_service_mesh,
            'monitoring_stack': self.check_monitoring_stack,
            'resource_availability': self.check_resource_availability
        }
    
    def check_database_connectivity(self) -> dict:
        """Verify database connections with actual queries"""
        results = {}
        
        # PostgreSQL connectivity test
        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                port=10000,
                database='sutazai',
                user='sutazai',
                password='sutazai',
                connect_timeout=5
            )
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM information_schema.tables;')
            table_count = cursor.fetchone()[0]
            conn.close()
            
            results['postgres'] = {
                'status': 'healthy',
                'details': f'{table_count} tables accessible'
            }
        except Exception as e:
            results['postgres'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Redis connectivity test
        try:
            import redis
            r = redis.Redis(host='localhost', port=10001, decode_responses=True, socket_timeout=5)
            r.ping()
            info = r.info()
            
            results['redis'] = {
                'status': 'healthy',
                'details': f"Memory: {info.get('used_memory_human', 'unknown')}"
            }
        except Exception as e:
            results['redis'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        return results
    
    def check_ai_model_availability(self) -> dict:
        """Verify AI models are loaded and responding"""
        results = {}
        
        # Check Ollama model availability
        try:
            response = requests.get('http://localhost:10104/api/tags', timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if 'tinyllama:latest' in model_names:
                    results['ollama'] = {
                        'status': 'healthy',
                        'details': f'TinyLlama model loaded, {len(models)} total models'
                    }
                else:
                    results['ollama'] = {
                        'status': 'degraded',
                        'details': f'Models available: {model_names}, but TinyLlama not found'
                    }
            else:
                results['ollama'] = {
                    'status': 'failed',
                    'error': f'HTTP {response.status_code}'
                }
        except Exception as e:
            results['ollama'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        return results
    
    def check_service_mesh(self) -> dict:
        """Check service mesh connectivity"""
        results = {}
        
        # Check RabbitMQ
        try:
            response = requests.get('http://localhost:10007', timeout=5)
            results['rabbitmq'] = {
                'status': 'healthy' if response.status_code == 200 else 'degraded',
                'details': f'HTTP {response.status_code}'
            }
        except Exception as e:
            results['rabbitmq'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        return results
    
    def check_monitoring_stack(self) -> dict:
        """Verify monitoring services are operational"""
        results = {}
        
        monitoring_services = [
            ('prometheus', 'http://localhost:10200/-/healthy'),
            ('grafana', 'http://localhost:10201/api/health'),
            ('loki', 'http://localhost:10202/ready')
        ]
        
        for service, url in monitoring_services:
            try:
                response = requests.get(url, timeout=5)
                results[service] = {
                    'status': 'healthy' if response.status_code == 200 else 'degraded',
                    'details': f'HTTP {response.status_code}'
                }
            except Exception as e:
                results[service] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def check_resource_availability(self) -> dict:
        """Check system resources"""
        import psutil
        
        results = {}
        
        # Memory check
        memory = psutil.virtual_memory()
        results['memory'] = {
            'status': 'healthy' if memory.percent < 85 else 'warning',
            'details': f'{memory.percent:.1f}% used, {memory.available // (1024**3)}GB available'
        }
        
        # Disk check
        disk = psutil.disk_usage('/')
        results['disk'] = {
            'status': 'healthy' if disk.percent < 85 else 'warning',
            'details': f'{disk.percent:.1f}% used, {disk.free // (1024**3)}GB free'
        }
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        results['cpu'] = {
            'status': 'healthy' if cpu_percent < 80 else 'warning',
            'details': f'{cpu_percent:.1f}% usage'
        }
        
        return results
    
    def run_deployment_validation(self) -> dict:
        """Run comprehensive deployment validation"""
        print("Running deployment health validation...")
        
        # Standard service health checks
        service_health = self.check_all_services(critical_only=False)
        
        # Additional deployment-specific checks
        deployment_results = {}
        for check_name, check_func in self.deployment_checks.items():
            print(f"  Checking {check_name}...")
            deployment_results[check_name] = check_func()
        
        # Combine results
        full_results = {
            'service_health': service_health,
            'deployment_checks': deployment_results,
            'timestamp': time.time(),
            'deployment_ready': self.assess_deployment_readiness(service_health, deployment_results)
        }
        
        return full_results
    
    def assess_deployment_readiness(self, service_health: dict, deployment_checks: dict) -> dict:
        """Assess if system is ready for deployment"""
        critical_failures = []
        warnings = []
        
        # Check critical services
        for service_name, result in service_health['results'].items():
            if result.get('critical', False) and result['status'] != 'healthy':
                critical_failures.append(f"Critical service {service_name} is {result['status']}")
        
        # Check deployment-specific validations
        for check_name, check_results in deployment_checks.items():
            for component, result in check_results.items():
                if result['status'] == 'failed':
                    if check_name == 'database_connectivity':
                        critical_failures.append(f"Database connectivity failed for {component}")
                    else:
                        warnings.append(f"{check_name} failed for {component}")
                elif result['status'] in ['warning', 'degraded']:
                    warnings.append(f"{check_name} degraded for {component}")
        
        ready = len(critical_failures) == 0
        
        return {
            'ready': ready,
            'critical_failures': critical_failures,
            'warnings': warnings,
            'recommendation': 'Deploy' if ready else 'Fix critical issues before deployment'
        }


def main():
    """Main entry point for deployment health checking"""
    checker = DeploymentHealthChecker()
    
    # Run comprehensive validation
    results = checker.run_deployment_validation()
    
    # Print results
    readiness = results['deployment_ready']
    print(f"\n{'='*60}")
    print(f"DEPLOYMENT READINESS: {'READY' if readiness['ready'] else 'NOT READY'}")
    print(f"{'='*60}")
    
    if readiness['critical_failures']:
        print("\n🚨 CRITICAL FAILURES:")
        for failure in readiness['critical_failures']:
            print(f"  - {failure}")
    
    if readiness['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in readiness['warnings']:
            print(f"  - {warning}")
    
    print(f"\nRECOMMENDATION: {readiness['recommendation']}")
    
    # Save detailed results
    output_file = f"/opt/sutazaiapp/logs/deployment_health_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if readiness['ready'] else 1)


if __name__ == '__main__':
    main()