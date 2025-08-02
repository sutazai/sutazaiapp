#!/usr/bin/env python3
"""
Comprehensive health check validation script
"""

import docker
import time
import json
import requests
from typing import Dict, List, Any
from datetime import datetime

class HealthCheckValidator:
    def __init__(self):
        self.client = docker.from_env()
        self.results = {}
        
    def get_all_containers(self) -> List[Dict[str, Any]]:
        """Get all running containers"""
        containers = []
        for container in self.client.containers.list():
            if 'sutazai' in container.name:
                containers.append({
                    'name': container.name,
                    'id': container.id,
                    'status': container.status,
                    'container': container
                })
        return containers
    
    def check_health_status(self, container) -> Dict[str, Any]:
        """Check health status of a container"""
        try:
            # Reload container to get latest status
            container.reload()
            
            # Get health status
            health = container.attrs.get('State', {}).get('Health', {})
            status = health.get('Status', 'none')
            
            # Get last health check log
            health_log = health.get('Log', [])
            last_check = health_log[-1] if health_log else {}
            
            return {
                'status': status,
                'failing_streak': health.get('FailingStreak', 0),
                'last_check': last_check.get('Start', 'Never'),
                'last_output': last_check.get('Output', ''),
                'exit_code': last_check.get('ExitCode', -1)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def test_health_endpoint(self, container_name: str, port: int = 8080) -> Dict[str, Any]:
        """Test health endpoint directly"""
        try:
            # Try to access health endpoint
            url = f"http://localhost:{port}/health"
            response = requests.get(url, timeout=5)
            
            return {
                'endpoint_accessible': True,
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:200]
            }
        except requests.exceptions.ConnectionError:
            return {
                'endpoint_accessible': False,
                'error': 'Connection refused'
            }
        except requests.exceptions.Timeout:
            return {
                'endpoint_accessible': False,
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'endpoint_accessible': False,
                'error': str(e)
            }
    
    def validate_container_health(self, container_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate health for a single container"""
        container_name = container_info['name']
        container = container_info['container']
        
        print(f"🔍 Checking {container_name}...")
        
        # Check Docker health status
        health_status = self.check_health_status(container)
        
        # For agent containers, test health endpoint
        endpoint_test = {}
        if 'agent' in container_name.lower() or any(term in container_name.lower() for term in ['specialist', 'coordinator', 'manager', 'orchestrator', 'master', 'engineer', 'analyst', 'optimizer', 'creator', 'solver']):
            endpoint_test = self.test_health_endpoint(container_name)
        
        # Check if container has healthcheck configuration
        healthcheck_config = container.attrs.get('Config', {}).get('Healthcheck', {})
        has_healthcheck = bool(healthcheck_config.get('Test'))
        
        result = {
            'container_name': container_name,
            'running': container_info['status'] == 'running',
            'has_healthcheck': has_healthcheck,
            'healthcheck_config': healthcheck_config,
            'health_status': health_status,
            'endpoint_test': endpoint_test,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Determine overall status
        if not result['running']:
            result['overall_status'] = 'not_running'
        elif not result['has_healthcheck']:
            result['overall_status'] = 'no_healthcheck'
        elif health_status['status'] == 'healthy':
            result['overall_status'] = 'healthy'
        elif health_status['status'] == 'unhealthy':
            result['overall_status'] = 'unhealthy'
        elif health_status['status'] == 'starting':
            result['overall_status'] = 'starting'
        else:
            result['overall_status'] = 'unknown'
        
        return result
    
    def validate_all_containers(self) -> Dict[str, Any]:
        """Validate all containers"""
        print("🏥 SutazAI Health Check Validator")
        print("=" * 50)
        
        containers = self.get_all_containers()
        print(f"Found {len(containers)} SutazAI containers")
        print()
        
        results = {}
        
        for container_info in containers:
            result = self.validate_container_health(container_info)
            results[container_info['name']] = result
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report"""
        summary = {
            'total_containers': len(results),
            'healthy': 0,
            'unhealthy': 0,
            'starting': 0,
            'no_healthcheck': 0,
            'not_running': 0,
            'unknown': 0,
            'containers_by_status': {
                'healthy': [],
                'unhealthy': [],
                'starting': [],
                'no_healthcheck': [],
                'not_running': [],
                'unknown': []
            }
        }
        
        for container_name, result in results.items():
            status = result['overall_status']
            summary[status] += 1
            summary['containers_by_status'][status].append(container_name)
        
        return summary
    
    def print_report(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Print formatted report"""
        print("\n" + "=" * 60)
        print("📊 HEALTH CHECK VALIDATION REPORT")
        print("=" * 60)
        
        print(f"Total Containers: {summary['total_containers']}")
        print()
        
        # Status summary
        print("Status Summary:")
        print(f"✅ Healthy:        {summary['healthy']}")
        print(f"❌ Unhealthy:      {summary['unhealthy']}")
        print(f"🔄 Starting:       {summary['starting']}")
        print(f"⚠️  No Healthcheck: {summary['no_healthcheck']}")
        print(f"🔴 Not Running:    {summary['not_running']}")
        print(f"❓ Unknown:        {summary['unknown']}")
        print()
        
        # Detailed issues
        if summary['no_healthcheck']:
            print("🔧 Containers needing health checks:")
            for container in summary['containers_by_status']['no_healthcheck']:
                print(f"   - {container}")
            print()
        
        if summary['unhealthy']:
            print("🚨 Unhealthy containers:")
            for container in summary['containers_by_status']['unhealthy']:
                result = results[container]
                print(f"   - {container}")
                if result['health_status'].get('last_output'):
                    print(f"     Error: {result['health_status']['last_output'][:100]}")
            print()
        
        if summary['not_running']:
            print("💀 Not running containers:")
            for container in summary['containers_by_status']['not_running']:
                print(f"   - {container}")
            print()
        
        # Success rate
        total_should_be_healthy = summary['total_containers'] - summary['not_running']
        if total_should_be_healthy > 0:
            success_rate = (summary['healthy'] / total_should_be_healthy) * 100
            print(f"🎯 Health Check Success Rate: {success_rate:.1f}%")
        
        print()
        print("=" * 60)

def main():
    validator = HealthCheckValidator()
    
    try:
        results = validator.validate_all_containers()
        summary = validator.generate_report(results)
        validator.print_report(results, summary)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/opt/sutazaiapp/reports/health_check_report_{timestamp}.json"
        
        # Create reports directory if it doesn't exist
        import os
        os.makedirs("/opt/sutazaiapp/reports", exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'summary': summary,
                'detailed_results': results
            }, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_file}")
        
        # Return appropriate exit code
        if summary['unhealthy'] > 0 or summary['no_healthcheck'] > 0:
            exit(1)
        else:
            exit(0)
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        exit(2)

if __name__ == "__main__":
    main()