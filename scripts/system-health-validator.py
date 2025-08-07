#!/usr/bin/env python3
"""
System Health Validation Script
Validates the complete SutazAI infrastructure deployment
"""

import os
import sys
import time
import json
import requests
import docker
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

def check_docker_containers() -> Tuple[bool, List[str]]:
    """Check status of critical Docker containers"""
    client = docker.from_env()
    critical_containers = ['sutazai-redis', 'sutazai-postgres', 'sutazai-backend']
    issues = []
    
    for container_name in critical_containers:
        try:
            container = client.containers.get(container_name)
            if container.status != 'running':
                issues.append(f"Container {container_name} is not running: {container.status}")
            elif hasattr(container.attrs, 'State') and 'Health' in container.attrs['State']:
                health = container.attrs['State']['Health']['Status']
                if health != 'healthy':
                    issues.append(f"Container {container_name} is unhealthy: {health}")
        except docker.errors.NotFound:
            issues.append(f"Container {container_name} not found")
        except Exception as e:
            issues.append(f"Error checking container {container_name}: {e}")
    
    return len(issues) == 0, issues

def check_service_endpoints() -> Tuple[bool, List[str]]:
    """Check if services are responding on their endpoints"""
    endpoints = [
        ('Redis', 'localhost:10003', 'tcp'),
        ('PostgreSQL', 'localhost:10000', 'tcp'),
        ('Backend API', 'http://localhost:10001/health', 'http'),
        ('Ollama', 'http://localhost:10002/api/tags', 'http')
    ]
    
    issues = []
    
    for name, endpoint, protocol in endpoints:
        try:
            if protocol == 'http':
                response = requests.get(endpoint, timeout=10)
                if response.status_code not in [200, 404]:  # 404 is OK for some endpoints
                    issues.append(f"{name} endpoint {endpoint} returned status {response.status_code}")
            elif protocol == 'tcp':
                import socket
                host, port = endpoint.split(':')
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, int(port)))
                sock.close()
                if result != 0:
                    issues.append(f"{name} TCP endpoint {endpoint} is not accessible")
        except Exception as e:
            issues.append(f"Error checking {name} endpoint {endpoint}: {e}")
    
    return len(issues) == 0, issues

def check_system_resources() -> Tuple[bool, List[str]]:
    """Check system resource usage"""
    import psutil
    
    issues = []
    
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        issues.append(f"High CPU usage: {cpu_percent:.1f}%")
    
    # Check memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        issues.append(f"High memory usage: {memory.percent:.1f}%")
    
    # Check disk usage
    disk = psutil.disk_usage('/')
    disk_percent = (disk.used / disk.total) * 100
    if disk_percent > 90:
        issues.append(f"High disk usage: {disk_percent:.1f}%")
    
    return len(issues) == 0, issues

def check_docker_system() -> Tuple[bool, List[str]]:
    """Check Docker system health"""
    issues = []
    
    try:
        client = docker.from_env()
        info = client.info()
        
        if info['ContainersRunning'] == 0:
            issues.append("No containers are running")
        
        # Check if Docker daemon is responding
        client.ping()
        
    except Exception as e:
        issues.append(f"Docker system check failed: {e}")
    
    return len(issues) == 0, issues

def generate_health_report() -> Dict:
    """Generate comprehensive health report"""
    print("🔍 Running SutazAI System Health Validation...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'overall_status': 'healthy',
        'issues': []
    }
    
    # Check Docker containers
    print("  Checking Docker containers...")
    containers_ok, container_issues = check_docker_containers()
    report['checks']['containers'] = {
        'status': 'pass' if containers_ok else 'fail',
        'issues': container_issues
    }
    if not containers_ok:
        report['overall_status'] = 'unhealthy'
        report['issues'].extend(container_issues)
    
    # Check service endpoints
    print("  Checking service endpoints...")
    endpoints_ok, endpoint_issues = check_service_endpoints()
    report['checks']['endpoints'] = {
        'status': 'pass' if endpoints_ok else 'fail',
        'issues': endpoint_issues
    }
    if not endpoints_ok:
        report['overall_status'] = 'degraded' if report['overall_status'] == 'healthy' else 'unhealthy'
        report['issues'].extend(endpoint_issues)
    
    # Check system resources
    print("  Checking system resources...")
    resources_ok, resource_issues = check_system_resources()
    report['checks']['resources'] = {
        'status': 'pass' if resources_ok else 'fail',
        'issues': resource_issues
    }
    if not resources_ok:
        report['overall_status'] = 'degraded' if report['overall_status'] == 'healthy' else 'unhealthy'
        report['issues'].extend(resource_issues)
    
    # Check Docker system
    print("  Checking Docker system...")
    docker_ok, docker_issues = check_docker_system()
    report['checks']['docker'] = {
        'status': 'pass' if docker_ok else 'fail',
        'issues': docker_issues
    }
    if not docker_ok:
        report['overall_status'] = 'unhealthy'
        report['issues'].extend(docker_issues)
    
    return report

def main():
    """Main function"""
    print("🚀 SutazAI Infrastructure Health Validator")
    print("=" * 50)
    
    # Generate health report
    report = generate_health_report()
    
    # Save report
    report_file = f"/opt/sutazaiapp/logs/health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('/opt/sutazaiapp/logs', exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n📊 Health Check Summary:")
    print("=" * 50)
    
    status_emoji = {
        'healthy': '✅',
        'degraded': '⚠️',
        'unhealthy': '❌'
    }
    
    print(f"Overall Status: {status_emoji.get(report['overall_status'], '❓')} {report['overall_status'].upper()}")
    
    print(f"\nIndividual Checks:")
    for check_name, check_data in report['checks'].items():
        status_symbol = "✅" if check_data['status'] == 'pass' else "❌"
        print(f"  {status_symbol} {check_name.title()}: {check_data['status'].upper()}")
        if check_data['issues']:
            for issue in check_data['issues']:
                print(f"    - {issue}")
    
    if report['issues']:
        print(f"\n⚠️  Issues Found ({len(report['issues'])}):")
        for i, issue in enumerate(report['issues'], 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n🎉 No issues found! System is healthy.")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    if report['overall_status'] == 'healthy':
        print("\n✅ All systems operational!")
        return 0
    elif report['overall_status'] == 'degraded':
        print("\n⚠️  System operational with minor issues")
        return 1
    else:
        print("\n❌ System has critical issues")
        return 2

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Health check failed with error: {e}")
        sys.exit(1)