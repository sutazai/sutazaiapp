#!/usr/bin/env python3
"""
Agent Health Validation Script

Focused health check for all SutazAI agent services.
Tests both HTTP endpoints and Docker container status.

Author: COORDINATED ARCHITECT TEAM
Created: 2025-08-11
Purpose: Validate agent services health and identify missing services
"""

import requests
import subprocess
import time
import sys
from typing import Dict, List, Tuple

# Agent services configuration from docker-compose.yml
AGENT_SERVICES = {
    'hardware-resource-optimizer': {
        'port': 11110,
        'container': 'sutazai-hardware-resource-optimizer',
        'critical': True
    },
    'jarvis-hardware-resource-optimizer': {
        'port': 11104,
        'container': 'sutazai-jarvis-hardware-resource-optimizer',
        'critical': False
    },
    'jarvis-automation-agent': {
        'port': 11102,
        'container': 'sutazai-jarvis-automation-agent',
        'critical': False
    },
    'ollama-integration': {
        'port': 8090,
        'container': 'sutazai-ollama-integration',
        'critical': True
    },
    'ai-agent-orchestrator': {
        'port': 8589,
        'container': 'sutazai-ai-agent-orchestrator',
        'critical': True
    },
    'task-assignment-coordinator': {
        'port': 8551,
        'container': 'sutazai-task-assignment-coordinator',
        'critical': True
    },
    'resource-arbitration-agent': {
        'port': 8588,
        'container': 'sutazai-resource-arbitration-agent',
        'critical': True
    }
}


def check_container_running(container_name: str) -> Tuple[bool, str]:
    """Check if Docker container is running."""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and container_name in result.stdout:
            return True, "Container running"
        else:
            return False, "Container not running or not found"
            
    except subprocess.TimeoutExpired:
        return False, "Docker command timeout"
    except Exception as e:
        return False, f"Error checking container: {str(e)}"


def check_health_endpoint(port: int) -> Tuple[bool, str, float]:
    """Check agent health endpoint."""
    start_time = time.time()
    
    try:
        response = requests.get(f'http://localhost:{port}/health', timeout=10)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if response.status_code == 200:
            return True, f"HTTP 200 ({response_time:.0f}ms)", response_time
        else:
            return False, f"HTTP {response.status_code} ({response_time:.0f}ms)", response_time
            
    except requests.exceptions.Timeout:
        response_time = (time.time() - start_time) * 1000
        return False, f"Timeout after 10s ({response_time:.0f}ms)", response_time
    except requests.exceptions.ConnectionError:
        response_time = (time.time() - start_time) * 1000
        return False, f"Connection refused ({response_time:.0f}ms)", response_time
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return False, f"Error: {str(e)} ({response_time:.0f}ms)", response_time


def validate_all_agents() -> Dict:
    """Validate all agent services."""
    results = {
        'healthy': [],
        'unhealthy': [],
        'missing': [],
        'summary': {
            'total': 0,
            'healthy_count': 0,
            'unhealthy_count': 0,
            'missing_count': 0,
            'critical_failures': 0
        }
    }
    
    print("🏥 SutazAI Agent Health Validation")
    print("=" * 60)
    print("")
    
    for agent_name, config in AGENT_SERVICES.items():
        results['summary']['total'] += 1
        port = config['port']
        container = config['container']
        is_critical = config['critical']
        
        print(f"🔍 Checking {agent_name}...")
        print(f"   Container: {container}")
        print(f"   Port: {port}")
        print(f"   Critical: {'Yes' if is_critical else 'No'}")
        
        # Check container status
        container_running, container_msg = check_container_running(container)
        
        # Check health endpoint
        endpoint_healthy, endpoint_msg, response_time = check_health_endpoint(port)
        
        # Determine overall status
        if container_running and endpoint_healthy:
            status = "✅ HEALTHY"
            results['healthy'].append({
                'name': agent_name,
                'container': container,
                'port': port,
                'response_time': response_time,
                'critical': is_critical
            })
            results['summary']['healthy_count'] += 1
        elif container_running and not endpoint_healthy:
            status = "⚠️ DEGRADED"
            results['unhealthy'].append({
                'name': agent_name,
                'container': container,
                'port': port,
                'issue': f"Container running but {endpoint_msg}",
                'critical': is_critical
            })
            results['summary']['unhealthy_count'] += 1
            if is_critical:
                results['summary']['critical_failures'] += 1
        else:
            status = "❌ MISSING"
            results['missing'].append({
                'name': agent_name,
                'container': container,
                'port': port,
                'issue': container_msg,
                'critical': is_critical
            })
            results['summary']['missing_count'] += 1
            if is_critical:
                results['summary']['critical_failures'] += 1
        
        print(f"   Status: {status}")
        if container_running:
            print(f"   Container: ✅ {container_msg}")
        else:
            print(f"   Container: ❌ {container_msg}")
        
        if endpoint_healthy:
            print(f"   Health Endpoint: ✅ {endpoint_msg}")
        else:
            print(f"   Health Endpoint: ❌ {endpoint_msg}")
        
        print("")
    
    return results


def print_summary(results: Dict) -> int:
    """Print validation summary and return exit code."""
    summary = results['summary']
    
    print("=" * 60)
    print("📊 AGENT VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"Total Agents: {summary['total']}")
    print(f"✅ Healthy: {summary['healthy_count']}")
    print(f"⚠️ Degraded: {summary['unhealthy_count']}")
    print(f"❌ Missing: {summary['missing_count']}")
    print(f"🚨 Critical Failures: {summary['critical_failures']}")
    
    health_percentage = (summary['healthy_count'] * 100) // summary['total']
    print(f"📈 Agent Health Score: {health_percentage}%")
    print("")
    
    # Detailed breakdown
    if results['healthy']:
        print("✅ HEALTHY AGENTS:")
        for agent in results['healthy']:
            critical_marker = " [CRITICAL]" if agent['critical'] else ""
            print(f"   • {agent['name']}{critical_marker} - {agent['response_time']:.0f}ms")
        print("")
    
    if results['unhealthy']:
        print("⚠️ DEGRADED AGENTS:")
        for agent in results['unhealthy']:
            critical_marker = " [CRITICAL]" if agent['critical'] else ""
            print(f"   • {agent['name']}{critical_marker} - {agent['issue']}")
        print("")
    
    if results['missing']:
        print("❌ MISSING AGENTS:")
        for agent in results['missing']:
            critical_marker = " [CRITICAL]" if agent['critical'] else ""
            print(f"   • {agent['name']}{critical_marker} - {agent['issue']}")
            print(f"     Container: {agent['container']}")
            print(f"     Expected Port: {agent['port']}")
        print("")
    
    # Recommendations
    if results['missing'] or results['unhealthy']:
        print("🔧 RECOMMENDATIONS:")
        
        if results['missing']:
            print("   Missing agents can be started with:")
            for agent in results['missing']:
                print(f"     docker-compose up -d {agent['container'].replace('sutazai-', '')}")
        
        if results['unhealthy']:
            print("   Degraded agents may need restart:")
            for agent in results['unhealthy']:
                print(f"     docker restart {agent['container']}")
        print("")
    
    # Determine exit code
    if summary['critical_failures'] > 0:
        print("🚨 CRITICAL FAILURES DETECTED - System degraded")
        return 2
    elif summary['unhealthy_count'] > 0 or summary['missing_count'] > 0:
        print("⚠️ NON-CRITICAL ISSUES DETECTED - System operational with warnings")
        return 1
    else:
        print("🎉 ALL AGENTS HEALTHY - System fully operational")
        return 0


def main():
    """Main validation function."""
    results = validate_all_agents()
    exit_code = print_summary(results)
    return exit_code


if __name__ == '__main__':
    exit(main())