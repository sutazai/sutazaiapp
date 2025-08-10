#!/usr/bin/env python3
"""
Network and Port Validation Test Suite
"""

import json
import socket
import subprocess
import time
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkValidator:
    """Network validation and port conflict detection"""
    
    def __init__(self):
        self.expected_ports = {
            5432: 'PostgreSQL',
            6379: 'Redis', 
            7474: 'Neo4j HTTP',
            7687: 'Neo4j Bolt',
            8001: 'ChromaDB'
        }
        
        self.test_results = {
            'timestamp': time.time(),
            'port_availability': {},
            'port_conflicts': [],
            'network_connectivity': {},
            'docker_networking': {},
            'overall_status': 'pending'
        }
    
    def check_port_availability(self) -> Dict[str, Any]:
        """Check if expected ports are available and accessible"""
        logger.info("Checking port availability...")
        
        results = {}
        
        for port, service in self.expected_ports.items():
            try:
                # Check if port is listening
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    # Port is open, test connection
                    response_time = self.measure_connection_time('localhost', port)
                    results[port] = {
                        'service': service,
                        'status': 'open',
                        'accessible': True,
                        'response_time_ms': response_time,
                        'message': f'{service} port {port} is accessible'
                    }
                else:
                    results[port] = {
                        'service': service,
                        'status': 'closed',
                        'accessible': False,
                        'response_time_ms': None,
                        'message': f'{service} port {port} is not accessible'
                    }
                    
            except Exception as e:
                results[port] = {
                    'service': service,
                    'status': 'error',
                    'accessible': False,
                    'response_time_ms': None,
                    'message': f'Error testing {service} port {port}: {str(e)}'
                }
        
        self.test_results['port_availability'] = results
        return results
    
    def measure_connection_time(self, host: str, port: int) -> float:
        """Measure connection response time"""
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((host, port))
            sock.close()
            end_time = time.time()
            return round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
        except (AssertionError, Exception) as e:
            logger.warning(f"Exception caught, returning: {e}")
            return None
    
    def detect_port_conflicts(self) -> List[Dict[str, Any]]:
        """Detect potential port conflicts"""
        logger.info("Detecting port conflicts...")
        
        conflicts = []
        
        try:
            # Get list of all listening ports
            result = subprocess.run(['netstat', '-tulpn'], capture_output=True, text=True)
            if result.returncode != 0:
                # Try ss if netstat is not available
                result = subprocess.run(['ss', '-tulpn'], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                port_usage = {}
                
                for line in lines:
                    if ':' in line and 'LISTEN' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            address_port = parts[3]
                            if ':' in address_port:
                                port_str = address_port.split(':')[-1]
                                try:
                                    port = int(port_str)
                                    if port in port_usage:
                                        conflicts.append({
                                            'port': port,
                                            'services': [port_usage[port], line.strip()],
                                            'severity': 'high',
                                            'message': f'Port {port} has multiple listeners'
                                        })
                                    else:
                                        port_usage[port] = line.strip()
                                except ValueError:
                                    continue
                                    
        except Exception as e:
            conflicts.append({
                'port': None,
                'services': [],
                'severity': 'medium',
                'message': f'Could not check for port conflicts: {str(e)}'
            })
        
        self.test_results['port_conflicts'] = conflicts
        return conflicts
    
    def test_docker_networking(self) -> Dict[str, Any]:
        """Test Docker network configuration"""
        logger.info("Testing Docker networking...")
        
        results = {}
        
        try:
            # Check Docker networks
            result = subprocess.run(['docker', 'network', 'ls'], capture_output=True, text=True)
            if result.returncode == 0:
                networks = []
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        networks.append({
                            'id': parts[0],
                            'name': parts[1],
                            'driver': parts[2]
                        })
                
                results['networks'] = networks
                results['network_count'] = len(networks)
            
            # Check container network connectivity
            containers = ['sutazai-postgres', 'sutazai-redis', 'sutazai-neo4j', 'sutazai-chromadb']
            container_connectivity = {}
            
            for container in containers:
                try:
                    # Check if container is running and get its IP
                    result = subprocess.run([
                        'docker', 'inspect', container, 
                        '--format={{.NetworkSettings.IPAddress}}'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        ip = result.stdout.strip()
                        container_connectivity[container] = {
                            'ip': ip,
                            'status': 'running',
                            'network_accessible': True
                        }
                    else:
                        container_connectivity[container] = {
                            'ip': None,
                            'status': 'not_running',
                            'network_accessible': False
                        }
                        
                except Exception as e:
                    container_connectivity[container] = {
                        'ip': None,
                        'status': 'error',
                        'network_accessible': False,
                        'error': str(e)
                    }
            
            results['container_connectivity'] = container_connectivity
            
        except Exception as e:
            results['error'] = str(e)
            results['status'] = 'failed'
        
        self.test_results['docker_networking'] = results
        return results
    
    def test_service_endpoints(self) -> Dict[str, Any]:
        """Test service-specific endpoints"""
        logger.info("Testing service endpoints...")
        
        endpoints = {
            'postgres': {'host': 'localhost', 'port': 5432, 'type': 'tcp'},
            'redis': {'host': 'localhost', 'port': 6379, 'type': 'tcp'},
            'neo4j_http': {'host': 'localhost', 'port': 7474, 'type': 'http', 'path': '/'},
            'neo4j_bolt': {'host': 'localhost', 'port': 7687, 'type': 'tcp'},
            'chromadb': {'host': 'localhost', 'port': 8001, 'type': 'http', 'path': '/api/v1/heartbeat'}
        }
        
        results = {}
        
        for service, config in endpoints.items():
            try:
                if config['type'] == 'tcp':
                    # TCP connection test
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    start_time = time.time()
                    result = sock.connect_ex((config['host'], config['port']))
                    end_time = time.time()
                    sock.close()
                    
                    if result == 0:
                        results[service] = {
                            'status': 'accessible',
                            'response_time_ms': round((end_time - start_time) * 1000, 2),
                            'endpoint': f"{config['host']}:{config['port']}"
                        }
                    else:
                        results[service] = {
                            'status': 'not_accessible',
                            'error': f'Connection failed with code {result}',
                            'endpoint': f"{config['host']}:{config['port']}"
                        }
                        
                elif config['type'] == 'http':
                    # HTTP endpoint test
                    import requests
                    url = f"http://{config['host']}:{config['port']}{config.get('path', '/')}"
                    
                    start_time = time.time()
                    response = requests.get(url, timeout=5)
                    end_time = time.time()
                    
                    results[service] = {
                        'status': 'accessible' if response.status_code == 200 else 'error',
                        'status_code': response.status_code,
                        'response_time_ms': round((end_time - start_time) * 1000, 2),
                        'endpoint': url
                    }
                    
            except Exception as e:
                results[service] = {
                    'status': 'error',
                    'error': str(e),
                    'endpoint': f"{config['host']}:{config['port']}"
                }
        
        self.test_results['network_connectivity'] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all network validation tests"""
        logger.info("Starting comprehensive network validation...")
        
        # Run all tests
        self.check_port_availability()
        self.detect_port_conflicts()
        self.test_docker_networking()
        self.test_service_endpoints()
        
        # Determine overall status
        port_issues = len([p for p in self.test_results['port_availability'].values() 
                          if not p['accessible']])
        conflicts = len(self.test_results['port_conflicts'])
        connectivity_issues = len([s for s in self.test_results['network_connectivity'].values() 
                                 if s['status'] != 'accessible'])
        
        if port_issues == 0 and conflicts == 0 and connectivity_issues == 0:
            self.test_results['overall_status'] = 'passed'
        elif port_issues > 0 or connectivity_issues > 0:
            self.test_results['overall_status'] = 'failed'
        else:
            self.test_results['overall_status'] = 'warning'
        
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate comprehensive network validation report"""
        report = []
        report.append("=" * 80)
        report.append("SUTAZAI NETWORK VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.ctime(self.test_results['timestamp'])}")
        report.append(f"Overall Status: {self.test_results['overall_status'].upper()}")
        report.append("")
        
        # Port Availability
        report.append("PORT AVAILABILITY:")
        report.append("-" * 40)
        for port, info in self.test_results['port_availability'].items():
            status_symbol = "✓" if info['accessible'] else "✗"
            report.append(f"  {status_symbol} Port {port} ({info['service']}): {info['message']}")
            if info.get('response_time_ms'):
                report.append(f"    Response time: {info['response_time_ms']}ms")
        report.append("")
        
        # Port Conflicts
        report.append("PORT CONFLICTS:")
        report.append("-" * 40)
        if self.test_results['port_conflicts']:
            for conflict in self.test_results['port_conflicts']:
                report.append(f"  ⚠ {conflict['message']}")
        else:
            report.append("  ✓ No port conflicts detected")
        report.append("")
        
        # Network Connectivity
        report.append("SERVICE CONNECTIVITY:")
        report.append("-" * 40)
        for service, info in self.test_results['network_connectivity'].items():
            status_symbol = "✓" if info['status'] == 'accessible' else "✗"
            report.append(f"  {status_symbol} {service}: {info['status']}")
            if info.get('response_time_ms'):
                report.append(f"    Response time: {info['response_time_ms']}ms")
            if info.get('error'):
                report.append(f"    Error: {info['error']}")
        report.append("")
        
        # Docker Networking
        report.append("DOCKER NETWORKING:")
        report.append("-" * 40)
        docker_info = self.test_results['docker_networking']
        if 'networks' in docker_info:
            report.append(f"  Networks found: {docker_info['network_count']}")
            for network in docker_info['networks']:
                report.append(f"    - {network['name']} ({network['driver']})")
        
        if 'container_connectivity' in docker_info:
            report.append("  Container connectivity:")
            for container, info in docker_info['container_connectivity'].items():
                status_symbol = "✓" if info['network_accessible'] else "✗"
                report.append(f"    {status_symbol} {container}: {info['status']}")
                if info.get('ip'):
                    report.append(f"      IP: {info['ip']}")
        report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    validator = NetworkValidator()
    
    try:
        # Run all validation tests
        results = validator.run_all_tests()
        
        # Generate and display report
        report = validator.generate_report()
        print(report)
        
        # Save results
        import os
        os.makedirs('/opt/sutazaiapp/backend/tests/reports', exist_ok=True)
        
        # Save JSON results
        with open('/opt/sutazaiapp/backend/tests/reports/network_validation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save text report
        with open('/opt/sutazaiapp/backend/tests/reports/network_validation_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Network validation results saved to /opt/sutazaiapp/backend/tests/reports/")
        
        # Exit with appropriate code
        if results['overall_status'] == 'passed':
            logger.info("All network validation tests passed!")
            return 0
        elif results['overall_status'] == 'warning':
            logger.warning("Network validation completed with warnings!")
            return 0
        else:
            logger.error("Network validation failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Network validation execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())