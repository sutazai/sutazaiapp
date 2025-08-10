#!/usr/bin/env python3
"""
Network Utilities for SutazAI
=============================

Consolidated network validation and connectivity utilities.
"""

import socket
import requests
import subprocess
import time
import ipaddress
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .common_utils import setup_logging

logger = setup_logging('network_utils')

class NetworkValidator:
    """Network validation and connectivity testing"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def check_port(self, host: str, port: int) -> Tuple[bool, str]:
        """Check if a port is open"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                result = sock.connect_ex((host, port))
                if result == 0:
                    return True, "Open"
                else:
                    return False, "Closed"
        except socket.gaierror:
            return False, "Host not found"
        except Exception as e:
            return False, str(e)
    
    def check_multiple_ports(self, host: str, ports: List[int]) -> Dict[int, Dict[str, Union[bool, str]]]:
        """Check multiple ports concurrently"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(ports), 20)) as executor:
            future_to_port = {
                executor.submit(self.check_port, host, port): port
                for port in ports
            }
            
            for future in as_completed(future_to_port):
                port = future_to_port[future]
                try:
                    is_open, message = future.result()
                    results[port] = {'open': is_open, 'message': message}
                except Exception as e:
                    results[port] = {'open': False, 'message': str(e)}
        
        return results
    
    def test_http_endpoint(self, url: str, expected_status: int = 200) -> Dict[str, Union[bool, str, int, float]]:
        """Test HTTP endpoint connectivity"""
        result = {
            'url': url,
            'accessible': False,
            'status_code': None,
            'response_time': None,
            'error': None
        }
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=self.timeout)
            response_time = time.time() - start_time
            
            result.update({
                'accessible': response.status_code == expected_status,
                'status_code': response.status_code,
                'response_time': round(response_time * 1000, 2)  # ms
            })
            
        except requests.exceptions.RequestException as e:
            result['error'] = str(e)
        
        return result
    
    def scan_port_range(self, host: str, start_port: int, end_port: int) -> List[int]:
        """Scan range of ports and return open ones"""
        open_ports = []
        ports_to_check = list(range(start_port, end_port + 1))
        
        results = self.check_multiple_ports(host, ports_to_check)
        
        for port, result in results.items():
            if result['open']:
                open_ports.append(port)
        
        return sorted(open_ports)
    
    def ping_host(self, host: str, count: int = 3) -> Dict[str, Union[bool, float, str]]:
        """Ping a host and return results"""
        result = {
            'host': host,
            'reachable': False,
            'avg_response_time': None,
            'packet_loss': None,
            'error': None
        }
        
        try:
            # Use system ping command
            cmd = ['ping', '-c', str(count), host]
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.timeout * count
            )
            
            if process.returncode == 0:
                output = process.stdout
                
                # Parse response time
                import re
                time_matches = re.findall(r'time=(\d+\.?\d*)', output)
                if time_matches:
                    times = [float(t) for t in time_matches]
                    result['avg_response_time'] = round(sum(times) / len(times), 2)
                
                # Parse packet loss
                loss_match = re.search(r'(\d+)% packet loss', output)
                if loss_match:
                    result['packet_loss'] = int(loss_match.group(1))
                
                result['reachable'] = True
            else:
                result['error'] = process.stderr.strip()
                
        except subprocess.TimeoutExpired:
            result['error'] = 'Ping timeout'
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def validate_network_configuration(self, config: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Validate network configuration with multiple services"""
        results = {}
        
        for service_name, endpoints in config.items():
            service_results = []
            
            for endpoint in endpoints:
                url = endpoint.get('url')
                host = endpoint.get('host')
                port = endpoint.get('port')
                
                if url:
                    # HTTP endpoint test
                    test_result = self.test_http_endpoint(url)
                    test_result['service'] = service_name
                    test_result['endpoint_type'] = 'http'
                    service_results.append(test_result)
                    
                elif host and port:
                    # Port connectivity test
                    is_open, message = self.check_port(host, port)
                    test_result = {
                        'service': service_name,
                        'endpoint_type': 'tcp',
                        'host': host,
                        'port': port,
                        'accessible': is_open,
                        'message': message
                    }
                    service_results.append(test_result)
            
            results[service_name] = service_results
        
        return results

def check_port_availability(ports: List[int], host: str = 'localhost') -> Dict[int, bool]:
    """Check if ports are available (not in use)"""
    validator = NetworkValidator()
    results = validator.check_multiple_ports(host, ports)
    
    # Return availability (True if port is closed/available)
    return {port: not result['open'] for port, result in results.items()}

def test_service_connectivity(services: Dict[str, str]) -> Dict[str, Dict]:
    """Test connectivity to multiple services"""
    validator = NetworkValidator()
    results = {}
    
    for service_name, url in services.items():
        results[service_name] = validator.test_http_endpoint(url)
    
    return results

def get_local_ip_addresses() -> List[str]:
    """Get all local IP addresses"""
    import netifaces
    
    addresses = []
    
    for interface in netifaces.interfaces():
        try:
            interface_addresses = netifaces.ifaddresses(interface)
            
            # Get IPv4 addresses
            if netifaces.AF_INET in interface_addresses:
                for addr_info in interface_addresses[netifaces.AF_INET]:
                    ip = addr_info.get('addr')
                    if ip and ip != '127.0.0.1':
                        addresses.append(ip)
        except Exception:
            continue
    
    return addresses

def validate_sutazai_services() -> Dict[str, Dict]:
    """Validate all SutazAI service endpoints"""
    services = {
        'backend': 'http://localhost:10010/health',
        'frontend': 'http://localhost:10011/',
        'ollama': 'http://localhost:10104/api/tags',
        'postgres': 'localhost:10000',
        'redis': 'localhost:10001',
        'neo4j_http': 'http://localhost:10002',
        'neo4j_bolt': 'localhost:10003',
        'rabbitmq_management': 'http://localhost:10008',
        'grafana': 'http://localhost:10201',
        'prometheus': 'http://localhost:10200',
        'hardware_optimizer': 'http://localhost:11110/health',
        'ai_orchestrator': 'http://localhost:8589/health'
    }
    
    validator = NetworkValidator()
    results = {}
    
    for service_name, endpoint in services.items():
        if endpoint.startswith('http'):
            # HTTP service
            result = validator.test_http_endpoint(endpoint)
        else:
            # TCP service
            host, port = endpoint.split(':')
            is_open, message = validator.check_port(host, int(port))
            result = {
                'accessible': is_open,
                'message': message,
                'endpoint_type': 'tcp'
            }
        
        results[service_name] = result
    
    return results

def discover_services_on_network(network: str = '192.168.1.0/24', ports: List[int] = None) -> Dict[str, List[int]]:
    """Discover services on network by scanning common ports"""
    if ports is None:
        ports = [22, 80, 443, 3000, 5000, 8000, 8080, 9000]
    
    try:
        network_obj = ipaddress.ip_network(network, strict=False)
    except ValueError as e:
        logger.error(f"Invalid network: {network}: {e}")
        return {}
    
    discovered = {}
    validator = NetworkValidator(timeout=1)  # Fast scan
    
    for ip in network_obj.hosts():
        ip_str = str(ip)
        
        # Quick ping test first
        ping_result = validator.ping_host(ip_str, count=1)
        if not ping_result['reachable']:
            continue
        
        # Scan ports
        open_ports = []
        port_results = validator.check_multiple_ports(ip_str, ports)
        
        for port, result in port_results.items():
            if result['open']:
                open_ports.append(port)
        
        if open_ports:
            discovered[ip_str] = open_ports
            logger.info(f"Found services on {ip_str}: ports {open_ports}")
    
    return discovered

def test_network_performance(target_host: str = 'google.com') -> Dict[str, Union[float, bool]]:
    """Test basic network performance metrics"""
    validator = NetworkValidator()
    
    # Test ping
    ping_result = validator.ping_host(target_host, count=10)
    
    # Test HTTP performance
    http_result = validator.test_http_endpoint(f'http://{target_host}')
    
    return {
        'ping_reachable': ping_result['reachable'],
        'ping_avg_time': ping_result['avg_response_time'],
        'ping_packet_loss': ping_result['packet_loss'],
        'http_accessible': http_result['accessible'],
        'http_response_time': http_result['response_time']
    }

if __name__ == "__main__":
    # Test network utilities
    logger.info("Testing network utilities...")
    
    # Test SutazAI services
    logger.info("Checking SutazAI services...")
    service_results = validate_sutazai_services()
    
    accessible_services = [name for name, result in service_results.items() 
                          if result.get('accessible', False)]
    logger.info(f"Accessible services: {len(accessible_services)}/{len(service_results)}")
    
    for service, result in service_results.items():
        status = "✓" if result.get('accessible', False) else "✗"
        logger.info(f"  {status} {service}: {result.get('message', 'N/A')}")
    
    # Test port availability
    logger.info("\nChecking port availability...")
    test_ports = [10010, 10011, 10104, 8589]
    port_availability = check_port_availability(test_ports)
    
    for port, available in port_availability.items():
        status = "Available" if available else "In use"
        logger.info(f"  Port {port}: {status}")
    
    logger.info("Network utilities test completed")