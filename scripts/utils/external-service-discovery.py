#!/usr/bin/env python3
"""
Purpose: Discovers and catalogs external services for SutazAI integration
Usage: python external-service-discovery.py [--scan-network] [--output json|yaml]
Requirements: docker, psutil, requests
"""

import json
import yaml
import socket
import subprocess
import argparse
import sys
from typing import Dict, List, Any
import psutil
import docker
import requests
from datetime import datetime

class ExternalServiceDiscovery:
    """Discovers external services running on the system"""
    
    # Common service signatures
    SERVICE_SIGNATURES = {
        5432: {'name': 'postgresql', 'type': 'database'},
        3306: {'name': 'mysql', 'type': 'database'},
        27017: {'name': 'mongodb', 'type': 'database'},
        6379: {'name': 'redis', 'type': 'cache'},
        11211: {'name': 'memcached', 'type': 'cache'},
        5672: {'name': 'rabbitmq', 'type': 'message_queue'},
        15672: {'name': 'rabbitmq_management', 'type': 'message_queue'},
        9092: {'name': 'kafka', 'type': 'message_queue'},
        9200: {'name': 'elasticsearch', 'type': 'search'},
        9300: {'name': 'elasticsearch_transport', 'type': 'search'},
        9090: {'name': 'prometheus', 'type': 'monitoring'},
        3000: {'name': 'grafana', 'type': 'monitoring'},
        8086: {'name': 'influxdb', 'type': 'timeseries_db'},
        2181: {'name': 'zookeeper', 'type': 'coordination'},
        8500: {'name': 'consul', 'type': 'service_discovery'},
        2375: {'name': 'docker', 'type': 'container_runtime'},
        2376: {'name': 'docker_tls', 'type': 'container_runtime'},
        80: {'name': 'http', 'type': 'web'},
        443: {'name': 'https', 'type': 'web'},
        8080: {'name': 'http_alt', 'type': 'web'},
        8443: {'name': 'https_alt', 'type': 'web'}
    }
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.discovered_services = []
        
    def scan_docker_containers(self) -> List[Dict[str, Any]]:
        """Scan all Docker containers"""
        containers = []
        
        try:
            for container in self.docker_client.containers.list(all=True):
                # Skip SutazAI containers
                if 'sutazai' in container.name.lower():
                    continue
                    
                container_info = {
                    'name': container.name,
                    'id': container.short_id,
                    'image': container.image.tags[0] if container.image.tags else container.image.short_id,
                    'status': container.status,
                    'ports': self._parse_container_ports(container),
                    'labels': container.labels,
                    'networks': list(container.attrs['NetworkSettings']['Networks'].keys()),
                    'environment': self._safe_env_vars(container)
                }
                
                # Try to identify service type
                container_info['service_type'] = self._identify_service_type(container_info)
                
                containers.append(container_info)
                
        except Exception as e:
            print(f"Error scanning Docker containers: {e}")
            
        return containers
    
    def scan_host_services(self) -> List[Dict[str, Any]]:
        """Scan services running directly on host"""
        services = []
        
        # Get all listening ports
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'LISTEN':
                port = conn.laddr.port
                
                # Skip high ports and SutazAI range
                if port > 10000:
                    continue
                    
                service_info = {
                    'port': port,
                    'address': conn.laddr.ip,
                    'pid': conn.pid,
                    'process': self._get_process_info(conn.pid) if conn.pid else None
                }
                
                # Check if it's a known service
                if port in self.SERVICE_SIGNATURES:
                    service_info.update(self.SERVICE_SIGNATURES[port])
                    
                # Try to probe the service
                if self._is_port_open('localhost', port):
                    service_info['probe_result'] = self._probe_service('localhost', port)
                    
                services.append(service_info)
                
        return services
    
    def scan_network_services(self, network_range: str = "172.16.0.0/12") -> List[Dict[str, Any]]:
        """Scan network for services (optional, requires nmap)"""
        services = []
        
        try:
            # Use nmap for network discovery if available
            result = subprocess.run(
                ['nmap', '-sV', '-p-', '--open', network_range],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Parse nmap output
                services = self._parse_nmap_output(result.stdout)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Network scan skipped (nmap not available or timeout)")
            
        return services
    
    def _parse_container_ports(self, container) -> Dict[str, Any]:
        """Parse container port mappings"""
        ports = {}
        
        for container_port, host_mappings in container.ports.items():
            if host_mappings:
                for mapping in host_mappings:
                    ports[container_port] = {
                        'host_ip': mapping['HostIp'],
                        'host_port': mapping['HostPort']
                    }
                    
        return ports
    
    def _safe_env_vars(self, container) -> Dict[str, str]:
        """Get environment variables, hiding sensitive values"""
        env_vars = {}
        
        try:
            for env in container.attrs['Config']['Env']:
                key, value = env.split('=', 1)
                
                # Hide sensitive values
                if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                    env_vars[key] = '***'
                else:
                    env_vars[key] = value
                    
        except Exception:
            pass
            
        return env_vars
    
    def _identify_service_type(self, container_info: Dict[str, Any]) -> str:
        """Identify service type from container info"""
        image = container_info['image'].lower()
        
        # Database services
        if any(db in image for db in ['postgres', 'mysql', 'mariadb', 'mongo', 'cassandra', 'redis']):
            return 'database'
            
        # Message queues
        if any(mq in image for mq in ['rabbitmq', 'kafka', 'activemq', 'nats']):
            return 'message_queue'
            
        # Web services
        if any(web in image for web in ['nginx', 'apache', 'httpd', 'caddy']):
            return 'web_server'
            
        # Monitoring
        if any(mon in image for mon in ['prometheus', 'grafana', 'elastic', 'kibana']):
            return 'monitoring'
            
        return 'unknown'
    
    def _get_process_info(self, pid: int) -> Dict[str, Any]:
        """Get process information"""
        try:
            proc = psutil.Process(pid)
            return {
                'name': proc.name(),
                'cmdline': ' '.join(proc.cmdline()),
                'create_time': datetime.fromtimestamp(proc.create_time()).isoformat()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def _is_port_open(self, host: str, port: int) -> bool:
        """Check if port is open"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    
    def _probe_service(self, host: str, port: int) -> Dict[str, Any]:
        """Probe service for identification"""
        probe_result = {'responsive': False}
        
        # Try HTTP probe
        if port in [80, 443, 8080, 8443, 3000, 9090, 9200]:
            try:
                protocol = 'https' if port in [443, 8443] else 'http'
                # SECURITY FIX: Enable TLS verification for production
                # For development with self-signed certs, consider using custom cert bundle
                try:
                    response = requests.get(f'{protocol}://{host}:{port}', timeout=2, verify=True)
                except requests.exceptions.SSLError:
                    # Fallback for local development with self-signed certs
                    # In production, use proper certificates
                    response = requests.get(f'{protocol}://{host}:{port}', timeout=2, verify=False)
                probe_result['responsive'] = True
                probe_result['http_status'] = response.status_code
                probe_result['headers'] = dict(response.headers)
            except Exception:
                pass
                
        return probe_result
    
    def _parse_nmap_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse nmap output (simplified)"""
        services = []
        # Basic parsing - would need more sophisticated parsing for production
        return services
    
    def generate_integration_config(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate integration configuration for discovered services"""
        config = {
            'version': '1.0',
            'discovered_at': datetime.now().isoformat(),
            'services': [],
            'integration_plan': []
        }
        
        # Process each discovered service
        for idx, service in enumerate(services):
            # Assign SutazAI proxy port
            proxy_port = 10200 + idx
            
            integration = {
                'original_service': service,
                'sutazai_integration': {
                    'proxy_port': proxy_port,
                    'adapter_name': f"sutazai-{service.get('name', 'unknown')}-adapter",
                    'health_check_endpoint': f"http://localhost:{proxy_port}/health",
                    'monitoring_enabled': True
                }
            }
            
            config['services'].append(service)
            config['integration_plan'].append(integration)
            
        return config
    
    def run(self, scan_network: bool = False, output_format: str = 'json') -> None:
        """Run the discovery process"""
        print("Starting external service discovery...")
        
        # Scan Docker containers
        print("Scanning Docker containers...")
        docker_services = self.scan_docker_containers()
        print(f"Found {len(docker_services)} non-SutazAI containers")
        
        # Scan host services
        print("Scanning host services...")
        host_services = self.scan_host_services()
        print(f"Found {len(host_services)} host services")
        
        # Optional network scan
        network_services = []
        if scan_network:
            print("Scanning network services...")
            network_services = self.scan_network_services()
            print(f"Found {len(network_services)} network services")
        
        # Combine all discoveries
        all_services = {
            'docker_containers': docker_services,
            'host_services': host_services,
            'network_services': network_services
        }
        
        # Generate integration config
        integration_config = self.generate_integration_config(
            docker_services + host_services + network_services
        )
        
        # Output results
        if output_format == 'json':
            print(json.dumps(all_services, indent=2))
            
            # Save integration config
            with open('external_services_integration.json', 'w') as f:
                json.dump(integration_config, f, indent=2)
                
        elif output_format == 'yaml':
            print(yaml.dump(all_services, default_flow_style=False))
            
            # Save integration config
            with open('external_services_integration.yaml', 'w') as f:
                yaml.dump(integration_config, f, default_flow_style=False)
        
        print(f"\nIntegration configuration saved to external_services_integration.{output_format}")

def main():
    parser = argparse.ArgumentParser(description='Discover external services for SutazAI integration')
    parser.add_argument('--scan-network', action='store_true', help='Scan network for services (requires nmap)')
    parser.add_argument('--output', choices=['json', 'yaml'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    discovery = ExternalServiceDiscovery()
    discovery.run(scan_network=args.scan_network, output_format=args.output)

if __name__ == '__main__':
    main()