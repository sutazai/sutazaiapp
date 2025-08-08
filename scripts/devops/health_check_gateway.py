#!/usr/bin/env python3
"""
API Gateway Services Health Verification Script

Verifies Kong API Gateway and Consul service discovery health based on CLAUDE.md truth document.
Per the truth document: Kong/Consul/RabbitMQ running but not configured or integrated.

Usage:
    python scripts/devops/health_check_gateway.py
    python scripts/devops/health_check_gateway.py --kong-port 10005 --consul-port 10006
    python scripts/devops/health_check_gateway.py --timeout 10 --verbose

Created: December 19, 2024
Author: infrastructure-devops-manager agent
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import urllib.request
import urllib.parse
import urllib.error
import socket


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration with timestamp."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_tcp_connection(host: str, port: int, timeout: float, service_name: str) -> bool:
    """Check TCP connectivity to service."""
    try:
        start_time = time.time()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            latency = int((time.time() - start_time) * 1000)
            logging.info(f"{service_name} TCP connection successful at {host}:{port} (~{latency}ms)")
            return True
    except Exception as e:
        logging.error(f"{service_name} TCP connection failed at {host}:{port}: {e}")
        return False


def make_http_request(url: str, timeout: float, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Make HTTP request and return response data."""
    try:
        start_time = time.time()
        req = urllib.request.Request(url)
        
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency = int((time.time() - start_time) * 1000)
            content = response.read().decode('utf-8')
            
            logging.debug(f"HTTP {response.getcode()} from {url} in {latency}ms")
            
            result = {
                'status_code': response.getcode(),
                'latency_ms': latency,
                'content_length': len(content)
            }
            
            try:
                result['json_data'] = json.loads(content) if content else {}
            except json.JSONDecodeError:
                result['content'] = content[:500]  # First 500 chars for debugging
            
            return result
            
    except urllib.error.HTTPError as e:
        logging.error(f"HTTP error {e.code} from {url}: {e.reason}")
        return {'status_code': e.code, 'error': str(e.reason)}
    except urllib.error.URLError as e:
        logging.error(f"URL error from {url}: {e.reason}")
        return {'error': str(e.reason)}
    except Exception as e:
        logging.error(f"Request to {url} failed: {e}")
        return {'error': str(e)}


def check_kong_gateway(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Comprehensive Kong API Gateway health check."""
    base_url = f"http://{host}:{port}"
    results = {
        'service': 'kong',
        'base_url': base_url,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_ok = check_tcp_connection(host, port, timeout, "Kong")
    results['checks']['tcp_connectivity'] = tcp_ok
    
    if not tcp_ok:
        results['status'] = 'unreachable'
        return results
    
    # 2. Root endpoint check
    logging.info("Checking Kong root endpoint...")
    root_response = make_http_request(f"{base_url}/", timeout)
    results['checks']['root_endpoint'] = root_response is not None and root_response.get('status_code', 0) < 500
    
    # 3. Admin API status
    logging.info("Checking Kong Admin API status...")
    status_response = make_http_request(f"{base_url}/status", timeout)
    results['checks']['admin_api'] = status_response is not None and status_response.get('status_code') == 200
    
    if results['checks']['admin_api'] and 'json_data' in status_response:
        kong_status = status_response['json_data']
        results['kong_info'] = {
            'version': kong_status.get('version'),
            'configuration_hash': kong_status.get('configuration_hash'),
            'plugins': kong_status.get('plugins', {}).get('available_on_server', [])
        }
        logging.info(f"Kong version: {kong_status.get('version', 'unknown')}")
    
    # 4. Services endpoint (expect empty since not configured per CLAUDE.md)
    logging.info("Checking Kong services configuration...")
    services_response = make_http_request(f"{base_url}/services", timeout)
    results['checks']['services_endpoint'] = services_response is not None and services_response.get('status_code') == 200
    
    if results['checks']['services_endpoint'] and 'json_data' in services_response:
        services_data = services_response['json_data']
        service_count = len(services_data.get('data', []))
        results['services_configured'] = service_count
        if service_count == 0:
            logging.info("Kong has no services configured (expected per CLAUDE.md)")
        else:
            logging.warning(f"Kong has {service_count} services configured (unexpected)")
    
    # 5. Routes endpoint
    logging.info("Checking Kong routes configuration...")
    routes_response = make_http_request(f"{base_url}/routes", timeout)
    results['checks']['routes_endpoint'] = routes_response is not None and routes_response.get('status_code') == 200
    
    if results['checks']['routes_endpoint'] and 'json_data' in routes_response:
        routes_data = routes_response['json_data']
        route_count = len(routes_data.get('data', []))
        results['routes_configured'] = route_count
        if route_count == 0:
            logging.info("Kong has no routes configured (expected per CLAUDE.md)")
        else:
            logging.info(f"Kong has {route_count} routes configured")
    
    # Overall status
    critical_checks = ['tcp_connectivity', 'admin_api']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ Kong Gateway is healthy and responding")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ Kong Gateway has critical issues")
    
    return results


def check_consul_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Comprehensive Consul service discovery health check."""
    base_url = f"http://{host}:{port}"
    results = {
        'service': 'consul',
        'base_url': base_url,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_ok = check_tcp_connection(host, port, timeout, "Consul")
    results['checks']['tcp_connectivity'] = tcp_ok
    
    if not tcp_ok:
        results['status'] = 'unreachable'
        return results
    
    # 2. Leader status check
    logging.info("Checking Consul leader status...")
    leader_response = make_http_request(f"{base_url}/v1/status/leader", timeout)
    results['checks']['leader_status'] = leader_response is not None and leader_response.get('status_code') == 200
    
    if results['checks']['leader_status'] and 'content' in leader_response:
        leader_info = leader_response['content'].strip('"')
        results['leader'] = leader_info
        logging.info(f"Consul leader: {leader_info}")
    
    # 3. Peers status
    logging.info("Checking Consul peers...")
    peers_response = make_http_request(f"{base_url}/v1/status/peers", timeout)
    results['checks']['peers_status'] = peers_response is not None and peers_response.get('status_code') == 200
    
    if results['checks']['peers_status'] and 'json_data' in peers_response:
        peers = peers_response['json_data']
        results['peer_count'] = len(peers) if isinstance(peers, list) else 0
        logging.info(f"Consul peers: {results['peer_count']}")
    
    # 4. Services catalog (expect minimal usage per CLAUDE.md)
    logging.info("Checking Consul services catalog...")
    services_response = make_http_request(f"{base_url}/v1/catalog/services", timeout)
    results['checks']['catalog_services'] = services_response is not None and services_response.get('status_code') == 200
    
    if results['checks']['catalog_services'] and 'json_data' in services_response:
        services = services_response['json_data']
        service_count = len(services) if isinstance(services, dict) else 0
        results['services_registered'] = service_count
        results['service_names'] = list(services.keys()) if isinstance(services, dict) else []
        
        if service_count <= 1:  # Just consul service itself
            logging.info(f"Consul has minimal services registered: {results['service_names']} (expected per CLAUDE.md)")
        else:
            logging.info(f"Consul has {service_count} services registered: {results['service_names']}")
    
    # 5. Nodes in cluster
    logging.info("Checking Consul cluster nodes...")
    nodes_response = make_http_request(f"{base_url}/v1/catalog/nodes", timeout)
    results['checks']['catalog_nodes'] = nodes_response is not None and nodes_response.get('status_code') == 200
    
    if results['checks']['catalog_nodes'] and 'json_data' in nodes_response:
        nodes = nodes_response['json_data']
        node_count = len(nodes) if isinstance(nodes, list) else 0
        results['node_count'] = node_count
        logging.info(f"Consul cluster has {node_count} nodes")
    
    # 6. Health checks
    logging.info("Checking Consul health checks...")
    health_response = make_http_request(f"{base_url}/v1/health/state/any", timeout)
    results['checks']['health_checks'] = health_response is not None and health_response.get('status_code') == 200
    
    if results['checks']['health_checks'] and 'json_data' in health_response:
        health_checks = health_response['json_data']
        results['health_check_count'] = len(health_checks) if isinstance(health_checks, list) else 0
        logging.info(f"Consul has {results['health_check_count']} health checks")
    
    # Overall status
    critical_checks = ['tcp_connectivity', 'leader_status', 'catalog_services']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ Consul is healthy and responding")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ Consul has critical issues")
    
    return results


def main():
    """Main function with comprehensive API Gateway health verification."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Kong + Consul health verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/devops/health_check_gateway.py
    python scripts/devops/health_check_gateway.py --kong-port 10005 --consul-port 10006
    python scripts/devops/health_check_gateway.py --host 127.0.0.1 --timeout 15
    python scripts/devops/health_check_gateway.py --skip-consul --verbose
        """
    )
    
    parser.add_argument('--host', default='localhost',
                       help='Gateway services host (default: localhost)')
    parser.add_argument('--kong-port', type=int, default=10005,
                       help='Kong Gateway port (default: 10005)')
    parser.add_argument('--consul-port', type=int, default=10006,
                       help='Consul service port (default: 10006)')
    parser.add_argument('--timeout', type=float, default=10.0,
                       help='Request timeout in seconds (default: 10.0)')
    parser.add_argument('--skip-kong', action='store_true',
                       help='Skip Kong Gateway checks')
    parser.add_argument('--skip-consul', action='store_true',
                       help='Skip Consul checks')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("Starting API Gateway health checks...")
    
    # Overall results
    results = {
        'timestamp': datetime.now().isoformat(),
        'category': 'api_gateway',
        'services': {}
    }
    
    all_services_healthy = True
    
    # Kong Gateway health check
    if not args.skip_kong:
        logging.info(f"=== Kong Gateway Health Check (port {args.kong_port}) ===")
        kong_results = check_kong_gateway(args.host, args.kong_port, args.timeout)
        results['services']['kong'] = kong_results
        
        if kong_results['status'] != 'healthy':
            all_services_healthy = False
    
    # Consul service discovery health check
    if not args.skip_consul:
        logging.info(f"=== Consul Service Discovery Health Check (port {args.consul_port}) ===")
        consul_results = check_consul_service(args.host, args.consul_port, args.timeout)
        results['services']['consul'] = consul_results
        
        if consul_results['status'] != 'healthy':
            all_services_healthy = False
    
    # Overall summary
    results['overall_status'] = 'healthy' if all_services_healthy else 'unhealthy'
    results['services_checked'] = len(results['services'])
    results['healthy_services'] = sum(1 for svc in results['services'].values() if svc['status'] == 'healthy')
    
    if all_services_healthy:
        logging.info("✅ All API Gateway services are healthy")
    else:
        logging.error("❌ One or more API Gateway services require attention")
    
    # Note about configuration status per CLAUDE.md
    logging.info("NOTE: Per CLAUDE.md, Kong/Consul are running but not configured or integrated")
    
    # Output results as JSON for CI/CD integration
    if args.verbose:
        print(json.dumps(results, indent=2))
    
    return 0 if all_services_healthy else 1


if __name__ == "__main__":
    sys.exit(main())