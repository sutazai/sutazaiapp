#!/usr/bin/env python3
"""
Vector Database Services Health Verification Script

Verifies vector database services (Qdrant, FAISS, ChromaDB) based on CLAUDE.md truth document.
Per the truth document:
- Qdrant (10101/10102): HEALTHY, not integrated with app
- FAISS (10103): HEALTHY, not integrated with app  
- ChromaDB (10100): STARTING, connection issues

Usage:
    python scripts/devops/health_check_vectordb.py
    python scripts/devops/health_check_vectordb.py --port-range 10100-10103
    python scripts/devops/health_check_vectordb.py --timeout 10 --verbose

Created: December 19, 2024
Author: infrastructure-devops-manager agent
"""

import argparse
import json
import logging
from scripts.lib.logging_utils import setup_logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import urllib.request
import urllib.parse
import urllib.error
import socket




def check_tcp_connection(host: str, port: int, timeout: float, service_name: str) -> bool:
    """Check TCP connectivity to vector database service."""
    try:
        start_time = time.time()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            latency = int((time.time() - start_time) * 1000)
            logging.info(f"{service_name} TCP connection successful at {host}:{port} (~{latency}ms)")
            return True
    except Exception as e:
        logging.debug(f"{service_name} TCP connection failed at {host}:{port}: {e}")
        return False


def make_http_request(url: str, timeout: float, headers: Optional[Dict[str, str]] = None, 
                     method: str = 'GET', data: Optional[bytes] = None) -> Optional[Dict[str, Any]]:
    """Make HTTP request and return response data."""
    try:
        start_time = time.time()
        req = urllib.request.Request(url, method=method, data=data)
        
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
                result['content'] = content[:200] if content else ""
            
            return result
            
    except urllib.error.HTTPError as e:
        return {'status_code': e.code, 'error': str(e.reason)}
    except urllib.error.URLError as e:
        return {'error': str(e.reason)}
    except Exception as e:
        return {'error': str(e)}


def check_qdrant_service(host: str, http_port: int, grpc_port: int, timeout: float) -> Dict[str, Any]:
    """Check Qdrant vector database service (ports 10101 HTTP, 10102 gRPC)."""
    results = {
        'service': 'qdrant',
        'ports': {'http': http_port, 'grpc': grpc_port},
        'checks': {}
    }
    
    # 1. TCP connectivity checks
    http_tcp = check_tcp_connection(host, http_port, timeout, "Qdrant HTTP")
    grpc_tcp = check_tcp_connection(host, grpc_port, timeout, "Qdrant gRPC")
    
    results['checks']['http_tcp'] = http_tcp
    results['checks']['grpc_tcp'] = grpc_tcp
    
    if not http_tcp:
        results['status'] = 'unreachable'
        return results
    
    # 2. Qdrant API health check
    logging.info("Checking Qdrant HTTP API...")
    base_url = f"http://{host}:{http_port}"
    
    # Try root endpoint
    root_response = make_http_request(f"{base_url}/", timeout)
    results['checks']['http_api'] = root_response is not None and root_response.get('status_code') == 200
    
    # 3. Collections endpoint
    collections_response = make_http_request(f"{base_url}/collections", timeout)
    results['checks']['collections_endpoint'] = collections_response is not None and collections_response.get('status_code') == 200
    
    if results['checks']['collections_endpoint'] and 'json_data' in collections_response:
        collections_data = collections_response['json_data']
        collection_count = len(collections_data.get('result', {}).get('collections', []))
        results['collections_count'] = collection_count
        logging.info(f"Qdrant has {collection_count} collections")
    
    # 4. Cluster info (if available)
    cluster_response = make_http_request(f"{base_url}/cluster", timeout)
    if cluster_response and cluster_response.get('status_code') == 200 and 'json_data' in cluster_response:
        cluster_info = cluster_response['json_data']
        results['cluster_info'] = {
            'status': cluster_info.get('result', {}).get('status'),
            'peer_id': cluster_info.get('result', {}).get('peer_id')
        }
        logging.debug(f"Qdrant cluster status: {results['cluster_info']}")
    
    # Overall status
    if results['checks']['http_tcp'] and results['checks']['http_api']:
        results['status'] = 'healthy'
        logging.info("✅ Qdrant is healthy (not integrated with app per CLAUDE.md)")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ Qdrant has issues")
    
    return results


def check_faiss_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Check FAISS vector database service (port 10103)."""
    results = {
        'service': 'faiss',
        'port': port,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_ok = check_tcp_connection(host, port, timeout, "FAISS")
    results['checks']['tcp_connectivity'] = tcp_ok
    
    if not tcp_ok:
        results['status'] = 'unreachable'
        return results
    
    # 2. HTTP API check (if available)
    base_url = f"http://{host}:{port}"
    
    # Try common endpoints
    endpoints_to_check = [
        ('/', 'root'),
        ('/health', 'health'),
        ('/status', 'status'),
        ('/api/v1/health', 'api_health')
    ]
    
    api_responsive = False
    for endpoint, name in endpoints_to_check:
        response = make_http_request(f"{base_url}{endpoint}", timeout)
        if response and response.get('status_code', 0) < 500:
            results['checks'][f'{name}_endpoint'] = True
            api_responsive = True
            logging.info(f"FAISS {name} endpoint responsive")
            break
        else:
            results['checks'][f'{name}_endpoint'] = False
    
    results['checks']['http_api'] = api_responsive
    
    # Overall status
    if tcp_ok:
        results['status'] = 'healthy'
        logging.info("✅ FAISS is healthy (not integrated with app per CLAUDE.md)")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ FAISS has issues")
    
    return results


def check_chromadb_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Check ChromaDB vector database service (port 10100) - known to have connection issues."""
    results = {
        'service': 'chromadb',
        'port': port,
        'checks': {}
    }
    
    # 1. TCP connectivity (expected to be problematic per CLAUDE.md)
    tcp_ok = check_tcp_connection(host, port, timeout, "ChromaDB")
    results['checks']['tcp_connectivity'] = tcp_ok
    
    if not tcp_ok:
        results['status'] = 'connection_issues'
        logging.warning("⚠️ ChromaDB connection issues (expected per CLAUDE.md)")
        return results
    
    # 2. ChromaDB API check
    base_url = f"http://{host}:{port}"
    
    # Try ChromaDB specific endpoints
    endpoints_to_check = [
        ('/api/v1', 'api_v1'),
        ('/api/v1/heartbeat', 'heartbeat'),
        ('/api/v1/version', 'version')
    ]
    
    api_working = False
    for endpoint, name in endpoints_to_check:
        response = make_http_request(f"{base_url}{endpoint}", timeout)
        if response and response.get('status_code') == 200:
            results['checks'][f'{name}_endpoint'] = True
            api_working = True
            
            if name == 'version' and 'json_data' in response:
                version_info = response['json_data']
                results['version'] = version_info
                logging.info(f"ChromaDB version info: {version_info}")
            break
        else:
            results['checks'][f'{name}_endpoint'] = False
    
    results['checks']['http_api'] = api_working
    
    # 3. Try to get collections (if API is working)
    if api_working:
        collections_response = make_http_request(f"{base_url}/api/v1/collections", timeout)
        if collections_response and collections_response.get('status_code') == 200:
            results['checks']['collections'] = True
            if 'json_data' in collections_response:
                collections = collections_response['json_data']
                results['collections_count'] = len(collections) if isinstance(collections, list) else 0
                logging.info(f"ChromaDB has {results.get('collections_count', 0)} collections")
        else:
            results['checks']['collections'] = False
    
    # Overall status (expect issues per CLAUDE.md)
    if tcp_ok and api_working:
        results['status'] = 'healthy'
        logging.info("✅ ChromaDB is working (unexpected improvement!)")
    elif tcp_ok:
        results['status'] = 'partial'
        logging.warning("⚠️ ChromaDB TCP works but API issues")
    else:
        results['status'] = 'connection_issues'
        logging.warning("⚠️ ChromaDB connection issues (expected per CLAUDE.md)")
    
    return results


def check_generic_vector_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Generic check for unknown vector database services."""
    results = {
        'service': f'vector_db_{port}',
        'port': port,
        'checks': {}
    }
    
    tcp_ok = check_tcp_connection(host, port, timeout, f"VectorDB:{port}")
    results['checks']['tcp_connectivity'] = tcp_ok
    
    if tcp_ok:
        # Try to determine service type via HTTP
        base_url = f"http://{host}:{port}"
        response = make_http_request(f"{base_url}/", timeout)
        
        if response and response.get('status_code', 0) < 500:
            results['checks']['http_responsive'] = True
            results['status'] = 'healthy'
            logging.info(f"✅ Vector service on port {port} is responsive")
        else:
            results['checks']['http_responsive'] = False
            results['status'] = 'tcp_only'
            logging.info(f"ℹ️ Vector service on port {port} - TCP only")
    else:
        results['status'] = 'unreachable'
    
    return results


def parse_port_range(range_str: str) -> Tuple[int, int]:
    """Parse port range string like '10100-10103'."""
    try:
        start_str, end_str = range_str.split('-')
        start_port = int(start_str.strip())
        end_port = int(end_str.strip())
        
        if start_port > end_port:
            raise ValueError("Start port must be <= end port")
        if start_port < 1 or end_port > 65535:
            raise ValueError("Ports must be in range 1-65535")
            
        return start_port, end_port
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid port range '{range_str}': {e}")


def main():
    """Main function with comprehensive vector database health verification."""
    parser = argparse.ArgumentParser(
        description="Comprehensive vector database services health verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/devops/health_check_vectordb.py
    python scripts/devops/health_check_vectordb.py --port-range 10100-10103
    python scripts/devops/health_check_vectordb.py --host 127.0.0.1 --timeout 15
    python scripts/devops/health_check_vectordb.py --specific-services --verbose
        """
    )
    
    parser.add_argument('--host', default='localhost',
                       help='Vector database services host (default: localhost)')
    parser.add_argument('--port-range', type=parse_port_range, default=(10100, 10103),
                       help='Port range to check (default: 10100-10103)')
    parser.add_argument('--timeout', type=float, default=5.0,
                       help='Request timeout in seconds (default: 5.0)')
    parser.add_argument('--specific-services', action='store_true',
                       help='Use specific service checks instead of generic')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    start_port, end_port = args.port_range
    logging.info(f"Starting Vector Database health checks (ports {start_port}-{end_port})...")
    
    # Overall results
    results = {
        'timestamp': datetime.now().isoformat(),
        'category': 'vector_databases',
        'port_range': f"{start_port}-{end_port}",
        'services': {}
    }
    
    all_services_status = []
    
    if args.specific_services:
        # Specific service checks based on CLAUDE.md known services
        logging.info("=== ChromaDB Health Check (port 10100) ===")
        if 10100 >= start_port and 10100 <= end_port:
            chromadb_results = check_chromadb_service(args.host, 10100, args.timeout)
            results['services']['chromadb'] = chromadb_results
            all_services_status.append(chromadb_results['status'])
        
        logging.info("=== Qdrant Health Check (ports 10101 HTTP, 10102 gRPC) ===")
        if 10101 >= start_port and 10101 <= end_port:
            qdrant_results = check_qdrant_service(args.host, 10101, 10102, args.timeout)
            results['services']['qdrant'] = qdrant_results  
            all_services_status.append(qdrant_results['status'])
        
        logging.info("=== FAISS Health Check (port 10103) ===")
        if 10103 >= start_port and 10103 <= end_port:
            faiss_results = check_faiss_service(args.host, 10103, args.timeout)
            results['services']['faiss'] = faiss_results
            all_services_status.append(faiss_results['status'])
    
    else:
        # Generic port range check
        for port in range(start_port, end_port + 1):
            logging.info(f"=== Checking Vector Service on Port {port} ===")
            service_results = check_generic_vector_service(args.host, port, args.timeout)
            results['services'][f'port_{port}'] = service_results
            all_services_status.append(service_results['status'])
    
    # Overall summary
    healthy_count = sum(1 for status in all_services_status if status == 'healthy')
    total_count = len(all_services_status)
    
    results['summary'] = {
        'total_services': total_count,
        'healthy_services': healthy_count,
        'unhealthy_services': total_count - healthy_count,
        'overall_status': 'healthy' if healthy_count == total_count else 'mixed'
    }
    
    if healthy_count == total_count:
        logging.info("✅ All vector database services are healthy")
    elif healthy_count > 0:
        logging.warning(f"⚠️ {healthy_count}/{total_count} vector database services are healthy")
    else:
        logging.error("❌ No vector database services are healthy")
    
    # Note about integration status per CLAUDE.md
    logging.info("NOTE: Per CLAUDE.md, vector databases are not integrated with the application")
    
    # Output results as JSON for CI/CD integration
    if args.verbose:
        print(json.dumps(results, indent=2))
    
    # Return 0 if at least half the services are healthy, 1 otherwise
    return 0 if healthy_count >= (total_count / 2) else 1


if __name__ == "__main__":
    sys.exit(main())
