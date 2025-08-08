#!/usr/bin/env python3
"""
Core Data Services Health Verification Script

Verifies Redis, PostgreSQL, and RabbitMQ services based on CLAUDE.md truth document.
Per the truth document:
- PostgreSQL (10000): HEALTHY, database has 14 tables (users, agents, tasks, etc.)
- Redis (10001): HEALTHY, cache layer functional
- RabbitMQ (10007/10008): HEALTHY, message queue running but not actively used

Usage:
    python scripts/devops/health_check_dataservices.py
    python scripts/devops/health_check_dataservices.py --redis-port 10001 --postgres-port 10000
    python scripts/devops/health_check_dataservices.py --timeout 10 --verbose

Created: December 19, 2024
Author: infrastructure-devops-manager agent
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import urllib.request
import urllib.parse
import urllib.error
import socket
import base64


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration with timestamp."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def check_tcp_connection(host: str, port: int, timeout: float, service_name: str) -> Dict[str, Any]:
    """Check TCP connectivity to data service with latency measurement."""
    result = {
        'service': service_name.lower(),
        'reachable': False,
        'latency_ms': None,
        'error': None
    }
    
    try:
        start_time = time.time()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            latency = int((time.time() - start_time) * 1000)
            result['reachable'] = True
            result['latency_ms'] = latency
            logging.info(f"{service_name} TCP connection successful at {host}:{port} (~{latency}ms)")
    except Exception as e:
        result['error'] = str(e)
        logging.error(f"{service_name} TCP connection failed at {host}:{port}: {e}")
    
    return result


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


def check_redis_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Check Redis cache service health (port 10001)."""
    results = {
        'service': 'redis',
        'port': port,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_result = check_tcp_connection(host, port, timeout, "Redis")
    results['checks']['tcp_connectivity'] = tcp_result['reachable']
    results['latency_ms'] = tcp_result['latency_ms']
    
    if not tcp_result['reachable']:
        results['status'] = 'unreachable'
        results['error'] = tcp_result['error']
        return results
    
    # 2. Redis protocol check (basic PING command)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            # Send PING command
            sock.send(b'*1\r\n$4\r\nPING\r\n')
            response = sock.recv(1024).decode('utf-8')
            
            if '+PONG' in response:
                results['checks']['ping_command'] = True
                logging.info("Redis PING command successful")
            else:
                results['checks']['ping_command'] = False
                logging.warning(f"Redis PING unexpected response: {response}")
                
    except Exception as e:
        results['checks']['ping_command'] = False
        logging.error(f"Redis PING command failed: {e}")
    
    # 3. Try INFO command to get server info
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            # Send INFO command
            sock.send(b'*1\r\n$4\r\nINFO\r\n')
            response = sock.recv(4096).decode('utf-8')
            
            if response.startswith('$') and 'redis_version' in response:
                results['checks']['info_command'] = True
                
                # Parse basic info
                info_data = {}
                for line in response.split('\r\n')[1:]:  # Skip length indicator
                    if ':' in line and not line.startswith('#'):
                        key, value = line.split(':', 1)
                        info_data[key] = value
                
                results['redis_info'] = {
                    'version': info_data.get('redis_version'),
                    'mode': info_data.get('redis_mode'),
                    'connected_clients': info_data.get('connected_clients'),
                    'used_memory_human': info_data.get('used_memory_human'),
                    'uptime_in_days': info_data.get('uptime_in_days')
                }
                
                logging.info(f"Redis version: {info_data.get('redis_version', 'unknown')}")
                logging.info(f"Redis clients: {info_data.get('connected_clients', 'unknown')}")
                
            else:
                results['checks']['info_command'] = False
                logging.warning("Redis INFO command failed or unexpected response")
                
    except Exception as e:
        results['checks']['info_command'] = False
        logging.error(f"Redis INFO command failed: {e}")
    
    # Overall status
    critical_checks = ['tcp_connectivity', 'ping_command']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ Redis is healthy and functional")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ Redis has critical issues")
    
    return results


def check_postgresql_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Check PostgreSQL database service health (port 10000)."""
    results = {
        'service': 'postgresql',
        'port': port,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_result = check_tcp_connection(host, port, timeout, "PostgreSQL")
    results['checks']['tcp_connectivity'] = tcp_result['reachable']
    results['latency_ms'] = tcp_result['latency_ms']
    
    if not tcp_result['reachable']:
        results['status'] = 'unreachable'
        results['error'] = tcp_result['error']
        return results
    
    # 2. PostgreSQL protocol handshake check
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            # Simple startup message (protocol version 3.0)
            startup_msg = b'\x00\x00\x00\x08\x04\xd2\x16/'  # SSL request
            sock.send(startup_msg)
            response = sock.recv(1024)
            
            if response:
                results['checks']['protocol_handshake'] = True
                logging.info("PostgreSQL protocol handshake successful")
                
                # Check for SSL support response
                if response == b'S':
                    results['ssl_supported'] = True
                elif response == b'N':
                    results['ssl_supported'] = False
                else:
                    results['ssl_supported'] = 'unknown'
                    
            else:
                results['checks']['protocol_handshake'] = False
                logging.warning("PostgreSQL protocol handshake failed")
                
    except Exception as e:
        results['checks']['protocol_handshake'] = False
        logging.error(f"PostgreSQL protocol check failed: {e}")
    
    # 3. Database connection attempt (basic check without credentials)
    # Note: This will fail without proper credentials but confirms PostgreSQL is running
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            # Startup message for database connection
            msg = bytearray()
            msg.extend(b'\x00\x00\x00\x00')  # Length placeholder
            msg.extend(b'\x00\x03\x00\x00')  # Protocol version 3.0
            msg.extend(b'user\x00test\x00')  # User parameter
            msg.extend(b'database\x00sutazai\x00')  # Database parameter
            msg.extend(b'\x00')  # Null terminator
            
            # Set correct length
            msg_len = len(msg)
            msg[0:4] = msg_len.to_bytes(4, 'big')
            
            sock.send(msg)
            response = sock.recv(1024)
            
            if response:
                results['checks']['database_response'] = True
                
                # Parse response type
                if response[0:1] == b'R':  # Authentication request
                    logging.info("PostgreSQL database responding (authentication required)")
                elif response[0:1] == b'E':  # Error response
                    logging.info("PostgreSQL database responding with error (expected without credentials)")
                else:
                    logging.info(f"PostgreSQL database response type: {response[0]}")
                    
            else:
                results['checks']['database_response'] = False
                
    except Exception as e:
        results['checks']['database_response'] = False
        logging.debug(f"PostgreSQL database connection attempt: {e}")
    
    # Overall status
    critical_checks = ['tcp_connectivity', 'protocol_handshake']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ PostgreSQL is healthy and responding")
        logging.info("NOTE: Per CLAUDE.md, database has 14 tables (users, agents, tasks, etc.)")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ PostgreSQL has critical issues")
    
    return results


def check_rabbitmq_service(host: str, amqp_port: int, mgmt_port: int, timeout: float) -> Dict[str, Any]:
    """Check RabbitMQ message queue service health (ports 10007 AMQP, 10008 management)."""
    results = {
        'service': 'rabbitmq',
        'ports': {'amqp': amqp_port, 'management': mgmt_port},
        'checks': {}
    }
    
    # 1. TCP connectivity checks
    amqp_tcp = check_tcp_connection(host, amqp_port, timeout, "RabbitMQ AMQP")
    mgmt_tcp = check_tcp_connection(host, mgmt_port, timeout, "RabbitMQ Management")
    
    results['checks']['amqp_tcp'] = amqp_tcp['reachable']
    results['checks']['management_tcp'] = mgmt_tcp['reachable']
    results['amqp_latency_ms'] = amqp_tcp['latency_ms']
    results['management_latency_ms'] = mgmt_tcp['latency_ms']
    
    if not amqp_tcp['reachable']:
        results['status'] = 'unreachable'
        results['error'] = amqp_tcp['error']
        return results
    
    # 2. AMQP protocol handshake
    try:
        with socket.create_connection((host, amqp_port), timeout=timeout) as sock:
            # AMQP protocol header
            sock.send(b'AMQP\x00\x00\x09\x01')
            response = sock.recv(1024)
            
            if response and len(response) >= 8:
                results['checks']['amqp_handshake'] = True
                
                # Parse AMQP response
                if response[:4] == b'AMQP':
                    version = f"{response[5]}.{response[6]}.{response[7]}"
                    results['amqp_version'] = version
                    logging.info(f"RabbitMQ AMQP handshake successful, version: {version}")
                else:
                    logging.info("RabbitMQ AMQP responding with connection start")
                    
            else:
                results['checks']['amqp_handshake'] = False
                logging.warning("RabbitMQ AMQP handshake failed")
                
    except Exception as e:
        results['checks']['amqp_handshake'] = False
        logging.error(f"RabbitMQ AMQP handshake failed: {e}")
    
    # 3. Management UI check (if management port is accessible)
    if mgmt_tcp['reachable']:
        logging.info("Checking RabbitMQ Management UI...")
        mgmt_url = f"http://{host}:{mgmt_port}"
        
        # Try management UI endpoints
        response = make_http_request(f"{mgmt_url}/", timeout)
        results['checks']['management_ui'] = response is not None and response.get('status_code', 0) < 500
        
        if results['checks']['management_ui']:
            logging.info("RabbitMQ Management UI is accessible")
            
            # Try to get cluster overview (may require auth)
            overview_response = make_http_request(f"{mgmt_url}/api/overview", timeout)
            if overview_response and overview_response.get('status_code') == 200:
                results['checks']['api_overview'] = True
                if 'json_data' in overview_response:
                    overview = overview_response['json_data']
                    results['cluster_info'] = {
                        'rabbitmq_version': overview.get('rabbitmq_version'),
                        'management_version': overview.get('management_version'),
                        'node': overview.get('node'),
                        'statistics_level': overview.get('statistics_level')
                    }
                    logging.info(f"RabbitMQ version: {overview.get('rabbitmq_version', 'unknown')}")
            elif overview_response and overview_response.get('status_code') == 401:
                results['checks']['api_overview'] = 'auth_required'
                logging.info("RabbitMQ API requires authentication")
            else:
                results['checks']['api_overview'] = False
        else:
            logging.warning("RabbitMQ Management UI not accessible")
    else:
        results['checks']['management_ui'] = False
        logging.warning("RabbitMQ Management port not reachable")
    
    # Overall status
    critical_checks = ['amqp_tcp', 'amqp_handshake']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ RabbitMQ is healthy and responding")
        logging.info("NOTE: Per CLAUDE.md, message queue running but not actively used")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ RabbitMQ has critical issues")
    
    return results


def main():
    """Main function with comprehensive data services health verification."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Redis, PostgreSQL, and RabbitMQ health verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/devops/health_check_dataservices.py
    python scripts/devops/health_check_dataservices.py --redis-port 10001 --postgres-port 10000
    python scripts/devops/health_check_dataservices.py --host 127.0.0.1 --timeout 15
    python scripts/devops/health_check_dataservices.py --skip-rabbitmq --verbose
        """
    )
    
    parser.add_argument('--host', default='localhost',
                       help='Data services host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=10001,
                       help='Redis port (default: 10001)')
    parser.add_argument('--postgres-port', type=int, default=10000,
                       help='PostgreSQL port (default: 10000)')
    parser.add_argument('--rabbitmq-amqp-port', type=int, default=10007,
                       help='RabbitMQ AMQP port (default: 10007)')
    parser.add_argument('--rabbitmq-mgmt-port', type=int, default=10008,
                       help='RabbitMQ Management port (default: 10008)')
    parser.add_argument('--timeout', type=float, default=10.0,
                       help='Request timeout in seconds (default: 10.0)')
    parser.add_argument('--skip-redis', action='store_true',
                       help='Skip Redis checks')
    parser.add_argument('--skip-postgres', action='store_true',
                       help='Skip PostgreSQL checks')
    parser.add_argument('--skip-rabbitmq', action='store_true',
                       help='Skip RabbitMQ checks')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("Starting Core Data Services health checks...")
    
    # Overall results
    results = {
        'timestamp': datetime.now().isoformat(),
        'category': 'data_services',
        'services': {}
    }
    
    all_services_healthy = True
    
    # Redis health check
    if not args.skip_redis:
        logging.info(f"=== Redis Cache Health Check (port {args.redis_port}) ===")
        redis_results = check_redis_service(args.host, args.redis_port, args.timeout)
        results['services']['redis'] = redis_results
        
        if redis_results['status'] != 'healthy':
            all_services_healthy = False
    
    # PostgreSQL health check
    if not args.skip_postgres:
        logging.info(f"=== PostgreSQL Database Health Check (port {args.postgres_port}) ===")
        postgres_results = check_postgresql_service(args.host, args.postgres_port, args.timeout)
        results['services']['postgresql'] = postgres_results
        
        if postgres_results['status'] != 'healthy':
            all_services_healthy = False
    
    # RabbitMQ health check
    if not args.skip_rabbitmq:
        logging.info(f"=== RabbitMQ Message Queue Health Check (ports {args.rabbitmq_amqp_port}/{args.rabbitmq_mgmt_port}) ===")
        rabbitmq_results = check_rabbitmq_service(args.host, args.rabbitmq_amqp_port, args.rabbitmq_mgmt_port, args.timeout)
        results['services']['rabbitmq'] = rabbitmq_results
        
        if rabbitmq_results['status'] != 'healthy':
            all_services_healthy = False
    
    # Overall summary
    results['overall_status'] = 'healthy' if all_services_healthy else 'unhealthy'
    results['services_checked'] = len(results['services'])
    results['healthy_services'] = sum(1 for svc in results['services'].values() if svc['status'] == 'healthy')
    
    if all_services_healthy:
        logging.info("✅ All core data services are healthy")
    else:
        logging.error("❌ One or more core data services require attention")
    
    # Output results as JSON for CI/CD integration
    if args.verbose:
        print(json.dumps(results, indent=2))
    
    return 0 if all_services_healthy else 1


if __name__ == "__main__":
    sys.exit(main())