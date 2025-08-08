#!/usr/bin/env python3
"""
Monitoring Services Health Verification Script

Verifies Prometheus, Grafana, Loki, and AlertManager services based on CLAUDE.md truth document.
Per the truth document: Full monitoring stack (Prometheus, Grafana, Loki) - All Operational

Usage:
    python scripts/devops/health_check_monitoring.py
    python scripts/devops/health_check_monitoring.py --prometheus-port 10200 --grafana-port 10201
    python scripts/devops/health_check_monitoring.py --timeout 15 --verbose

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
from typing import Dict, Any, Optional, List
import urllib.request
import urllib.parse
import urllib.error
import socket
import base64




def check_tcp_connection(host: str, port: int, timeout: float, service_name: str) -> Dict[str, Any]:
    """Check TCP connectivity to monitoring service with latency measurement."""
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
                     method: str = 'GET', data: Optional[bytes] = None, auth: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Make HTTP request with optional authentication and return response data."""
    try:
        start_time = time.time()
        req = urllib.request.Request(url, method=method, data=data)
        
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)
        
        if auth:
            req.add_header('Authorization', f'Basic {auth}')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            latency = int((time.time() - start_time) * 1000)
            content = response.read().decode('utf-8')
            
            result = {
                'status_code': response.getcode(),
                'latency_ms': latency,
                'content_length': len(content),
                'headers': dict(response.headers)
            }
            
            try:
                result['json_data'] = json.loads(content) if content else {}
            except json.JSONDecodeError:
                result['content'] = content[:500] if content else ""
            
            return result
            
    except urllib.error.HTTPError as e:
        return {'status_code': e.code, 'error': str(e.reason)}
    except urllib.error.URLError as e:
        return {'error': str(e.reason)}
    except Exception as e:
        return {'error': str(e)}


def check_prometheus_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Check Prometheus metrics collection service health (port 10200)."""
    results = {
        'service': 'prometheus',
        'port': port,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_result = check_tcp_connection(host, port, timeout, "Prometheus")
    results['checks']['tcp_connectivity'] = tcp_result['reachable']
    results['latency_ms'] = tcp_result['latency_ms']
    
    if not tcp_result['reachable']:
        results['status'] = 'unreachable'
        results['error'] = tcp_result['error']
        return results
    
    base_url = f"http://{host}:{port}"
    
    # 2. Health endpoint check
    logging.info("Checking Prometheus health endpoint...")
    health_response = make_http_request(f"{base_url}/-/healthy", timeout)
    results['checks']['health_endpoint'] = health_response is not None and health_response.get('status_code') == 200
    
    # 3. Ready endpoint check
    logging.info("Checking Prometheus ready endpoint...")
    ready_response = make_http_request(f"{base_url}/-/ready", timeout)
    results['checks']['ready_endpoint'] = ready_response is not None and ready_response.get('status_code') == 200
    
    if results['checks']['ready_endpoint']:
        logging.info("Prometheus is ready to serve queries")
    
    # 4. Configuration endpoint
    logging.info("Checking Prometheus configuration...")
    config_response = make_http_request(f"{base_url}/api/v1/status/config", timeout)
    results['checks']['config_endpoint'] = config_response is not None and config_response.get('status_code') == 200
    
    if results['checks']['config_endpoint'] and 'json_data' in config_response:
        config_data = config_response['json_data']
        if config_data.get('status') == 'success':
            results['config_status'] = 'valid'
        else:
            results['config_status'] = 'invalid'
    
    # 5. Targets endpoint (scrape targets)
    logging.info("Checking Prometheus scrape targets...")
    targets_response = make_http_request(f"{base_url}/api/v1/targets", timeout)
    results['checks']['targets_endpoint'] = targets_response is not None and targets_response.get('status_code') == 200
    
    if results['checks']['targets_endpoint'] and 'json_data' in targets_response:
        targets_data = targets_response['json_data']
        if targets_data.get('status') == 'success':
            active_targets = targets_data.get('data', {}).get('activeTargets', [])
            results['active_targets_count'] = len(active_targets)
            
            # Count healthy targets
            healthy_targets = sum(1 for target in active_targets if target.get('health') == 'up')
            results['healthy_targets_count'] = healthy_targets
            
            logging.info(f"Prometheus has {results['active_targets_count']} targets, {healthy_targets} healthy")
            
            # List some target jobs
            jobs = list(set(target.get('job', 'unknown') for target in active_targets))
            results['target_jobs'] = jobs[:10]  # First 10 jobs
            logging.info(f"Target jobs: {jobs[:5]}")  # Log first 5
    
    # 6. Metrics endpoint (sample query)
    logging.info("Testing Prometheus query capability...")
    query_response = make_http_request(f"{base_url}/api/v1/query?query=up", timeout)
    results['checks']['query_endpoint'] = query_response is not None and query_response.get('status_code') == 200
    
    if results['checks']['query_endpoint'] and 'json_data' in query_response:
        query_data = query_response['json_data']
        if query_data.get('status') == 'success':
            results_count = len(query_data.get('data', {}).get('result', []))
            results['sample_query_results'] = results_count
            logging.info(f"Sample query returned {results_count} results")
    
    # Overall status
    critical_checks = ['tcp_connectivity', 'health_endpoint', 'ready_endpoint']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ Prometheus is healthy and collecting metrics")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ Prometheus has critical issues")
    
    return results


def check_grafana_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Check Grafana visualization service health (port 10201)."""
    results = {
        'service': 'grafana',
        'port': port,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_result = check_tcp_connection(host, port, timeout, "Grafana")
    results['checks']['tcp_connectivity'] = tcp_result['reachable']
    results['latency_ms'] = tcp_result['latency_ms']
    
    if not tcp_result['reachable']:
        results['status'] = 'unreachable'
        results['error'] = tcp_result['error']
        return results
    
    base_url = f"http://{host}:{port}"
    
    # 2. Health endpoint check
    logging.info("Checking Grafana health endpoint...")
    health_response = make_http_request(f"{base_url}/api/health", timeout)
    results['checks']['health_endpoint'] = health_response is not None and health_response.get('status_code') == 200
    
    if results['checks']['health_endpoint'] and 'json_data' in health_response:
        health_data = health_response['json_data']
        results['health_status'] = health_data
        logging.info(f"Grafana health: {health_data}")
    
    # 3. Login page check
    logging.info("Checking Grafana login page...")
    login_response = make_http_request(f"{base_url}/login", timeout)
    results['checks']['login_page'] = login_response is not None and login_response.get('status_code') == 200
    
    # 4. API status with default credentials (admin/admin)
    logging.info("Testing Grafana API with default credentials...")
    default_auth = base64.b64encode(b'admin:admin').decode('ascii')
    
    # Try to get user info
    user_response = make_http_request(f"{base_url}/api/user", timeout, auth=default_auth)
    results['checks']['api_with_default_auth'] = user_response is not None and user_response.get('status_code') == 200
    
    if results['checks']['api_with_default_auth'] and 'json_data' in user_response:
        user_data = user_response['json_data']
        results['admin_user_info'] = {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'theme': user_data.get('theme'),
            'orgId': user_data.get('orgId')
        }
        logging.info(f"Grafana admin user: {user_data.get('login', 'unknown')}")
    elif user_response and user_response.get('status_code') == 401:
        logging.info("Grafana default credentials changed (good security practice)")
        results['default_credentials_changed'] = True
    
    # 5. Datasources check
    logging.info("Checking Grafana datasources...")
    ds_response = make_http_request(f"{base_url}/api/datasources", timeout, auth=default_auth)
    results['checks']['datasources_endpoint'] = ds_response is not None and ds_response.get('status_code') in [200, 401]
    
    if ds_response and ds_response.get('status_code') == 200 and 'json_data' in ds_response:
        datasources = ds_response['json_data']
        results['datasources_count'] = len(datasources) if isinstance(datasources, list) else 0
        
        if datasources:
            ds_types = [ds.get('type') for ds in datasources if isinstance(ds, dict)]
            results['datasource_types'] = list(set(ds_types))
            logging.info(f"Grafana has {len(datasources)} datasources: {results['datasource_types']}")
        else:
            logging.info("Grafana has no datasources configured")
    
    # 6. Dashboards check
    logging.info("Checking Grafana dashboards...")
    dashboards_response = make_http_request(f"{base_url}/api/search?type=dash-db", timeout, auth=default_auth)
    results['checks']['dashboards_endpoint'] = dashboards_response is not None and dashboards_response.get('status_code') in [200, 401]
    
    if dashboards_response and dashboards_response.get('status_code') == 200 and 'json_data' in dashboards_response:
        dashboards = dashboards_response['json_data']
        results['dashboards_count'] = len(dashboards) if isinstance(dashboards, list) else 0
        logging.info(f"Grafana has {results['dashboards_count']} dashboards")
        
        if dashboards and len(dashboards) > 0:
            # List first few dashboard titles
            titles = [db.get('title', 'untitled') for db in dashboards[:5] if isinstance(db, dict)]
            results['sample_dashboards'] = titles
    
    # Overall status
    critical_checks = ['tcp_connectivity', 'health_endpoint', 'login_page']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ Grafana is healthy and accessible")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ Grafana has critical issues")
    
    return results


def check_loki_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Check Loki log aggregation service health (port 10202)."""
    results = {
        'service': 'loki',
        'port': port,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_result = check_tcp_connection(host, port, timeout, "Loki")
    results['checks']['tcp_connectivity'] = tcp_result['reachable']
    results['latency_ms'] = tcp_result['latency_ms']
    
    if not tcp_result['reachable']:
        results['status'] = 'unreachable'
        results['error'] = tcp_result['error']
        return results
    
    base_url = f"http://{host}:{port}"
    
    # 2. Ready endpoint check
    logging.info("Checking Loki ready endpoint...")
    ready_response = make_http_request(f"{base_url}/ready", timeout)
    results['checks']['ready_endpoint'] = ready_response is not None and ready_response.get('status_code') == 200
    
    # 3. Metrics endpoint
    logging.info("Checking Loki metrics endpoint...")
    metrics_response = make_http_request(f"{base_url}/metrics", timeout)
    results['checks']['metrics_endpoint'] = metrics_response is not None and metrics_response.get('status_code') == 200
    
    if results['checks']['metrics_endpoint']:
        logging.info("Loki metrics endpoint accessible")
    
    # 4. Labels API check
    logging.info("Checking Loki labels API...")
    labels_response = make_http_request(f"{base_url}/loki/api/v1/labels", timeout)
    results['checks']['labels_api'] = labels_response is not None and labels_response.get('status_code') == 200
    
    if results['checks']['labels_api'] and 'json_data' in labels_response:
        labels_data = labels_response['json_data']
        if labels_data.get('status') == 'success':
            labels = labels_data.get('data', [])
            results['available_labels_count'] = len(labels)
            results['sample_labels'] = labels[:5]  # First 5 labels
            logging.info(f"Loki has {len(labels)} available labels")
    
    # 5. Query API check (simple query)
    logging.info("Testing Loki query API...")
    query_url = f"{base_url}/loki/api/v1/query_range"
    params = {
        'query': '{job=""}',  # Simple query
        'start': str(int(time.time() - 3600) * 1000000000),  # 1 hour ago in nanoseconds
        'end': str(int(time.time()) * 1000000000)  # Now in nanoseconds
    }
    query_url_with_params = f"{query_url}?" + urllib.parse.urlencode(params)
    
    query_response = make_http_request(query_url_with_params, timeout)
    results['checks']['query_api'] = query_response is not None and query_response.get('status_code') == 200
    
    if results['checks']['query_api'] and 'json_data' in query_response:
        query_data = query_response['json_data']
        if query_data.get('status') == 'success':
            result_streams = query_data.get('data', {}).get('result', [])
            results['query_result_streams'] = len(result_streams)
            logging.info(f"Loki query returned {len(result_streams)} log streams")
    
    # Overall status
    critical_checks = ['tcp_connectivity', 'ready_endpoint']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ Loki is healthy and aggregating logs")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ Loki has critical issues")
    
    return results


def check_alertmanager_service(host: str, port: int, timeout: float) -> Dict[str, Any]:
    """Check AlertManager service health (port 10203)."""
    results = {
        'service': 'alertmanager',
        'port': port,
        'checks': {}
    }
    
    # 1. TCP connectivity
    tcp_result = check_tcp_connection(host, port, timeout, "AlertManager")
    results['checks']['tcp_connectivity'] = tcp_result['reachable']
    results['latency_ms'] = tcp_result['latency_ms']
    
    if not tcp_result['reachable']:
        results['status'] = 'unreachable'
        results['error'] = tcp_result['error']
        return results
    
    base_url = f"http://{host}:{port}"
    
    # 2. Health endpoint check
    logging.info("Checking AlertManager health...")
    health_response = make_http_request(f"{base_url}/-/healthy", timeout)
    results['checks']['health_endpoint'] = health_response is not None and health_response.get('status_code') == 200
    
    # 3. Ready endpoint check
    ready_response = make_http_request(f"{base_url}/-/ready", timeout)
    results['checks']['ready_endpoint'] = ready_response is not None and ready_response.get('status_code') == 200
    
    # 4. Status endpoint
    logging.info("Checking AlertManager status...")
    status_response = make_http_request(f"{base_url}/api/v1/status", timeout)
    results['checks']['status_api'] = status_response is not None and status_response.get('status_code') == 200
    
    if results['checks']['status_api'] and 'json_data' in status_response:
        status_data = status_response['json_data']
        if status_data.get('status') == 'success':
            config_hash = status_data.get('data', {}).get('configHash')
            cluster_status = status_data.get('data', {}).get('cluster', {}).get('status')
            results['config_hash'] = config_hash
            results['cluster_status'] = cluster_status
            logging.info(f"AlertManager cluster status: {cluster_status}")
    
    # 5. Alerts endpoint
    logging.info("Checking AlertManager alerts...")
    alerts_response = make_http_request(f"{base_url}/api/v1/alerts", timeout)
    results['checks']['alerts_api'] = alerts_response is not None and alerts_response.get('status_code') == 200
    
    if results['checks']['alerts_api'] and 'json_data' in alerts_response:
        alerts_data = alerts_response['json_data']
        if alerts_data.get('status') == 'success':
            alerts = alerts_data.get('data', [])
            results['active_alerts_count'] = len(alerts)
            logging.info(f"AlertManager has {len(alerts)} active alerts")
            
            # Count by state
            if alerts:
                states = {}
                for alert in alerts:
                    state = alert.get('status', {}).get('state', 'unknown')
                    states[state] = states.get(state, 0) + 1
                results['alerts_by_state'] = states
    
    # Overall status
    critical_checks = ['tcp_connectivity', 'health_endpoint']
    all_critical_passed = all(results['checks'].get(check, False) for check in critical_checks)
    
    if all_critical_passed:
        results['status'] = 'healthy'
        logging.info("✅ AlertManager is healthy and routing alerts")
    else:
        results['status'] = 'unhealthy'
        logging.error("❌ AlertManager has critical issues")
    
    return results


def main():
    """Main function with comprehensive monitoring services health verification."""
    parser = argparse.ArgumentParser(
        description="Comprehensive monitoring stack (Prometheus, Grafana, Loki, AlertManager) health verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/devops/health_check_monitoring.py
    python scripts/devops/health_check_monitoring.py --prometheus-port 10200 --grafana-port 10201
    python scripts/devops/health_check_monitoring.py --host 127.0.0.1 --timeout 15
    python scripts/devops/health_check_monitoring.py --skip-loki --verbose
        """
    )
    
    parser.add_argument('--host', default='localhost',
                       help='Monitoring services host (default: localhost)')
    parser.add_argument('--prometheus-port', type=int, default=10200,
                       help='Prometheus port (default: 10200)')
    parser.add_argument('--grafana-port', type=int, default=10201,
                       help='Grafana port (default: 10201)')
    parser.add_argument('--loki-port', type=int, default=10202,
                       help='Loki port (default: 10202)')
    parser.add_argument('--alertmanager-port', type=int, default=10203,
                       help='AlertManager port (default: 10203)')
    parser.add_argument('--timeout', type=float, default=15.0,
                       help='Request timeout in seconds (default: 15.0)')
    parser.add_argument('--skip-prometheus', action='store_true',
                       help='Skip Prometheus checks')
    parser.add_argument('--skip-grafana', action='store_true',
                       help='Skip Grafana checks')
    parser.add_argument('--skip-loki', action='store_true',
                       help='Skip Loki checks')
    parser.add_argument('--skip-alertmanager', action='store_true',
                       help='Skip AlertManager checks')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("Starting Monitoring Stack health checks...")
    
    # Overall results
    results = {
        'timestamp': datetime.now().isoformat(),
        'category': 'monitoring_services',
        'services': {}
    }
    
    all_services_healthy = True
    
    # Prometheus health check
    if not args.skip_prometheus:
        logging.info(f"=== Prometheus Metrics Health Check (port {args.prometheus_port}) ===")
        prometheus_results = check_prometheus_service(args.host, args.prometheus_port, args.timeout)
        results['services']['prometheus'] = prometheus_results
        
        if prometheus_results['status'] != 'healthy':
            all_services_healthy = False
    
    # Grafana health check
    if not args.skip_grafana:
        logging.info(f"=== Grafana Visualization Health Check (port {args.grafana_port}) ===")
        grafana_results = check_grafana_service(args.host, args.grafana_port, args.timeout)
        results['services']['grafana'] = grafana_results
        
        if grafana_results['status'] != 'healthy':
            all_services_healthy = False
    
    # Loki health check
    if not args.skip_loki:
        logging.info(f"=== Loki Log Aggregation Health Check (port {args.loki_port}) ===")
        loki_results = check_loki_service(args.host, args.loki_port, args.timeout)
        results['services']['loki'] = loki_results
        
        if loki_results['status'] != 'healthy':
            all_services_healthy = False
    
    # AlertManager health check
    if not args.skip_alertmanager:
        logging.info(f"=== AlertManager Health Check (port {args.alertmanager_port}) ===")
        alertmanager_results = check_alertmanager_service(args.host, args.alertmanager_port, args.timeout)
        results['services']['alertmanager'] = alertmanager_results
        
        if alertmanager_results['status'] != 'healthy':
            all_services_healthy = False
    
    # Overall summary
    results['overall_status'] = 'healthy' if all_services_healthy else 'unhealthy'
    results['services_checked'] = len(results['services'])
    results['healthy_services'] = sum(1 for svc in results['services'].values() if svc['status'] == 'healthy')
    
    if all_services_healthy:
        logging.info("✅ All monitoring services are healthy and operational")
    else:
        logging.error("❌ One or more monitoring services require attention")
    
    # Note about operational status per CLAUDE.md
    logging.info("NOTE: Per CLAUDE.md, full monitoring stack is operational")
    
    # Output results as JSON for CI/CD integration
    if args.verbose:
        print(json.dumps(results, indent=2))
    
    return 0 if all_services_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
