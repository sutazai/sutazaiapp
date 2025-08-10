#!/usr/bin/env python3
"""
Debug hardware-resource-optimizer detection issues
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import sys
import json
import time
import requests
import socket
from pathlib import Path

# Add monitoring directory to path
sys.path.insert(0, str(Path(__file__).parent))

from static_monitor import EnhancedMonitor

def debug_endpoint_detection():
    """Debug the endpoint detection process step by step"""
    print("=== DEBUGGING ENDPOINT DETECTION ===")
    
    monitor = EnhancedMonitor()
    agent_id = 'hardware-resource-optimizer'
    agent_info = monitor.agent_registry['agents'][agent_id]
    
    print(f"Agent ID: {agent_id}")
    print(f"Agent Info: {agent_info}")
    
    # Step 1: Check if agent is deployed
    is_deployed = monitor._is_agent_deployed(agent_id)
    print(f"Is deployed: {is_deployed}")
    
    # Step 2: Test port connection for 8116
    port_open = monitor._test_port_connection(8116)
    print(f"Port 8116 open: {port_open}")
    
    # Step 3: Test endpoint verification
    endpoint = "http://localhost:8116"
    endpoint_verified = monitor._verify_agent_endpoint(endpoint, agent_id)
    print(f"Endpoint verified: {endpoint_verified}")
    
    # Step 4: Get endpoint through normal process
    detected_endpoint = monitor._get_agent_endpoint(agent_id, agent_info)
    print(f"Detected endpoint: {detected_endpoint}")
    
    # Step 5: Test health check with various timeouts
    for timeout in [1, 2, 3, 5]:
        print(f"\nTesting health check with {timeout}s timeout:")
        health_status, response_time = monitor._check_agent_health(agent_id, agent_info, timeout)
        print(f"  Status: {health_status}, Response Time: {response_time}ms")
    
    monitor.cleanup()

def debug_health_paths():
    """Debug which health paths work"""
    print("\n=== DEBUGGING HEALTH PATHS ===")
    
    base_url = "http://localhost:8116"
    health_paths = ['/health', '/status', '/ping', '/api/health', '/heartbeat', '/']
    
    for path in health_paths:
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}{path}", timeout=3)
            response_time = (time.time() - start_time) * 1000
            
            print(f"Path {path:12}: Status {response.status_code}, Time: {response_time:.0f}ms")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"             JSON response: {json.dumps(data, indent=2)[:100]}...")
                except Exception as e:
                    # TODO: Review this exception handling
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    print(f"             Text response: {response.text[:100]}...")
        except Exception as e:
            print(f"Path {path:12}: ERROR - {e}")

def debug_multiple_calls():
    """Test multiple consecutive calls to see consistency"""
    print("\n=== DEBUGGING MULTIPLE CALLS ===")
    
    monitor = EnhancedMonitor()
    agent_id = 'hardware-resource-optimizer'
    agent_info = monitor.agent_registry['agents'][agent_id]
    
    results = []
    for i in range(5):
        print(f"Call {i+1}:")
        health_status, response_time = monitor._check_agent_health(agent_id, agent_info, 3)
        results.append((health_status, response_time))
        print(f"  Status: {health_status}, Response Time: {response_time}ms")
        time.sleep(1)  # Wait between calls
    
    # Analyze consistency
    statuses = [r[0] for r in results]
    unique_statuses = set(statuses)
    
    print(f"\nConsistency Analysis:")
    print(f"Unique statuses: {unique_statuses}")
    print(f"Status distribution: {dict(zip(*zip(*[(s, statuses.count(s)) for s in unique_statuses])))}")
    
    monitor.cleanup()

def test_agent_type_detection():
    """Test agent type detection for hardware-resource-optimizer"""
    print("\n=== DEBUGGING AGENT TYPE DETECTION ===")
    
    monitor = EnhancedMonitor()
    agent_id = 'hardware-resource-optimizer'
    agent_info = monitor.agent_registry['agents'][agent_id]
    
    agent_type = monitor._get_agent_type(agent_info)
    print(f"Detected agent type: {agent_type}")
    
    # Check port ranges for this type
    port_ranges = monitor._get_port_ranges_by_type(agent_type)
    print(f"Port ranges for type {agent_type}: {port_ranges}")
    
    # Check if 8116 is in the ranges
    port_8116_in_ranges = any(8116 in range(start, end) for start_list in port_ranges for start, end in [start_list] if isinstance(start_list, list) for start, end in [(start_list[0], start_list[-1] + 1)])
    print(f"Port 8116 in ranges: {port_8116_in_ranges}")
    
    monitor.cleanup()

if __name__ == "__main__":
    debug_endpoint_detection()
    debug_health_paths()
    debug_multiple_calls()
    test_agent_type_detection()