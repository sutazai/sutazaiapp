#!/usr/bin/env python3
"""Debug health check process"""

import sys
sys.path.append('/opt/sutazaiapp/scripts/monitoring')

from static_monitor import EnhancedMonitor

def debug_health_check():
    """Debug individual health check steps"""
    print("=== Debugging Health Check Process ===\n")
    
    monitor = EnhancedMonitor()
    
    test_agents = [
        'ollama-integration-specialist',
        'senior-ai-engineer',
        'code-generation-improver'
    ]
    
    for agent_id in test_agents:
        print(f"=== {agent_id} ===")
        
        # Step 1: Check if deployed
        is_deployed = monitor._is_agent_deployed(agent_id)
        print(f"1. Is deployed: {is_deployed}")
        
        if not is_deployed:
            print("   → Would return 'offline' here")
            continue
        
        # Step 2: Get container info
        container_info = monitor._get_container_info(agent_id)
        print(f"2. Container info: {container_info}")
        
        if container_info:
            container_status = container_info['status']
            print(f"3. Container status: {container_status}")
            
            # Check what would happen for different status values
            if container_status == 'exited':
                print("   → Would return 'offline' (exited)")
                continue
            elif container_status == 'restarting':
                print("   → Would return 'warning' (restarting)")
                continue
            elif container_status == 'unhealthy':
                print("   → Would return 'critical' (unhealthy)")
                continue
            elif container_status == 'starting':
                print("   → Would return 'warning' (starting)")
                continue
            else:
                print(f"   → Status '{container_status}' - continuing to endpoint check")
        
        # Step 3: Get endpoint
        # We need to simulate agent_info for this
        agent_info = {'name': agent_id}  # Minimal agent info
        endpoint = monitor._get_agent_endpoint(agent_id, agent_info)
        print(f"4. Endpoint: {endpoint}")
        
        # Step 3b: Debug port extraction
        if container_info:
            ports = container_info.get('ports', [])
            print(f"   Container ports: {ports}")
            for port_mapping in ports:
                if '->' in port_mapping and 'tcp' in port_mapping:
                    try:
                        external_part = port_mapping.split('->')[0]
                        if ':' in external_part:
                            external_port = external_part.split(':')[-1]
                            print(f"   Extracted port: {external_port}")
                            test_connection = monitor._test_port_connection(int(external_port))
                            print(f"   Port {external_port} connectable: {test_connection}")
                    except Exception as e:
                        print(f"   Port extraction error: {e}")
        
        if not endpoint:
            if container_info and container_info['status'] in ['running', 'healthy']:
                print("   → Would return 'warning' (running but no endpoint)")
            else:
                print("   → Would return 'offline' (no endpoint)")
            continue
        
        # Step 4: Would attempt health check
        print(f"5. Would attempt health check at: {endpoint}")
        
        print()

if __name__ == "__main__":
    debug_health_check()