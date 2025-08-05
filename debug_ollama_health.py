#!/usr/bin/env python3
"""Debug Ollama health check specifically"""

import sys
sys.path.append('/opt/sutazaiapp/scripts/monitoring')

from static_monitor import EnhancedMonitor

def debug_ollama():
    """Debug Ollama health check step by step"""
    print("=== Debugging Ollama Health Check ===\n")
    
    monitor = EnhancedMonitor()
    agent_id = 'ollama-integration-specialist'
    agent_info = {'name': agent_id}
    
    print(f"Agent ID: {agent_id}")
    
    # Step 1: Deployment check
    is_deployed = monitor._is_agent_deployed(agent_id)
    print(f"1. Is deployed: {is_deployed}")
    
    if not is_deployed:
        print("   → Would return 'offline' here")
        return
    
    # Step 2: Container info
    container_info = monitor._get_container_info(agent_id)
    print(f"2. Container info: {container_info}")
    
    if container_info:
        container_status = container_info['status']
        print(f"3. Container status: {container_status}")
        
        # Check early returns
        if container_status == 'exited':
            print("   → Would return 'offline' (exited)")
            return
        elif container_status == 'restarting':
            print("   → Would return 'warning' (restarting)")
            return
        elif container_status == 'unhealthy':
            print("   → Would return 'critical' (unhealthy)")
            return
        elif container_status == 'starting':
            print("   → Would return 'warning' (starting)")
            return
        else:
            print(f"   → Status '{container_status}' - continuing to endpoint check")
    
    # Step 3: Endpoint detection
    endpoint = monitor._get_agent_endpoint(agent_id, agent_info)
    print(f"4. Endpoint: {endpoint}")
    
    if not endpoint:
        if container_info and container_info['status'] in ['running', 'healthy']:
            print("   → Would return 'warning' (running but no endpoint)")
        else:
            print("   → Would return 'offline' (no endpoint)")
        return
    
    # Step 4: Health check simulation
    print(f"5. Would attempt health check at: {endpoint}")
    
    # Test the actual health check logic
    import requests
    import time
    
    health_paths = ['/health', '/status', '/ping', '/api/health', '/heartbeat']
    
    for path in health_paths:
        try:
            print(f"   Testing {endpoint}{path}")
            response = requests.get(f"{endpoint}{path}", timeout=2)
            print(f"     Status: {response.status_code}")
            if response.status_code in [200, 201, 204]:
                print(f"     → Would return 'healthy' from path {path}")
                return
            elif response.status_code in [400, 401, 403, 405]:
                print(f"     → Service running but no health endpoint")
                print(f"     → Would return 'healthy' from path {path}")
                return
        except requests.exceptions.ConnectionError as e:
            print(f"     Connection error: {e}")
        except Exception as e:
            print(f"     Error: {e}")
    
    # Step 5: Fallback logic
    print("6. All health paths failed - checking fallback logic")
    
    if container_info and container_info['status'] in ['running', 'healthy']:
        container_name = container_info['name']
        print(f"   Container name: {container_name}")
        
        # Service containers that should have HTTP endpoints
        service_containers = ['sutazai-ollama', 'sutazai-backend', 'sutazai-postgres', 
                            'sutazai-redis', 'sutazai-chromadb', 'sutazai-qdrant', 'sutazai-neo4j']
        
        is_service_container = any(service in container_name for service in service_containers)
        print(f"   Is service container: {is_service_container}")
        
        if is_service_container:
            print("   → Service container should respond to HTTP - would return 'offline'")
        else:
            print("   → Agent container without HTTP endpoint - would return 'healthy'")
    else:
        print("   → No running container - would return 'offline'")

if __name__ == "__main__":
    debug_ollama()