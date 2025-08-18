#!/usr/bin/env python3
"""
Fix Backend Mesh Startup Script
Purpose: Ensure backend starts with proper mesh integration
Created: 2025-08-17 UTC
"""
import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path

# Add backend to path
sys.path.insert(0, '/opt/sutazaiapp/backend')

def check_service_health(service_name, port):
    """Check if a service is healthy"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code in [200, 204]
    except:
        return False

def start_backend_with_mesh():
    """Start backend with mesh integration enabled"""
    print("Starting backend with mesh integration...")
    
    # Set environment variables for mesh
    env = os.environ.copy()
    env.update({
        "MESH_ENABLED": "true",
        "CONSUL_HOST": "localhost",
        "CONSUL_PORT": "10006",
        "ENABLE_DIND_BRIDGE": "true",
        "MCP_ORCHESTRATOR_URL": "http://localhost:12375",
        "PYTHONPATH": "/opt/sutazaiapp/backend"
    })
    
    # Check if backend is already running
    if check_service_health("backend", 10010):
        print("✓ Backend already running")
        return True
    
    # Start backend using docker-compose
    print("Starting backend container...")
    result = subprocess.run(
        ["docker-compose", "-f", "/opt/sutazaiapp/docker/docker-compose.consolidated.yml", 
         "up", "-d", "backend"],
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        print(f"Error starting backend: {result.stderr}")
        
        # Try starting manually
        print("Attempting manual backend start...")
        backend_cmd = [
            "python3", "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "10010",
            "--reload"
        ]
        
        # Start in background
        process = subprocess.Popen(
            backend_cmd,
            cwd="/opt/sutazaiapp/backend",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"Backend process started with PID: {process.pid}")
        
        # Save PID for later management
        with open("/tmp/backend.pid", "w") as f:
            f.write(str(process.pid))
    
    # Wait for backend to be healthy
    print("Waiting for backend to become healthy...")
    max_attempts = 30
    for i in range(max_attempts):
        if check_service_health("backend", 10010):
            print("✓ Backend is healthy")
            return True
        time.sleep(2)
        print(f"  Attempt {i+1}/{max_attempts}...")
    
    print("✗ Backend failed to start")
    return False

def initialize_mesh_in_backend():
    """Initialize mesh components in the backend"""
    print("Initializing mesh in backend...")
    
    # Create initialization request
    init_request = {
        "enable_mesh": True,
        "enable_dind_bridge": True,
        "consul_host": "localhost",
        "consul_port": 10006,
        "register_services": True
    }
    
    try:
        # Initialize mesh via API
        response = requests.post(
            "http://localhost:10010/api/v1/mesh/initialize",
            json=init_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Mesh initialized: {result}")
            return True
        else:
            print(f"✗ Mesh initialization failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Error initializing mesh: {e}")
    
    return False

def register_backend_with_consul():
    """Register backend service with Consul"""
    print("Registering backend with Consul...")
    
    service_definition = {
        "ID": "backend-api-10010",
        "Name": "backend-api",
        "Tags": ["api", "backend", "fastapi", "mesh-enabled"],
        "Address": "localhost",
        "Port": 10010,
        "Check": {
            "HTTP": "http://localhost:10010/health",
            "Interval": "10s",
            "Timeout": "5s",
            "DeregisterCriticalServiceAfter": "1m"
        }
    }
    
    try:
        response = requests.put(
            "http://localhost:10006/v1/agent/service/register",
            json=service_definition
        )
        
        if response.status_code == 200:
            print("✓ Backend registered with Consul")
            return True
        else:
            print(f"✗ Registration failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Error registering with Consul: {e}")
    
    return False

def verify_mesh_endpoints():
    """Verify mesh endpoints are working"""
    print("Verifying mesh endpoints...")
    
    endpoints = [
        ("/api/v1/mesh/status", "Mesh status"),
        ("/api/v1/mesh/services", "Service list"),
        ("/api/v1/mesh/dind/status", "DinD bridge status"),
        ("/api/v1/mcp/status", "MCP status")
    ]
    
    working = 0
    failed = 0
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:10010{endpoint}", timeout=5)
            if response.status_code in [200, 204]:
                print(f"  ✓ {description}: Working")
                working += 1
            else:
                print(f"  ✗ {description}: HTTP {response.status_code}")
                failed += 1
        except Exception as e:
            print(f"  ✗ {description}: {e}")
            failed += 1
    
    print(f"\nEndpoint Summary: {working} working, {failed} failed")
    return failed == 0

def main():
    """Main execution"""
    print("="*50)
    print("BACKEND MESH INTEGRATION FIX")
    print("="*50)
    print()
    
    # Step 1: Start backend
    if not start_backend_with_mesh():
        print("Failed to start backend. Exiting.")
        sys.exit(1)
    
    # Step 2: Initialize mesh
    if not initialize_mesh_in_backend():
        print("Warning: Mesh initialization via API failed")
        print("Mesh may initialize automatically on first request")
    
    # Step 3: Register with Consul
    if not register_backend_with_consul():
        print("Warning: Consul registration failed")
    
    # Step 4: Verify endpoints
    if verify_mesh_endpoints():
        print("\n✓ Backend mesh integration successful!")
    else:
        print("\n⚠ Some mesh endpoints not working")
        print("Check logs: docker logs sutazai-backend")
    
    # Show final status
    print("\n" + "="*50)
    print("FINAL STATUS")
    print("="*50)
    
    try:
        response = requests.get("http://localhost:10010/api/v1/mesh/status")
        if response.status_code == 200:
            status = response.json()
            print(json.dumps(status, indent=2))
        else:
            print("Could not retrieve mesh status")
    except Exception as e:
        print(f"Error getting status: {e}")

if __name__ == "__main__":
    main()