#!/usr/bin/env python3
"""
Real Service Mesh Functionality Test
Tests actual integration between Consul, Kong, and backend services
"""
import requests
import json
import time
import sys

def test_consul_health():
    """Test if Consul is healthy and accessible"""
    try:
        response = requests.get("http://localhost:10006/v1/status/leader")
        if response.status_code == 200:
            print("✓ Consul is healthy and has a leader")
            return True
        else:
            print(f"✗ Consul health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Consul: {e}")
        return False

def test_kong_health():
    """Test if Kong API Gateway is healthy"""
    try:
        response = requests.get("http://localhost:10015/status")
        if response.status_code == 200:
            print("✓ Kong API Gateway is healthy")
            return True
        else:
            print(f"✗ Kong health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Kong: {e}")
        return False

def test_service_registration():
    """Test if services can register with Consul"""
    try:
        # Register a test service
        service_data = {
            "ID": "test-service-1",
            "Name": "test-service",
            "Address": "localhost",
            "Port": 9999,
            "Tags": ["test", "mesh"],
            "Check": {
                "HTTP": "http://localhost:9999/health",
                "Interval": "10s",
                "DeregisterCriticalServiceAfter": "1m"
            }
        }
        
        response = requests.put(
            "http://localhost:10006/v1/agent/service/register",
            json=service_data
        )
        
        if response.status_code == 200:
            print("✓ Successfully registered test service with Consul")
            
            # Verify registration
            services = requests.get("http://localhost:10006/v1/agent/services").json()
            if "test-service-1" in services:
                print("✓ Service appears in Consul registry")
                
                # Cleanup
                requests.put("http://localhost:10006/v1/agent/service/deregister/test-service-1")
                return True
            else:
                print("✗ Service not found in Consul registry")
                return False
        else:
            print(f"✗ Service registration failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Service registration error: {e}")
        return False

def test_service_discovery():
    """Test if services can be discovered through Consul"""
    try:
        # Get all registered services
        response = requests.get("http://localhost:10006/v1/catalog/services")
        
        if response.status_code == 200:
            services = response.json()
            print(f"✓ Found {len(services)} services in Consul catalog")
            
            for service_name in services:
                # Get service instances
                instances = requests.get(
                    f"http://localhost:10006/v1/catalog/service/{service_name}"
                ).json()
                print(f"  - {service_name}: {len(instances)} instance(s)")
            
            return True
        else:
            print(f"✗ Service discovery failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Service discovery error: {e}")
        return False

def test_kong_routes():
    """Test if Kong has any configured routes"""
    try:
        response = requests.get("http://localhost:10015/routes")
        
        if response.status_code == 200:
            routes = response.json()["data"]
            print(f"✓ Kong has {len(routes)} configured route(s)")
            
            for route in routes:
                print(f"  - Route: {route.get('name', 'unnamed')} -> {route.get('paths', [])}")
            
            return True
        else:
            print(f"✗ Kong routes query failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Kong routes error: {e}")
        return False

def test_kong_services():
    """Test if Kong has any configured services"""
    try:
        response = requests.get("http://localhost:10015/services")
        
        if response.status_code == 200:
            services = response.json()["data"]
            print(f"✓ Kong has {len(services)} configured service(s)")
            
            for service in services:
                print(f"  - Service: {service.get('name', 'unnamed')} -> {service.get('host', 'unknown')}")
            
            return True
        else:
            print(f"✗ Kong services query failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Kong services error: {e}")
        return False

def test_backend_health():
    """Test if backend service is running"""
    try:
        response = requests.get("http://localhost:10010/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Backend is healthy: {data}")
            return True
        else:
            print(f"✗ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("✗ Backend health check timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend service")
        return False
    except Exception as e:
        print(f"✗ Backend health check error: {e}")
        return False

def main():
    """Run all service mesh tests"""
    print("=" * 60)
    print("SERVICE MESH FUNCTIONALITY TEST")
    print("=" * 60)
    
    results = []
    
    # Test each component
    print("\n1. Testing Consul Service Discovery...")
    results.append(test_consul_health())
    
    print("\n2. Testing Kong API Gateway...")
    results.append(test_kong_health())
    
    print("\n3. Testing Backend Service...")
    results.append(test_backend_health())
    
    print("\n4. Testing Service Registration...")
    results.append(test_service_registration())
    
    print("\n5. Testing Service Discovery...")
    results.append(test_service_discovery())
    
    print("\n6. Testing Kong Routes...")
    results.append(test_kong_routes())
    
    print("\n7. Testing Kong Services...")
    results.append(test_kong_services())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed - Service mesh is functional!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed - Service mesh has issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())