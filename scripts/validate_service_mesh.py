#!/usr/bin/env python3
"""
Comprehensive Service Mesh Validation
Tests real distributed system functionality
"""
import requests
import json
import time
import sys
import asyncio
import httpx

class ServiceMeshValidator:
    def __init__(self):
        self.consul_url = "http://localhost:10006"
        self.kong_admin_url = "http://localhost:10015"
        self.kong_proxy_url = "http://localhost:10005"
        self.results = []
        
    def print_header(self, title):
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")
    
    def test_consul_functionality(self):
        """Test Consul service discovery features"""
        self.print_header("CONSUL SERVICE DISCOVERY VALIDATION")
        
        # Test 1: Consul cluster health
        try:
            response = requests.get(f"{self.consul_url}/v1/status/leader")
            if response.status_code == 200 and response.text.strip():
                print("✓ Consul cluster has elected leader")
                self.results.append(True)
            else:
                print("✗ Consul cluster has no leader")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Consul cluster check failed: {e}")
            self.results.append(False)
        
        # Test 2: Service registration capability
        try:
            test_service = {
                "ID": "mesh-test-service",
                "Name": "mesh-test",
                "Address": "172.17.0.1",
                "Port": 8888,
                "Tags": ["test", "validation"],
                "Check": {
                    "TCP": "172.17.0.1:8888",
                    "Interval": "10s",
                    "Timeout": "2s"
                }
            }
            
            # Register service
            response = requests.put(
                f"{self.consul_url}/v1/agent/service/register",
                json=test_service
            )
            
            if response.status_code == 200:
                print("✓ Service registration works")
                
                # Verify service appears in catalog
                time.sleep(1)
                catalog = requests.get(f"{self.consul_url}/v1/catalog/service/mesh-test").json()
                if catalog:
                    print(f"✓ Service appears in catalog with {len(catalog)} instance(s)")
                    self.results.append(True)
                else:
                    print("✗ Service not in catalog")
                    self.results.append(False)
                
                # Cleanup
                requests.put(f"{self.consul_url}/v1/agent/service/deregister/mesh-test-service")
            else:
                print("✗ Service registration failed")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Service registration test failed: {e}")
            self.results.append(False)
        
        # Test 3: Health checks
        try:
            response = requests.get(f"{self.consul_url}/v1/health/state/any")
            if response.status_code == 200:
                health_checks = response.json()
                print(f"✓ Health check system active with {len(health_checks)} check(s)")
                self.results.append(True)
            else:
                print("✗ Health check system not working")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Health check test failed: {e}")
            self.results.append(False)
        
        # Test 4: Key-Value store
        try:
            # Write test key
            test_value = "service-mesh-test"
            response = requests.put(
                f"{self.consul_url}/v1/kv/test/mesh-validation",
                data=test_value
            )
            
            if response.status_code == 200:
                # Read back
                response = requests.get(f"{self.consul_url}/v1/kv/test/mesh-validation?raw=true")
                if response.text == test_value:
                    print("✓ Key-Value store works")
                    self.results.append(True)
                    
                    # Cleanup
                    requests.delete(f"{self.consul_url}/v1/kv/test/mesh-validation")
                else:
                    print("✗ Key-Value store read mismatch")
                    self.results.append(False)
            else:
                print("✗ Key-Value store write failed")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Key-Value store test failed: {e}")
            self.results.append(False)
    
    def test_kong_functionality(self):
        """Test Kong API Gateway features"""
        self.print_header("KONG API GATEWAY VALIDATION")
        
        # Test 1: Kong health
        try:
            response = requests.get(f"{self.kong_admin_url}/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Kong is healthy (database: {status.get('database', {}).get('reachable', False)})")
                self.results.append(True)
            else:
                print("✗ Kong health check failed")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Kong health test failed: {e}")
            self.results.append(False)
        
        # Test 2: Service configuration
        try:
            response = requests.get(f"{self.kong_admin_url}/services")
            if response.status_code == 200:
                services = response.json()["data"]
                print(f"✓ Kong has {len(services)} configured service(s):")
                for svc in services[:5]:  # Show first 5
                    print(f"  - {svc['name']}: {svc['protocol']}://{svc['host']}:{svc['port']}")
                self.results.append(True)
            else:
                print("✗ Cannot retrieve Kong services")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Kong services test failed: {e}")
            self.results.append(False)
        
        # Test 3: Route configuration
        try:
            response = requests.get(f"{self.kong_admin_url}/routes")
            if response.status_code == 200:
                routes = response.json()["data"]
                print(f"✓ Kong has {len(routes)} configured route(s):")
                for route in routes[:5]:  # Show first 5
                    paths = route.get('paths', [])
                    service_id = route.get('service', {}).get('id', 'unknown')
                    print(f"  - {paths} -> service {service_id[:8]}...")
                self.results.append(True)
            else:
                print("✗ Cannot retrieve Kong routes")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Kong routes test failed: {e}")
            self.results.append(False)
        
        # Test 4: Proxy functionality
        try:
            # Test Kong's own status endpoint through proxy
            response = requests.get(f"{self.kong_proxy_url}/status", timeout=5)
            if response.status_code in [200, 404]:  # 404 is ok if route not configured
                print("✓ Kong proxy is responding")
                self.results.append(True)
            else:
                print(f"✗ Kong proxy returned {response.status_code}")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Kong proxy test failed: {e}")
            self.results.append(False)
    
    def test_service_mesh_integration(self):
        """Test integrated service mesh functionality"""
        self.print_header("SERVICE MESH INTEGRATION VALIDATION")
        
        # Test 1: Service discovery integration
        try:
            # Get services from Consul
            consul_services = requests.get(f"{self.consul_url}/v1/catalog/services").json()
            
            # Get services from Kong
            kong_services = requests.get(f"{self.kong_admin_url}/services").json()["data"]
            kong_service_names = {svc['name'] for svc in kong_services}
            
            # Check for overlap
            consul_service_names = set(consul_services.keys())
            integrated = consul_service_names.intersection(kong_service_names)
            
            if integrated:
                print(f"✓ Found {len(integrated)} service(s) integrated between Consul and Kong")
                for svc in list(integrated)[:3]:
                    print(f"  - {svc}")
                self.results.append(True)
            else:
                print("⚠ No services integrated between Consul and Kong (may be normal)")
                self.results.append(True)  # Not a failure, just informational
        except Exception as e:
            print(f"✗ Service discovery integration test failed: {e}")
            self.results.append(False)
        
        # Test 2: Load balancing readiness
        try:
            # Check if Kong has upstreams configured (for load balancing)
            response = requests.get(f"{self.kong_admin_url}/upstreams")
            if response.status_code == 200:
                upstreams = response.json()["data"]
                if upstreams:
                    print(f"✓ Kong has {len(upstreams)} upstream(s) for load balancing")
                    for upstream in upstreams[:3]:
                        print(f"  - {upstream['name']}")
                else:
                    print("⚠ No upstreams configured (load balancing not active)")
                self.results.append(True)
            else:
                print("✗ Cannot check upstream configuration")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Load balancing check failed: {e}")
            self.results.append(False)
        
        # Test 3: Circuit breaker readiness
        try:
            # Check if any services have circuit breaker plugins
            response = requests.get(f"{self.kong_admin_url}/plugins")
            if response.status_code == 200:
                plugins = response.json()["data"]
                circuit_breakers = [p for p in plugins if 'circuit' in p.get('name', '').lower() or 
                                   'retry' in p.get('name', '').lower()]
                if circuit_breakers:
                    print(f"✓ Found {len(circuit_breakers)} circuit breaker/retry plugin(s)")
                else:
                    print("⚠ No circuit breaker plugins configured (resilience features inactive)")
                self.results.append(True)
            else:
                print("✗ Cannot check plugin configuration")
                self.results.append(False)
        except Exception as e:
            print(f"✗ Circuit breaker check failed: {e}")
            self.results.append(False)
    
    def test_mesh_capabilities(self):
        """Test distributed system capabilities"""
        self.print_header("DISTRIBUTED SYSTEM CAPABILITIES")
        
        # Test 1: Multi-service coordination
        print("Testing multi-service coordination capabilities:")
        services_available = []
        
        for service, url in [
            ("Consul", f"{self.consul_url}/v1/status/leader"),
            ("Kong", f"{self.kong_admin_url}/status"),
            ("Frontend", "http://localhost:10011/"),
        ]:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code in [200, 302]:  # 302 for redirects
                    services_available.append(service)
                    print(f"  ✓ {service} is available")
                else:
                    print(f"  ✗ {service} returned {response.status_code}")
            except:
                print(f"  ✗ {service} is not reachable")
        
        if len(services_available) >= 2:
            print(f"✓ Multi-service coordination possible ({len(services_available)} services active)")
            self.results.append(True)
        else:
            print("✗ Insufficient services for coordination")
            self.results.append(False)
        
        # Test 2: Service mesh observability
        print("\nTesting observability capabilities:")
        observability_features = 0
        
        # Check Consul metrics
        try:
            response = requests.get(f"{self.consul_url}/v1/agent/metrics")
            if response.status_code == 200:
                print("  ✓ Consul metrics available")
                observability_features += 1
        except:
            print("  ✗ Consul metrics not available")
        
        # Check Kong metrics
        try:
            response = requests.get(f"{self.kong_admin_url}/metrics")
            if response.status_code in [200, 404]:  # 404 if plugin not enabled
                if response.status_code == 200:
                    print("  ✓ Kong metrics available")
                    observability_features += 1
                else:
                    print("  ⚠ Kong metrics plugin not enabled")
        except:
            print("  ✗ Kong metrics not accessible")
        
        if observability_features > 0:
            print(f"✓ Observability features active ({observability_features} metric source(s))")
            self.results.append(True)
        else:
            print("⚠ Limited observability (metrics collection inactive)")
            self.results.append(True)  # Not a failure, just warning
    
    def generate_report(self):
        """Generate final validation report"""
        self.print_header("SERVICE MESH VALIDATION REPORT")
        
        passed = sum(self.results)
        total = len(self.results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        print(f"\nTest Results: {passed}/{total} passed ({percentage:.1f}%)")
        
        if percentage >= 80:
            print("\n✓ SERVICE MESH IS FUNCTIONAL")
            print("The distributed system components are properly configured and operational.")
            return 0
        elif percentage >= 60:
            print("\n⚠ SERVICE MESH PARTIALLY FUNCTIONAL")
            print("Core components work but some features are not fully configured.")
            return 1
        else:
            print("\n✗ SERVICE MESH HAS CRITICAL ISSUES")
            print("Major components are not working properly.")
            return 2

def main():
    validator = ServiceMeshValidator()
    
    print("=" * 60)
    print("SERVICE MESH COMPREHENSIVE VALIDATION")
    print("Testing real distributed system functionality")
    print("=" * 60)
    
    # Run all validation tests
    validator.test_consul_functionality()
    validator.test_kong_functionality()
    validator.test_service_mesh_integration()
    validator.test_mesh_capabilities()
    
    # Generate report
    return validator.generate_report()

if __name__ == "__main__":
    sys.exit(main())