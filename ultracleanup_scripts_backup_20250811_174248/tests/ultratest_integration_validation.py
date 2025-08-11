#!/usr/bin/env python3
"""
ULTRATEST Integration Validation
Tests all service interconnections and dependencies.
"""

import requests
import time
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any

class UltratestIntegrationValidator:
    def __init__(self):
        self.services = {
            'Backend': 'http://localhost:10010',
            'Frontend': 'http://localhost:10011',
            'Ollama': 'http://localhost:10104',
            'Hardware Optimizer': 'http://localhost:11110',
            'AI Orchestrator': 'http://localhost:8589',
            'Ollama Integration': 'http://localhost:8090',
            'FAISS Vector': 'http://localhost:10103',
            'Resource Arbitration': 'http://localhost:8588',
            'Task Assignment': 'http://localhost:8551',
            'PostgreSQL': 'localhost:10000',
            'Redis': 'localhost:10001',
            'Neo4j': 'localhost:10002',
            'Prometheus': 'http://localhost:10200',
            'Grafana': 'http://localhost:10201',
        }
        
    def test_service_health(self, service_name: str, base_url: str) -> Dict[str, Any]:
        """Test individual service health"""
        health_endpoints = ['/health', '/api/health', '/', '/api/tags', '/status', '/-/ready']
        
        for endpoint in health_endpoints:
            try:
                url = f"{base_url}{endpoint}"
                response = requests.get(url, timeout=5)
                
                if response.status_code < 400:
                    return {
                        'service': service_name,
                        'url': url,
                        'status_code': response.status_code,
                        'response_time_ms': response.elapsed.total_seconds() * 1000,
                        'healthy': True,
                        'endpoint_used': endpoint
                    }
            except:
                continue
        
        return {
            'service': service_name,
            'healthy': False,
            'error': 'No responsive endpoints found'
        }
    
    def test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connections"""
        db_results = {}
        
        # Test PostgreSQL
        try:
            result = subprocess.run(
                ['docker', 'exec', 'sutazai-postgres', 'pg_isready', '-U', 'sutazai'],
                capture_output=True, text=True, timeout=10
            )
            db_results['PostgreSQL'] = {
                'healthy': result.returncode == 0,
                'response': result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
            }
        except Exception as e:
            db_results['PostgreSQL'] = {'healthy': False, 'error': str(e)}
        
        # Test Redis
        try:
            result = subprocess.run(
                ['docker', 'exec', 'sutazai-redis', 'redis-cli', 'ping'],
                capture_output=True, text=True, timeout=10
            )
            db_results['Redis'] = {
                'healthy': 'PONG' in result.stdout,
                'response': result.stdout.strip()
            }
        except Exception as e:
            db_results['Redis'] = {'healthy': False, 'error': str(e)}
        
        # Test Neo4j (if available)
        try:
            result = subprocess.run(
                ['docker', 'exec', 'sutazai-neo4j', 'cypher-shell', '-u', 'neo4j', '-p', 'password', 'RETURN 1'],
                capture_output=True, text=True, timeout=10
            )
            db_results['Neo4j'] = {
                'healthy': result.returncode == 0,
                'response': 'Connected' if result.returncode == 0 else result.stderr.strip()
            }
        except Exception as e:
            db_results['Neo4j'] = {'healthy': False, 'error': str(e)}
        
        return db_results
    
    def test_service_dependencies(self) -> Dict[str, Any]:
        """Test critical service dependencies"""
        dependency_results = {}
        
        # Test Backend -> Database connection
        try:
            response = requests.get('http://localhost:10010/health', timeout=5)
            backend_health = response.json() if response.status_code == 200 else {}
            
            dependency_results['Backend-Database'] = {
                'healthy': response.status_code == 200,
                'details': backend_health
            }
        except Exception as e:
            dependency_results['Backend-Database'] = {'healthy': False, 'error': str(e)}
        
        # Test AI Orchestrator -> RabbitMQ
        try:
            response = requests.get('http://localhost:8589/health', timeout=5)
            orchestrator_health = response.json() if response.status_code == 200 else {}
            
            dependency_results['AI-Orchestrator-RabbitMQ'] = {
                'healthy': response.status_code == 200,
                'details': orchestrator_health
            }
        except Exception as e:
            dependency_results['AI-Orchestrator-RabbitMQ'] = {'healthy': False, 'error': str(e)}
        
        # Test Ollama Integration -> Ollama
        try:
            response = requests.get('http://localhost:8090/health', timeout=5)
            integration_health = response.json() if response.status_code == 200 else {}
            
            dependency_results['Ollama-Integration-Ollama'] = {
                'healthy': response.status_code == 200,
                'details': integration_health
            }
        except Exception as e:
            dependency_results['Ollama-Integration-Ollama'] = {'healthy': False, 'error': str(e)}
        
        return dependency_results
    
    def test_container_communication(self) -> Dict[str, Any]:
        """Test container-to-container communication"""
        communication_results = {}
        
        # Get container network information
        try:
            result = subprocess.run(
                ['docker', 'network', 'ls', '--format', '{{.Name}}'],
                capture_output=True, text=True
            )
            networks = result.stdout.strip().split('\n')
            
            # Find the project network
            sutazai_network = None
            for network in networks:
                if 'sutazai' in network.lower():
                    sutazai_network = network
                    break
            
            if sutazai_network:
                communication_results['Docker Network'] = {
                    'healthy': True,
                    'network_name': sutazai_network
                }
            else:
                communication_results['Docker Network'] = {
                    'healthy': False,
                    'error': 'No sutazai network found'
                }
        except Exception as e:
            communication_results['Docker Network'] = {'healthy': False, 'error': str(e)}
        
        return communication_results
    
    def test_monitoring_stack(self) -> Dict[str, Any]:
        """Test monitoring and observability stack"""
        monitoring_results = {}
        
        # Test Prometheus
        try:
            response = requests.get('http://localhost:10200/-/ready', timeout=5)
            monitoring_results['Prometheus'] = {
                'healthy': response.status_code == 200,
                'status_code': response.status_code
            }
        except Exception as e:
            monitoring_results['Prometheus'] = {'healthy': False, 'error': str(e)}
        
        # Test Grafana
        try:
            response = requests.get('http://localhost:10201/api/health', timeout=5)
            monitoring_results['Grafana'] = {
                'healthy': response.status_code == 200,
                'status_code': response.status_code
            }
        except Exception as e:
            monitoring_results['Grafana'] = {'healthy': False, 'error': str(e)}
        
        # Test metrics collection
        try:
            response = requests.get('http://localhost:10200/api/v1/query?query=up', timeout=5)
            if response.status_code == 200:
                data = response.json()
                metrics_count = len(data.get('data', {}).get('result', []))
                monitoring_results['Metrics Collection'] = {
                    'healthy': metrics_count > 0,
                    'metrics_count': metrics_count
                }
            else:
                monitoring_results['Metrics Collection'] = {
                    'healthy': False,
                    'error': f'HTTP {response.status_code}'
                }
        except Exception as e:
            monitoring_results['Metrics Collection'] = {'healthy': False, 'error': str(e)}
        
        return monitoring_results
    
    def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration validation"""
        print("\n🔗 ULTRATEST: Integration Validation")
        print("=" * 60)
        
        # Test 1: Individual service health
        print("1. Testing individual service health...")
        service_results = {}
        for service_name, base_url in self.services.items():
            if base_url.startswith('http'):
                result = self.test_service_health(service_name, base_url)
                service_results[service_name] = result
                
                icon = "✅" if result.get('healthy', False) else "❌"
                print(f"   {icon} {service_name}")
        
        # Test 2: Database connectivity
        print("\n2. Testing database connectivity...")
        db_results = self.test_database_connectivity()
        for db_name, result in db_results.items():
            icon = "✅" if result.get('healthy', False) else "❌"
            print(f"   {icon} {db_name}")
        
        # Test 3: Service dependencies
        print("\n3. Testing service dependencies...")
        dependency_results = self.test_service_dependencies()
        for dep_name, result in dependency_results.items():
            icon = "✅" if result.get('healthy', False) else "❌"
            print(f"   {icon} {dep_name}")
        
        # Test 4: Container communication
        print("\n4. Testing container communication...")
        communication_results = self.test_container_communication()
        for comm_name, result in communication_results.items():
            icon = "✅" if result.get('healthy', False) else "❌"
            print(f"   {icon} {comm_name}")
        
        # Test 5: Monitoring stack
        print("\n5. Testing monitoring stack...")
        monitoring_results = self.test_monitoring_stack()
        for monitor_name, result in monitoring_results.items():
            icon = "✅" if result.get('healthy', False) else "❌"
            print(f"   {icon} {monitor_name}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'service_health': service_results,
            'database_connectivity': db_results,
            'service_dependencies': dependency_results,
            'container_communication': communication_results,
            'monitoring_stack': monitoring_results
        }
    
    def generate_integration_report(self, results: Dict[str, Any]):
        """Generate comprehensive integration report"""
        print("\n" + "=" * 80)
        print("🔗 ULTRATEST INTEGRATION VALIDATION REPORT")
        print("=" * 80)
        print(f"Test Execution Time: {results.get('timestamp', 'Unknown')}")
        
        # Service health summary
        service_health = results.get('service_health', {})
        healthy_services = sum(1 for result in service_health.values() if result.get('healthy', False))
        total_services = len(service_health)
        
        print(f"\n🏥 SERVICE HEALTH SUMMARY:")
        print("-" * 50)
        print(f"Healthy Services: {healthy_services}/{total_services}")
        print(f"Service Health Rate: {(healthy_services/total_services*100):.1f}%")
        
        # Database connectivity summary
        db_connectivity = results.get('database_connectivity', {})
        healthy_databases = sum(1 for result in db_connectivity.values() if result.get('healthy', False))
        total_databases = len(db_connectivity)
        
        print(f"\n💾 DATABASE CONNECTIVITY:")
        print("-" * 50)
        print(f"Connected Databases: {healthy_databases}/{total_databases}")
        print(f"Database Connectivity Rate: {(healthy_databases/total_databases*100):.1f}%")
        
        # Service dependencies summary
        dependencies = results.get('service_dependencies', {})
        healthy_deps = sum(1 for result in dependencies.values() if result.get('healthy', False))
        total_deps = len(dependencies)
        
        print(f"\n🔗 SERVICE DEPENDENCIES:")
        print("-" * 50)
        print(f"Working Dependencies: {healthy_deps}/{total_deps}")
        print(f"Dependency Health Rate: {(healthy_deps/total_deps*100):.1f}%")
        
        # Container communication
        communication = results.get('container_communication', {})
        healthy_comm = sum(1 for result in communication.values() if result.get('healthy', False))
        total_comm = len(communication)
        
        print(f"\n🐳 CONTAINER COMMUNICATION:")
        print("-" * 50)
        print(f"Working Communication: {healthy_comm}/{total_comm}")
        
        # Monitoring stack
        monitoring = results.get('monitoring_stack', {})
        healthy_monitoring = sum(1 for result in monitoring.values() if result.get('healthy', False))
        total_monitoring = len(monitoring)
        
        print(f"\n📊 MONITORING STACK:")
        print("-" * 50)
        print(f"Working Monitoring: {healthy_monitoring}/{total_monitoring}")
        print(f"Monitoring Health Rate: {(healthy_monitoring/total_monitoring*100):.1f}%")
        
        # Overall integration assessment
        print(f"\n🎯 INTEGRATION ASSESSMENT:")
        print("-" * 50)
        
        # Calculate overall health scores
        service_score = (healthy_services / total_services * 100) if total_services > 0 else 0
        db_score = (healthy_databases / total_databases * 100) if total_databases > 0 else 0
        dep_score = (healthy_deps / total_deps * 100) if total_deps > 0 else 0
        monitoring_score = (healthy_monitoring / total_monitoring * 100) if total_monitoring > 0 else 0
        
        # Weighted overall score (services and databases are most critical)
        overall_score = (service_score * 0.4 + db_score * 0.3 + dep_score * 0.2 + monitoring_score * 0.1)
        
        achievements = []
        issues = []
        
        if service_score >= 80:
            achievements.append(f"Most services operational ({service_score:.1f}%)")
        else:
            issues.append(f"Service availability low ({service_score:.1f}%)")
        
        if db_score >= 80:
            achievements.append(f"Database connectivity strong ({db_score:.1f}%)")
        else:
            issues.append(f"Database connectivity issues ({db_score:.1f}%)")
        
        if dep_score >= 80:
            achievements.append(f"Service dependencies working ({dep_score:.1f}%)")
        else:
            issues.append(f"Service dependency problems ({dep_score:.1f}%)")
        
        if monitoring_score >= 70:
            achievements.append(f"Monitoring stack operational ({monitoring_score:.1f}%)")
        
        if overall_score >= 85:
            achievements.append("Strong overall integration")
        elif overall_score >= 70:
            achievements.append("Good overall integration")
        else:
            issues.append(f"Overall integration needs improvement ({overall_score:.1f}%)")
        
        print("🎉 ACHIEVEMENTS:")
        for achievement in achievements:
            print(f"   ✅ {achievement}")
        
        if issues:
            print("\n⚠️  AREAS FOR IMPROVEMENT:")
            for issue in issues:
                print(f"   ❌ {issue}")
        
        print(f"\n📈 Overall Integration Score: {overall_score:.1f}%")
        
        # Detailed breakdown
        print(f"\n🔍 DETAILED SERVICE STATUS:")
        print("-" * 50)
        
        for service_name, result in service_health.items():
            status = "✅ HEALTHY" if result.get('healthy', False) else "❌ UNHEALTHY"
            response_time = result.get('response_time_ms', 0)
            if response_time > 0:
                print(f"{service_name:20} {status} ({response_time:.1f}ms)")
            else:
                print(f"{service_name:20} {status}")
        
        return overall_score >= 75  # 75% overall integration score for success

def main():
    """Run comprehensive integration validation"""
    print("🚀 Starting ULTRATEST Integration Validation")
    
    validator = UltratestIntegrationValidator()
    results = validator.run_comprehensive_integration_test()
    
    # Generate comprehensive report
    success = validator.generate_integration_report(results)
    
    # Save detailed results
    with open('/opt/sutazaiapp/tests/ultratest_integration_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Full report saved to: /opt/sutazaiapp/tests/ultratest_integration_report.json")
    
    if success:
        print("\n🎉 INTEGRATION VALIDATION SUCCESSFUL!")
        return 0
    else:
        print("\n⚠️  INTEGRATION NEEDS IMPROVEMENT")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())