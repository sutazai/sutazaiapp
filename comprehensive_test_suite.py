#!/usr/bin/env python3
"""
Comprehensive Testing Suite for SutazAI System
Testing QA Validator - AI-Powered Validation Framework
"""
import requests
import redis
import psycopg2
import json
import time
import subprocess
import asyncio
import httpx
from datetime import datetime
from typing import Dict, List, Any, Tuple

class ComprehensiveTestSuite:
    def __init__(self):
        self.test_results = {
            'api_endpoints': {},
            'database': {},
            'redis': {},
            'ollama': {},
            'frontend': {},
            'agents': {},
            'integration': {},
            'performance': {},
            'security': {}
        }
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:8501"
        self.ollama_url = "http://localhost:11434"
        
        # Test configurations
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'sutazai_db',
            'user': 'sutazai',
            'password': 'sutazai123'
        }
        
        self.redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }

    def log_test(self, category: str, test_name: str, status: str, details: str = "", duration: float = 0.0):
        """Log test result"""
        self.test_results[category][test_name] = {
            'status': status,
            'details': details,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        print(f"[{status.upper()}] {category}.{test_name}: {details[:100]}")

    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test all API endpoints functionality"""
        print("\n🔍 Testing API Endpoints...")
        
        endpoints = [
            ("GET", "/", "Root endpoint"),
            ("GET", "/health", "Health check"),
            ("GET", "/agents", "Agents list"),
            ("GET", "/models", "Available models"),
            ("GET", "/metrics", "System metrics"),
            ("GET", "/public/metrics", "Public metrics"),
            ("POST", "/simple-chat", "Simple chat", {"message": "Hello"}),
            ("POST", "/chat", "Enhanced chat", {"message": "Hello", "model": "tinyllama"}),
            ("POST", "/think", "AI thinking", {"query": "What is 2+2?"}),
            ("POST", "/reason", "AI reasoning", {"type": "deductive", "description": "Calculate 2+2"}),
            ("POST", "/execute", "Task execution", {"description": "Simple test task", "type": "test"}),
            ("POST", "/learn", "Knowledge learning", {"content": "Test content", "type": "text"}),
            ("POST", "/improve", "Self improvement"),
            ("GET", "/api/v1/system/status", "System status"),
        ]
        
        for method, endpoint, description, *data in endpoints:
            start_time = time.time()
            try:
                payload = data[0] if data else None
                
                if method == "GET":
                    response = requests.get(f"{self.backend_url}{endpoint}", timeout=10)
                elif method == "POST":
                    response = requests.post(f"{self.backend_url}{endpoint}", 
                                           json=payload, timeout=30)
                
                duration = time.time() - start_time
                
                if response.status_code in [200, 201]:
                    self.log_test('api_endpoints', f"{method}_{endpoint.replace('/', '_')}", 
                                "PASS", f"{description} - Status: {response.status_code}", duration)
                else:
                    self.log_test('api_endpoints', f"{method}_{endpoint.replace('/', '_')}", 
                                "FAIL", f"{description} - Status: {response.status_code}", duration)
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test('api_endpoints', f"{method}_{endpoint.replace('/', '_')}", 
                            "ERROR", f"{description} - Error: {str(e)}", duration)

    def test_database_operations(self) -> Dict[str, Any]:
        """Test database connectivity and CRUD operations"""
        print("\n🗄️ Testing Database Operations...")
        
        try:
            # Test connection
            start_time = time.time()
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            duration = time.time() - start_time
            
            self.log_test('database', 'connection', 'PASS', 
                        f"Connected to PostgreSQL database", duration)
            
            # Test CREATE
            start_time = time.time()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS test_validation (
                    id SERIAL PRIMARY KEY,
                    test_name VARCHAR(100),
                    test_result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            duration = time.time() - start_time
            self.log_test('database', 'create_table', 'PASS', 
                        "Test table created successfully", duration)
            
            # Test INSERT
            start_time = time.time()
            cur.execute("""
                INSERT INTO test_validation (test_name, test_result) 
                VALUES (%s, %s) RETURNING id
            """, ("validation_test", "PASS"))
            test_id = cur.fetchone()[0]
            conn.commit()
            duration = time.time() - start_time
            self.log_test('database', 'insert_record', 'PASS', 
                        f"Record inserted with ID {test_id}", duration)
            
            # Test SELECT
            start_time = time.time()
            cur.execute("SELECT * FROM test_validation WHERE id = %s", (test_id,))
            record = cur.fetchone()
            duration = time.time() - start_time
            if record:
                self.log_test('database', 'select_record', 'PASS', 
                            f"Record retrieved: {record[1]}", duration)
            else:
                self.log_test('database', 'select_record', 'FAIL', 
                            "Failed to retrieve record", duration)
            
            # Test UPDATE
            start_time = time.time()
            cur.execute("""
                UPDATE test_validation 
                SET test_result = %s 
                WHERE id = %s
            """, ("UPDATED", test_id))
            conn.commit()
            duration = time.time() - start_time
            self.log_test('database', 'update_record', 'PASS', 
                        "Record updated successfully", duration)
            
            # Test DELETE
            start_time = time.time()
            cur.execute("DELETE FROM test_validation WHERE id = %s", (test_id,))
            conn.commit()
            duration = time.time() - start_time
            self.log_test('database', 'delete_record', 'PASS', 
                        "Record deleted successfully", duration)
            
            # Cleanup
            cur.execute("DROP TABLE IF EXISTS test_validation")
            conn.commit()
            
            cur.close()
            conn.close()
            
        except Exception as e:
            self.log_test('database', 'general', 'ERROR', f"Database error: {str(e)}")

    def test_redis_operations(self) -> Dict[str, Any]:
        """Test Redis caching operations"""
        print("\n🗃️ Testing Redis Operations...")
        
        try:
            # Test connection
            start_time = time.time()
            r = redis.Redis(**self.redis_config)
            r.ping()
            duration = time.time() - start_time
            self.log_test('redis', 'connection', 'PASS', 
                        "Connected to Redis successfully", duration)
            
            # Test SET
            start_time = time.time()
            r.set('test_key', 'test_value', ex=60)  # Expire in 60 seconds
            duration = time.time() - start_time
            self.log_test('redis', 'set_operation', 'PASS', 
                        "Key set successfully", duration)
            
            # Test GET
            start_time = time.time()
            value = r.get('test_key')
            duration = time.time() - start_time
            if value and value.decode() == 'test_value':
                self.log_test('redis', 'get_operation', 'PASS', 
                            f"Key retrieved: {value.decode()}", duration)
            else:
                self.log_test('redis', 'get_operation', 'FAIL', 
                            "Failed to retrieve correct value", duration)
            
            # Test EXISTS
            start_time = time.time()
            exists = r.exists('test_key')
            duration = time.time() - start_time
            if exists:
                self.log_test('redis', 'exists_operation', 'PASS', 
                            "Key exists check passed", duration)
            else:
                self.log_test('redis', 'exists_operation', 'FAIL', 
                            "Key exists check failed", duration)
            
            # Test DELETE
            start_time = time.time()
            deleted = r.delete('test_key')
            duration = time.time() - start_time
            if deleted:
                self.log_test('redis', 'delete_operation', 'PASS', 
                            "Key deleted successfully", duration)
            else:
                self.log_test('redis', 'delete_operation', 'FAIL', 
                            "Failed to delete key", duration)
            
            # Test Memory Usage
            start_time = time.time()
            memory_info = r.info('memory')
            duration = time.time() - start_time
            used_memory = memory_info.get('used_memory_human', 'Unknown')
            self.log_test('redis', 'memory_usage', 'PASS', 
                        f"Memory usage: {used_memory}", duration)
            
        except Exception as e:
            self.log_test('redis', 'general', 'ERROR', f"Redis error: {str(e)}")

    def test_ollama_inference(self) -> Dict[str, Any]:
        """Test Ollama model inference and response quality"""
        print("\n🧠 Testing Ollama Model Inference...")
        
        try:
            # Test service availability
            start_time = time.time()
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                self.log_test('ollama', 'service_availability', 'PASS', 
                            f"Ollama service available with {len(models)} models", duration)
                
                if models:
                    model_name = models[0]['name']
                    
                    # Test simple inference
                    test_prompts = [
                        ("simple_math", "What is 2+2?"),
                        ("code_generation", "Write a simple Python hello world function"),
                        ("reasoning", "Explain the concept of recursion in programming"),
                        ("creative", "Write a short poem about AI")
                    ]
                    
                    for test_name, prompt in test_prompts:
                        start_time = time.time()
                        try:
                            inference_response = requests.post(
                                f"{self.ollama_url}/api/generate",
                                json={
                                    "model": model_name,
                                    "prompt": prompt,
                                    "stream": False
                                },
                                timeout=60
                            )
                            duration = time.time() - start_time
                            
                            if inference_response.status_code == 200:
                                result = inference_response.json()
                                response_text = result.get('response', '')
                                if response_text and len(response_text) > 10:
                                    self.log_test('ollama', f'inference_{test_name}', 'PASS', 
                                                f"Generated {len(response_text)} characters", duration)
                                else:
                                    self.log_test('ollama', f'inference_{test_name}', 'FAIL', 
                                                "Empty or too short response", duration)
                            else:
                                self.log_test('ollama', f'inference_{test_name}', 'FAIL', 
                                            f"Status: {inference_response.status_code}", duration)
                        except Exception as e:
                            duration = time.time() - start_time
                            self.log_test('ollama', f'inference_{test_name}', 'ERROR', 
                                        f"Error: {str(e)}", duration)
                else:
                    self.log_test('ollama', 'model_availability', 'FAIL', 
                                "No models available for testing")
            else:
                self.log_test('ollama', 'service_availability', 'FAIL', 
                            f"Service unavailable - Status: {response.status_code}", duration)
                
        except Exception as e:
            self.log_test('ollama', 'general', 'ERROR', f"Ollama error: {str(e)}")

    def test_frontend_accessibility(self) -> Dict[str, Any]:
        """Test frontend accessibility and functionality"""
        print("\n🌐 Testing Frontend Accessibility...")
        
        try:
            # Test frontend availability
            start_time = time.time()
            response = requests.get(self.frontend_url, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test('frontend', 'availability', 'PASS', 
                            "Frontend accessible", duration)
                
                # Check for key elements in response
                content = response.text
                if 'SutazAI' in content:
                    self.log_test('frontend', 'branding', 'PASS', 
                                "SutazAI branding present")
                else:
                    self.log_test('frontend', 'branding', 'FAIL', 
                                "SutazAI branding missing")
                
                if 'streamlit' in content.lower():
                    self.log_test('frontend', 'framework', 'PASS', 
                                "Streamlit framework detected")
                else:
                    self.log_test('frontend', 'framework', 'WARNING', 
                                "Streamlit framework not clearly detected")
                
            else:
                self.log_test('frontend', 'availability', 'FAIL', 
                            f"Frontend unavailable - Status: {response.status_code}", duration)
                
        except Exception as e:
            self.log_test('frontend', 'general', 'ERROR', f"Frontend error: {str(e)}")

    def test_agent_communication(self) -> Dict[str, Any]:
        """Test inter-agent communication and coordination"""
        print("\n🤖 Testing Agent Communication...")
        
        try:
            # Test agents list endpoint
            start_time = time.time()
            response = requests.get(f"{self.backend_url}/agents", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                agents_data = response.json()
                agents = agents_data.get('agents', [])
                
                self.log_test('agents', 'list_agents', 'PASS', 
                            f"Retrieved {len(agents)} agents", duration)
                
                # Test agent status
                active_agents = [a for a in agents if a.get('status') == 'active']
                healthy_agents = [a for a in agents if a.get('health') == 'healthy']
                
                self.log_test('agents', 'active_agents', 'PASS' if active_agents else 'WARNING', 
                            f"{len(active_agents)} active agents")
                self.log_test('agents', 'healthy_agents', 'PASS' if healthy_agents else 'WARNING', 
                            f"{len(healthy_agents)} healthy agents")
                
                # Test agent capabilities
                for agent in agents[:3]:  # Test first 3 agents
                    agent_id = agent.get('id', 'unknown')
                    capabilities = agent.get('capabilities', [])
                    if capabilities:
                        self.log_test('agents', f'capabilities_{agent_id}', 'PASS', 
                                    f"Agent has {len(capabilities)} capabilities")
                    else:
                        self.log_test('agents', f'capabilities_{agent_id}', 'WARNING', 
                                    "Agent has no defined capabilities")
                
            else:
                self.log_test('agents', 'list_agents', 'FAIL', 
                            f"Failed to get agents - Status: {response.status_code}", duration)
            
            # Test agent coordination through chat
            start_time = time.time()
            chat_response = requests.post(
                f"{self.backend_url}/chat",
                json={
                    "message": "Test agent coordination",
                    "agent": "task_coordinator"
                },
                timeout=30
            )
            duration = time.time() - start_time
            
            if chat_response.status_code == 200:
                self.log_test('agents', 'coordination_test', 'PASS', 
                            "Agent coordination test successful", duration)
            else:
                self.log_test('agents', 'coordination_test', 'FAIL', 
                            f"Agent coordination failed - Status: {chat_response.status_code}", duration)
                
        except Exception as e:
            self.log_test('agents', 'general', 'ERROR', f"Agent communication error: {str(e)}")

    def test_system_integration(self) -> Dict[str, Any]:
        """Run comprehensive system integration tests"""
        print("\n🔗 Testing System Integration...")
        
        # Test end-to-end workflow
        try:
            # 1. Check system status
            start_time = time.time()
            status_response = requests.get(f"{self.backend_url}/api/v1/system/status", timeout=10)
            duration = time.time() - start_time
            
            if status_response.status_code == 200:
                self.log_test('integration', 'system_status', 'PASS', 
                            "System status check passed", duration)
                
                status_data = status_response.json()
                services = status_data.get('services', {})
                
                # Check critical services
                critical_services = ['ollama', 'database', 'redis']
                for service in critical_services:
                    if service in str(services):
                        self.log_test('integration', f'service_{service}', 'PASS', 
                                    f"{service} service integrated")
                    else:
                        self.log_test('integration', f'service_{service}', 'WARNING', 
                                    f"{service} service status unclear")
            else:
                self.log_test('integration', 'system_status', 'FAIL', 
                            f"System status check failed - Status: {status_response.status_code}", duration)
            
            # 2. Test complete AI pipeline
            start_time = time.time()
            pipeline_test = requests.post(
                f"{self.backend_url}/think",
                json={
                    "query": "Integrate database storage, AI processing, and cache optimization",
                    "reasoning_type": "systematic"
                },
                timeout=30
            )
            duration = time.time() - start_time
            
            if pipeline_test.status_code == 200:
                self.log_test('integration', 'ai_pipeline', 'PASS', 
                            "Complete AI pipeline test passed", duration)
            else:
                self.log_test('integration', 'ai_pipeline', 'FAIL', 
                            f"AI pipeline test failed - Status: {pipeline_test.status_code}", duration)
            
            # 3. Test performance under load
            start_time = time.time()
            concurrent_requests = []
            for i in range(5):
                try:
                    response = requests.post(
                        f"{self.backend_url}/simple-chat",
                        json={"message": f"Test message {i}"},
                        timeout=20
                    )
                    concurrent_requests.append(response.status_code == 200)
                except:
                    concurrent_requests.append(False)
            
            duration = time.time() - start_time
            success_rate = sum(concurrent_requests) / len(concurrent_requests)
            
            if success_rate >= 0.8:
                self.log_test('integration', 'load_test', 'PASS', 
                            f"Load test passed with {success_rate*100:.1f}% success rate", duration)
            else:
                self.log_test('integration', 'load_test', 'FAIL', 
                            f"Load test failed with {success_rate*100:.1f}% success rate", duration)
                
        except Exception as e:
            self.log_test('integration', 'general', 'ERROR', f"Integration error: {str(e)}")

    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks and response times"""
        print("\n⚡ Testing Performance Benchmarks...")
        
        # API response time benchmark
        endpoints = [
            ("/health", "GET", None),
            ("/agents", "GET", None),
            ("/simple-chat", "POST", {"message": "Quick test"}),
        ]
        
        for endpoint, method, payload in endpoints:
            times = []
            for _ in range(5):  # Run 5 times for average
                start_time = time.time()
                try:
                    if method == "GET":
                        response = requests.get(f"{self.backend_url}{endpoint}", timeout=10)
                    else:
                        response = requests.post(f"{self.backend_url}{endpoint}", 
                                               json=payload, timeout=20)
                    
                    if response.status_code in [200, 201]:
                        times.append(time.time() - start_time)
                except:
                    pass
            
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                
                status = "PASS" if avg_time < 2.0 else "WARNING" if avg_time < 5.0 else "FAIL"
                self.log_test('performance', f'response_time_{endpoint.replace("/", "_")}', 
                            status, f"Avg: {avg_time:.2f}s, Max: {max_time:.2f}s")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        print("🚀 Starting Comprehensive SutazAI System Validation...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        self.test_api_endpoints()
        self.test_database_operations()
        self.test_redis_operations()
        self.test_ollama_inference()
        self.test_frontend_accessibility()
        self.test_agent_communication()
        self.test_system_integration()
        self.test_performance_benchmarks()
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = self.generate_test_summary(total_duration)
        
        print("\n" + "=" * 60)
        print("🏁 Test Execution Complete!")
        print(f"Total Duration: {total_duration:.2f} seconds")
        
        return {
            'results': self.test_results,
            'summary': summary,
            'execution_time': total_duration
        }

    def generate_test_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'warnings': 0,
            'categories': {},
            'critical_failures': [],
            'recommendations': []
        }
        
        for category, tests in self.test_results.items():
            category_summary = {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0, 'warnings': 0}
            
            for test_name, result in tests.items():
                status = result['status']
                category_summary['total'] += 1
                summary['total_tests'] += 1
                
                if status == 'PASS':
                    category_summary['passed'] += 1
                    summary['passed'] += 1
                elif status == 'FAIL':
                    category_summary['failed'] += 1
                    summary['failed'] += 1
                    if category in ['api_endpoints', 'database', 'ollama']:
                        summary['critical_failures'].append(f"{category}.{test_name}")
                elif status == 'ERROR':
                    category_summary['errors'] += 1
                    summary['errors'] += 1
                    summary['critical_failures'].append(f"{category}.{test_name}")
                elif status == 'WARNING':
                    category_summary['warnings'] += 1
                    summary['warnings'] += 1
            
            summary['categories'][category] = category_summary
        
        # Generate recommendations
        if summary['critical_failures']:
            summary['recommendations'].append("Address critical failures in core system components")
        
        if summary['failed'] > 0:
            summary['recommendations'].append("Investigate and fix failed tests")
        
        if summary['warnings'] > 0:
            summary['recommendations'].append("Review warnings for potential improvements")
        
        success_rate = (summary['passed'] / summary['total_tests']) * 100 if summary['total_tests'] > 0 else 0
        
        if success_rate >= 90:
            summary['overall_status'] = 'EXCELLENT'
        elif success_rate >= 75:
            summary['overall_status'] = 'GOOD'
        elif success_rate >= 50:
            summary['overall_status'] = 'FAIR'
        else:
            summary['overall_status'] = 'POOR'
        
        summary['success_rate'] = success_rate
        
        return summary

def main():
    """Main test execution function"""
    test_suite = ComprehensiveTestSuite()
    results = test_suite.run_all_tests()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'/opt/sutazaiapp/test_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results['summary']
    print(f"\n📊 Test Summary:")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ({summary['success_rate']:.1f}%)")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Overall Status: {summary['overall_status']}")
    
    if summary['critical_failures']:
        print(f"\n⚠️ Critical Failures:")
        for failure in summary['critical_failures']:
            print(f"  - {failure}")
    
    if summary['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
    
    return results

if __name__ == "__main__":
    main()