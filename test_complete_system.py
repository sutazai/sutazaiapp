#!/usr/bin/env python3
"""
SutazAI System Comprehensive Test Suite
Tests all components and provides operational verification
"""

import asyncio
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import requests
import psutil
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class SutazAITester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = []
        self.start_time = time.time()
        
    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text.center(60)}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def print_test(self, name: str, status: bool, details: str = ""):
        """Print test result"""
        if status:
            status_text = f"{Fore.GREEN}✓ PASSED{Style.RESET_ALL}"
        else:
            status_text = f"{Fore.RED}✗ FAILED{Style.RESET_ALL}"
        
        print(f"{status_text} {name}")
        if details:
            print(f"  {Fore.YELLOW}→ {details}{Style.RESET_ALL}")
        
        self.test_results.append((name, status, details))
    
    def test_health_endpoint(self) -> bool:
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                ollama_status = data.get("services", {}).get("ollama", {}).get("status")
                agents_online = data.get("services", {}).get("external_agents", {}).get("online", 0)
                
                details = f"Ollama: {ollama_status}, Agents online: {agents_online}"
                self.print_test("Health Endpoint", True, details)
                return True
            else:
                self.print_test("Health Endpoint", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Health Endpoint", False, str(e))
            return False
    
    def test_models_endpoint(self) -> List[str]:
        """Test models endpoint and return available models"""
        try:
            response = requests.get(f"{self.base_url}/api/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                
                if models:
                    self.print_test("Models Endpoint", True, f"Found {len(models)} models: {', '.join(models)}")
                else:
                    self.print_test("Models Endpoint", False, "No models available")
                
                return models
            else:
                self.print_test("Models Endpoint", False, f"Status code: {response.status_code}")
                return []
        except Exception as e:
            self.print_test("Models Endpoint", False, str(e))
            return []
    
    def test_chat_simple(self) -> bool:
        """Test simple chat functionality"""
        try:
            payload = {
                "message": "What is 2+2?",
                "model": "llama3.2:1b"
            }
            
            start = time.time()
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=30)
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                ollama_success = data.get("ollama_success", False)
                response_text = data.get("response", "")[:50] + "..."
                
                details = f"Ollama: {ollama_success}, Time: {duration:.2f}s, Response: {response_text}"
                self.print_test("Simple Chat", True, details)
                return True
            else:
                self.print_test("Simple Chat", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Simple Chat", False, str(e))
            return False
    
    def test_chat_complex(self) -> bool:
        """Test complex chat query"""
        try:
            payload = {
                "message": "Explain how the SutazAI system can self-improve using external AI agents",
                "model": "llama3.2:1b",
                "temperature": 0.7,
                "max_tokens": 200
            }
            
            start = time.time()
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=90)
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                ollama_success = data.get("ollama_success", False)
                tokens = data.get("tokens_used", 0)
                
                details = f"Ollama: {ollama_success}, Time: {duration:.2f}s, Tokens: {tokens}"
                self.print_test("Complex Chat", True, details)
                return True
            else:
                self.print_test("Complex Chat", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Complex Chat", False, str(e))
            return False
    
    def test_agents_endpoint(self) -> Dict:
        """Test agents endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/agents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                total = data.get("total", 0)
                online = data.get("online", 0)
                
                details = f"Total agents: {total}, Online: {online}"
                self.print_test("Agents Endpoint", True, details)
                return data
            else:
                self.print_test("Agents Endpoint", False, f"Status code: {response.status_code}")
                return {}
        except Exception as e:
            self.print_test("Agents Endpoint", False, str(e))
            return {}
    
    def test_performance_endpoint(self) -> bool:
        """Test performance monitoring"""
        try:
            response = requests.get(f"{self.base_url}/api/performance/summary", timeout=5)
            if response.status_code == 200:
                data = response.json()
                cpu = data.get("system", {}).get("cpu_usage", 0)
                memory = data.get("system", {}).get("memory_usage", 0)
                requests_total = data.get("api", {}).get("total_requests", 0)
                
                details = f"CPU: {cpu:.1f}%, Memory: {memory:.1f}%, Requests: {requests_total}"
                self.print_test("Performance Monitoring", True, details)
                return True
            else:
                self.print_test("Performance Monitoring", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Performance Monitoring", False, str(e))
            return False
    
    def test_system_resources(self) -> bool:
        """Test system resource availability"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        issues = []
        if cpu_percent > 90:
            issues.append(f"High CPU: {cpu_percent}%")
        if memory.percent > 90:
            issues.append(f"High Memory: {memory.percent}%")
        if disk.percent > 95:
            issues.append(f"Low Disk: {disk.percent}% used")
        
        if issues:
            self.print_test("System Resources", False, ", ".join(issues))
            return False
        else:
            details = f"CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%"
            self.print_test("System Resources", True, details)
            return True
    
    def test_docker_services(self) -> bool:
        """Test Docker services"""
        services = {
            "PostgreSQL": 5432,
            "Redis": 6379,
            "Ollama": 11434,
            "Qdrant": 6333,
            "ChromaDB": 8001
        }
        
        all_running = True
        running_services = []
        
        for service, port in services.items():
            try:
                response = requests.get(f"http://localhost:{port}", timeout=2)
                running_services.append(service)
            except:
                try:
                    # Try alternate check for some services
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    if result == 0:
                        running_services.append(service)
                    else:
                        all_running = False
                except:
                    all_running = False
        
        details = f"Running: {', '.join(running_services)}"
        self.print_test("Docker Services", all_running, details)
        return all_running
    
    def test_ollama_models(self) -> bool:
        """Test Ollama model availability"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                if models:
                    model_names = [m["name"] for m in models]
                    details = f"Available: {', '.join(model_names)}"
                    self.print_test("Ollama Models", True, details)
                    return True
                else:
                    self.print_test("Ollama Models", False, "No models loaded")
                    return False
            else:
                self.print_test("Ollama Models", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.print_test("Ollama Models", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        self.print_header("SutazAI System Test Suite")
        print(f"\n{Fore.YELLOW}Starting comprehensive system tests...{Style.RESET_ALL}")
        
        # Infrastructure tests
        self.print_header("Infrastructure Tests")
        self.test_system_resources()
        self.test_docker_services()
        self.test_ollama_models()
        
        # API tests
        self.print_header("API Endpoint Tests")
        self.test_health_endpoint()
        self.test_models_endpoint()
        self.test_agents_endpoint()
        self.test_performance_endpoint()
        
        # Functionality tests
        self.print_header("Functionality Tests")
        self.test_chat_simple()
        self.test_chat_complex()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("Test Summary")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, status, _ in self.test_results if status)
        failed_tests = total_tests - passed_tests
        duration = time.time() - self.start_time
        
        print(f"\n{Fore.CYAN}Total Tests:{Style.RESET_ALL} {total_tests}")
        print(f"{Fore.GREEN}Passed:{Style.RESET_ALL} {passed_tests}")
        print(f"{Fore.RED}Failed:{Style.RESET_ALL} {failed_tests}")
        print(f"{Fore.YELLOW}Duration:{Style.RESET_ALL} {duration:.2f} seconds")
        
        if failed_tests == 0:
            print(f"\n{Fore.GREEN}{'🎉 ALL TESTS PASSED! 🎉'.center(60)}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}The SutazAI system is fully operational!{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}⚠ Some tests failed. Review the results above.{Style.RESET_ALL}")
            print("\nFailed tests:")
            for name, status, details in self.test_results:
                if not status:
                    print(f"  - {name}: {details}")
        
        # Provide recommendations
        self.print_header("Recommendations")
        
        if failed_tests > 0:
            print("To fix issues:")
            print("1. Check logs: tail -f /opt/sutazaiapp/logs/backend_complete.log")
            print("2. Restart backend: systemctl restart sutazai-complete-backend")
            print("3. Check Docker: docker ps")
            print("4. Run diagnostics: python3 fix_all_issues.py")
        else:
            print("Your system is ready! Next steps:")
            print("1. Try the chat interface with complex queries")
            print("2. Monitor performance: curl http://localhost:8000/api/performance/summary | jq")
            print("3. Deploy AI agents when needed: docker-compose up -d autogpt crewai")
            print("4. Add more models: docker exec sutazai-ollama ollama pull mistral:7b")

def main():
    """Main test execution"""
    tester = SutazAITester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Tests interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Test suite error: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()