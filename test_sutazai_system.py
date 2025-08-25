#!/usr/bin/env python3
"""
SutazAI System Comprehensive Test Suite
Tests all components from PRD including Jarvis, Agents, and MCP
"""

import asyncio
import json
import time
import requests
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class SutazAISystemTester:
    """Complete system testing for production readiness"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "components": {}
        }
        self.base_url = "http://localhost:10010"
        
    def test_backend_api(self) -> bool:
        """Test backend API endpoints"""
        print("\n🔍 Testing Backend API...")
        
        endpoints = [
            ("/health", "GET", None),
            ("/api/v1/agents", "GET", None),
            ("/api/v1/models", "GET", None),
            ("/api/v1/documents", "GET", None),
        ]
        
        results = []
        for endpoint, method, data in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                if method == "GET":
                    response = requests.get(url, timeout=5)
                else:
                    response = requests.post(url, json=data, timeout=5)
                    
                success = response.status_code in [200, 201]
                results.append(success)
                print(f"  ✅ {endpoint}: {response.status_code}" if success else f"  ❌ {endpoint}: {response.status_code}")
                
            except Exception as e:
                print(f"  ❌ {endpoint}: {str(e)[:50]}")
                results.append(False)
                
        return all(results)
        
    def test_database_connection(self) -> bool:
        """Test database connection and pooling"""
        print("\n🔍 Testing Database Connection...")
        
        try:
            # Test connection pooling endpoint if available
            response = requests.get(f"{self.base_url}/api/v1/db/status", timeout=5)
            if response.status_code == 200:
                pool_status = response.json()
                print(f"  ✅ Database Pool: Size={pool_status.get('pool_size', 'N/A')}, Active={pool_status.get('checked_out', 'N/A')}")
                return True
        except:
            pass
            
        # Fallback to psql test
        try:
            result = subprocess.run(
                ["psql", "-h", "localhost", "-p", "10000", "-U", "sutazai", "-d", "sutazai", "-c", "SELECT 1"],
                capture_output=True,
                text=True,
                timeout=5,
                env={"PGPASSWORD": "sutazai123"}
            )
            success = result.returncode == 0
            print(f"  {'✅' if success else '❌'} PostgreSQL Direct Connection")
            return success
        except:
            print("  ❌ Database connection failed")
            return False
            
    def test_redis_cache(self) -> bool:
        """Test Redis cache functionality"""
        print("\n🔍 Testing Redis Cache...")
        
        try:
            import redis
            r = redis.Redis(host='localhost', port=10001, decode_responses=True)
            
            # Test SET/GET
            test_key = f"test_{int(time.time())}"
            r.set(test_key, "test_value", ex=10)
            value = r.get(test_key)
            
            # Test memory usage
            info = r.info('memory')
            used_memory = info.get('used_memory_human', 'N/A')
            
            print(f"  ✅ Redis Cache: Memory={used_memory}, Test={'✅' if value == 'test_value' else '❌'}")
            return value == "test_value"
            
        except Exception as e:
            print(f"  ❌ Redis Cache: {str(e)[:50]}")
            return False
            
    def test_vector_databases(self) -> bool:
        """Test ChromaDB and Qdrant"""
        print("\n🔍 Testing Vector Databases...")
        
        results = []
        
        # Test ChromaDB
        try:
            response = requests.get("http://localhost:10100/api/v1/heartbeat", timeout=5)
            success = response.status_code == 200
            results.append(success)
            print(f"  {'✅' if success else '❌'} ChromaDB")
        except:
            print("  ❌ ChromaDB: Not accessible")
            results.append(False)
            
        # Test Qdrant
        try:
            response = requests.get("http://localhost:10101/health", timeout=5)
            success = response.status_code == 200
            results.append(success)
            print(f"  {'✅' if success else '❌'} Qdrant")
        except:
            print("  ❌ Qdrant: Not accessible")
            results.append(False)
            
        return all(results) if results else False
        
    def test_ollama_models(self) -> bool:
        """Test Ollama and model availability"""
        print("\n🔍 Testing Ollama & Models...")
        
        try:
            response = requests.get("http://localhost:10104/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                print(f"  ✅ Ollama: {len(models)} models available")
                
                # Check for TinyLlama
                has_tinyllama = any('tinyllama' in name.lower() for name in model_names)
                print(f"  {'✅' if has_tinyllama else '⚠️'} TinyLlama: {'Available' if has_tinyllama else 'Not found'}")
                
                # Check for Qwen
                has_qwen = any('qwen' in name.lower() for name in model_names)
                print(f"  {'ℹ️' if not has_qwen else '✅'} Qwen: {'Available' if has_qwen else 'Load on demand'}")
                
                return True
            return False
        except Exception as e:
            print(f"  ❌ Ollama: {str(e)[:50]}")
            return False
            
    def test_mcp_servers(self) -> bool:
        """Test MCP server configurations"""
        print("\n🔍 Testing MCP Servers...")
        
        mcp_config_path = Path("mcp-servers-config.json")
        if not mcp_config_path.exists():
            print("  ⚠️ MCP config not found")
            return False
            
        with open(mcp_config_path) as f:
            config = json.load(f)
            
        server_count = len(config.get('mcpServers', {}))
        print(f"  ✅ MCP Config: {server_count} servers configured")
        
        # Test a few key servers
        key_servers = ['sequential-thinking', 'claude-flow', 'ruv-swarm']
        for server in key_servers:
            if server in config.get('mcpServers', {}):
                print(f"    ✅ {server}: Configured")
            else:
                print(f"    ❌ {server}: Missing")
                
        return server_count > 0
        
    def test_agent_system(self) -> bool:
        """Test agent orchestration system"""
        print("\n🔍 Testing Agent System...")
        
        try:
            # Test agent registry
            response = requests.get(f"{self.base_url}/api/v1/agents", timeout=5)
            if response.status_code == 200:
                agents = response.json()
                print(f"  ✅ Agent Registry: {len(agents)} agents registered")
                
                # Test agent health
                healthy = sum(1 for a in agents if a.get('status') == 'healthy')
                print(f"  ℹ️ Agent Health: {healthy}/{len(agents)} healthy")
                
                return len(agents) > 0
            return False
        except Exception as e:
            print(f"  ❌ Agent System: {str(e)[:50]}")
            return False
            
    def test_resource_usage(self) -> bool:
        """Test system resource usage"""
        print("\n🔍 Testing Resource Usage...")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"  ℹ️ CPU Usage: {cpu_percent}%")
        print(f"  ℹ️ Memory: {memory.percent}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
        print(f"  ℹ️ Disk: {disk.percent}% used")
        
        # Check if resources are within acceptable limits
        if memory.percent > 90:
            print("  ⚠️ High memory usage detected!")
            return False
        if cpu_percent > 90:
            print("  ⚠️ High CPU usage detected!")
            return False
            
        print("  ✅ Resource usage within limits")
        return True
        
    def test_jarvis_interface(self) -> bool:
        """Test Jarvis voice/chat interface"""
        print("\n🔍 Testing Jarvis Interface...")
        
        try:
            # Test frontend
            response = requests.get("http://localhost:10011", timeout=5)
            frontend_ok = response.status_code == 200
            print(f"  {'✅' if frontend_ok else '❌'} Frontend (Streamlit)")
            
            # Test Jarvis command endpoint
            test_command = {
                "text": "Hello Jarvis, what is the system status?",
                "voice": False
            }
            response = requests.post(
                f"{self.base_url}/api/v1/jarvis/command",
                json=test_command,
                timeout=10
            )
            jarvis_ok = response.status_code in [200, 201]
            print(f"  {'✅' if jarvis_ok else '❌'} Jarvis Command Processing")
            
            return frontend_ok or jarvis_ok
            
        except Exception as e:
            print(f"  ❌ Jarvis Interface: {str(e)[:50]}")
            return False
            
    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 60)
        print("🚀 SutazAI System Comprehensive Test Suite")
        print("=" * 60)
        
        tests = [
            ("Backend API", self.test_backend_api),
            ("Database", self.test_database_connection),
            ("Redis Cache", self.test_redis_cache),
            ("Vector DBs", self.test_vector_databases),
            ("Ollama Models", self.test_ollama_models),
            ("MCP Servers", self.test_mcp_servers),
            ("Agent System", self.test_agent_system),
            ("Resources", self.test_resource_usage),
            ("Jarvis", self.test_jarvis_interface),
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results["components"][test_name] = result
                if result:
                    self.test_results["tests_passed"] += 1
                else:
                    self.test_results["tests_failed"] += 1
            except Exception as e:
                print(f"\n❌ Test '{test_name}' crashed: {str(e)}")
                self.test_results["components"][test_name] = False
                self.test_results["tests_failed"] += 1
                
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total = self.test_results["tests_passed"] + self.test_results["tests_failed"]
        success_rate = (self.test_results["tests_passed"] / total * 100) if total > 0 else 0
        
        print(f"✅ Passed: {self.test_results['tests_passed']}/{total}")
        print(f"❌ Failed: {self.test_results['tests_failed']}/{total}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # System readiness assessment
        critical_components = ["Backend API", "Database", "Ollama Models"]
        critical_ok = all(
            self.test_results["components"].get(comp, False) 
            for comp in critical_components
        )
        
        if critical_ok and success_rate >= 70:
            print("\n✅ SYSTEM STATUS: PRODUCTION READY")
        elif critical_ok and success_rate >= 50:
            print("\n⚠️ SYSTEM STATUS: PARTIALLY OPERATIONAL")
        else:
            print("\n❌ SYSTEM STATUS: NOT READY")
            
        # Save results
        with open("test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n📁 Detailed results saved to test_results.json")
        
        return success_rate >= 70

if __name__ == "__main__":
    tester = SutazAISystemTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)