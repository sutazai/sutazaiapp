#!/usr/bin/env python3
"""
SutazAI System Validation Demo
Demonstrates the working capabilities of the deployed system
"""

import json
import requests
import time
from datetime import datetime

class SutazaiValidator:
    def __init__(self):
        self.services = {
            "postgres": "5432",
            "redis": "6379", 
            "qdrant": "6333",
            "chromadb": "8001",
            "ollama": "11434"
        }
        self.results = {}
        
    def test_service_health(self, service_name, port):
        """Test if a service is healthy"""
        try:
            if service_name == "postgres":
                # For postgres, we just check if port is open
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('localhost', int(port)))
                sock.close()
                return result == 0
                
            elif service_name == "redis":
                # Redis health check
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('localhost', int(port)))
                sock.close()
                return result == 0
                
            elif service_name == "qdrant":
                response = requests.get(f"http://localhost:{port}/healthz", timeout=5)
                return response.status_code == 200
                
            elif service_name == "chromadb":
                response = requests.get(f"http://localhost:{port}/api/v1/heartbeat", timeout=5)
                return response.status_code == 200 or "deprecated" in response.text
                
            elif service_name == "ollama":
                response = requests.get(f"http://localhost:{port}/api/version", timeout=5)
                return response.status_code == 200
                
        except Exception as e:
            print(f"Error testing {service_name}: {e}")
            return False
            
    def test_vector_database_functionality(self):
        """Test vector database operations"""
        results = {}
        
        # Test Qdrant
        try:
            # Create a test collection
            collection_data = {
                "vectors": {
                    "size": 768,
                    "distance": "Cosine"
                }
            }
            
            # Try to create collection
            response = requests.put(
                "http://localhost:6333/collections/test_collection",
                json=collection_data,
                timeout=10
            )
            
            if response.status_code in [200, 409]:  # 409 = already exists
                results["qdrant_collection_create"] = True
                
                # Try to get collection info
                response = requests.get(
                    "http://localhost:6333/collections/test_collection",
                    timeout=5
                )
                results["qdrant_collection_info"] = response.status_code == 200
            else:
                results["qdrant_collection_create"] = False
                results["qdrant_collection_info"] = False
                
        except Exception as e:
            results["qdrant_error"] = str(e)
            
        return results
        
    def test_ollama_functionality(self):
        """Test Ollama model management"""
        results = {}
        
        try:
            # Get version
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code == 200:
                results["ollama_version"] = response.json()["version"]
                
            # List models
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                results["ollama_models"] = [model["name"] for model in models.get("models", [])]
                results["ollama_models_count"] = len(models.get("models", []))
            else:
                results["ollama_models"] = []
                results["ollama_models_count"] = 0
                
        except Exception as e:
            results["ollama_error"] = str(e)
            
        return results
        
    def run_validation(self):
        """Run complete system validation"""
        print("🚀 Starting SutazAI System Validation")
        print("="*60)
        
        # Test service health
        print("\n📋 Testing Service Health:")
        for service, port in self.services.items():
            is_healthy = self.test_service_health(service, port)
            status = "✅ HEALTHY" if is_healthy else "❌ UNHEALTHY"
            print(f"  {service.upper()}: {status}")
            self.results[f"{service}_health"] = is_healthy
            
        # Test vector database functionality
        print("\n🔍 Testing Vector Database Functionality:")
        vector_results = self.test_vector_database_functionality()
        for key, value in vector_results.items():
            status = "✅ PASS" if value else "❌ FAIL"
            print(f"  {key}: {status}")
        self.results.update(vector_results)
        
        # Test Ollama functionality
        print("\n🤖 Testing AI Model Management:")
        ollama_results = self.test_ollama_functionality()
        for key, value in ollama_results.items():
            if "error" in key:
                print(f"  {key}: ❌ {value}")
            else:
                print(f"  {key}: ✅ {value}")
        self.results.update(ollama_results)
        
        # Generate summary
        print("\n📊 Validation Summary:")
        healthy_services = sum(1 for k, v in self.results.items() if k.endswith("_health") and v)
        total_services = len(self.services)
        print(f"  Services Health: {healthy_services}/{total_services} healthy")
        
        # Save results
        self.results["validation_timestamp"] = datetime.now().isoformat()
        self.results["validation_summary"] = {
            "healthy_services": healthy_services,
            "total_services": total_services,
            "overall_health": healthy_services / total_services * 100
        }
        
        with open("validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n✅ Validation Complete! Results saved to validation_results.json")
        print(f"📊 Overall System Health: {healthy_services/total_services*100:.1f}%")
        
        return self.results

if __name__ == "__main__":
    validator = SutazaiValidator()
    results = validator.run_validation()