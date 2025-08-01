#!/usr/bin/env python3
"""
Fixed Ollama integration for the backend
Properly handles model responses without default fallbacks
"""

import requests
import json
import time
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.default_timeout = 120  # 2 minutes for model responses
        
    def generate(self, prompt, model="llama3.2:1b", stream=False):
        """Generate response from Ollama model"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "num_predict": 500  # Limit response length
            }
        }
        
        try:
            logger.info(f"Sending request to Ollama: model={model}, prompt_length={len(prompt)}")
            start_time = time.time()
            
            response = requests.post(url, json=payload, timeout=self.default_timeout)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                elapsed = time.time() - start_time
                logger.info(f"Ollama responded in {elapsed:.2f}s with {len(response_text)} chars")
                
                return {
                    "success": True,
                    "response": response_text,
                    "model": model,
                    "duration": elapsed,
                    "context": result.get("context", [])
                }
            else:
                logger.error(f"Ollama returned status {response.status_code}")
                return {
                    "success": False,
                    "error": f"Model returned status {response.status_code}",
                    "response": None
                }
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama timeout after {self.default_timeout}s")
            return {
                "success": False,
                "error": "Request timed out. The model may be loading or the prompt is too complex.",
                "response": None
            }
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama")
            return {
                "success": False,
                "error": "Cannot connect to Ollama. Please ensure the service is running.",
                "response": None
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def list_models(self):
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except:
            return []
    
    def pull_model(self, model_name):
        """Pull a model (non-blocking)"""
        try:
            payload = {"name": model_name}
            response = requests.post(f"{self.base_url}/api/pull", json=payload, timeout=5)
            return response.status_code == 200
        except:
            return False

# Test the client
if __name__ == "__main__":
    print("Testing Ollama Client...")
    client = OllamaClient()
    
    # List models
    models = client.list_models()
    print(f"\nAvailable models: {len(models)}")
    for model in models:
        print(f"  - {model['name']}")
    
    # Test generation
    print("\nTesting generation...")
    result = client.generate("What is 2+2?", model="llama3.2:1b")
    
    if result["success"]:
        print(f"✅ Success! Response: {result['response']}")
        print(f"   Duration: {result['duration']:.2f}s")
    else:
        print(f"❌ Failed: {result['error']}")