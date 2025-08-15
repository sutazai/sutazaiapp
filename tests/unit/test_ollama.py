#!/usr/bin/env python3
"""Test and diagnose Ollama performance issues"""

import logging

logger = logging.getLogger(__name__)
import requests
import time
import json
import sys

def test_ollama():
    """Test Ollama with   configuration"""
    
    # Test 1: Check if Ollama is running
    try:
        r = requests.get("http://localhost:10104/api/tags", timeout=2)
        models = r.json().get("models", [])
        logger.info(f"✅ Ollama is running with {len(models)} models")
        for model in models:
            logger.info(f"  - {model['name']}: {model['size'] / 1024 / 1024:.0f}MB")
    except Exception as e:
        logger.info(f"❌ Cannot connect to Ollama: {e}")
        return False
    
    # Test 2: Try a   generation
    logger.info("\n🔄 Testing generation with   settings...")
    
    payload = {
        "model": "tinyllama",
        "prompt": "Hi",
        "stream": False,
        "options": {
            "num_predict": 1,  # Generate only 1 token
            "temperature": 0,   # Deterministic
            "top_k": 1,        # Only top choice
            "top_p": 0.1,      #   sampling
            "repeat_penalty": 1.0,
            "seed": 42,        # Fixed seed
            "num_ctx": 128,    #   context
            "num_batch": 1,    # Single batch
            "num_thread": 1    # Single thread
        }
    }
    
    start = time.time()
    try:
        r = requests.post(
            "http://localhost:10104/api/generate",
            json=payload,
            timeout=10
        )
        elapsed = time.time() - start
        
        if r.status_code == 200:
            result = r.json()
            logger.info(f"✅ Response in {elapsed:.2f}s")
            logger.info(f"Generated: {result.get('response', 'NO RESPONSE')}")
            
            if elapsed < 2:
                logger.info("🎉 PERFORMANCE IS GOOD!")
            elif elapsed < 5:
                logger.info("⚠️ Performance is acceptable but could be better")
            else:
                logger.info("❌ Performance is too slow!")
                
            return elapsed < 5
        else:
            logger.error(f"❌ Error {r.status_code}: {r.text}")
            return False
            
    except requests.Timeout:
        logger.info(f"❌ Request timed out after 10 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama()
    sys.exit(0 if success else 1)