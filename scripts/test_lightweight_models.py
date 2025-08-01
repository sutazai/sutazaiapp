#!/usr/bin/env python3
"""
Test script for ultra-lightweight Ollama models.
Tests the smallest models to ensure they work for basic AI agent tasks.
"""

import subprocess
import time
import json

def test_model(model_name, prompt, description):
    """Test a specific model with a prompt."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}: {description}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([
            'docker', 'exec', 'sutazai-ollama', 'ollama', 'run', model_name, prompt
        ], capture_output=True, text=True, timeout=30)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.2f}s)")
            print(f"Response: {result.stdout.strip()}")
            return True, duration
        else:
            print(f"❌ FAILED")
            print(f"Error: {result.stderr}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (>30s)")
        return False, 30.0
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, 0.0

def get_model_info():
    """Get information about available models."""
    try:
        result = subprocess.run([
            'docker', 'exec', 'sutazai-ollama', 'ollama', 'list'
        ], capture_output=True, text=True)
        
        print("\n📊 Available Models:")
        print("=" * 60)
        print(result.stdout)
        
    except Exception as e:
        print(f"Error getting model info: {e}")

def main():
    print("🔬 Testing Ultra-Lightweight Ollama Models for SutazAI")
    print("Resource-constrained environment optimization test")
    
    get_model_info()
    
    # Test cases designed for AI agent functionality
    test_cases = [
        ("smollm:135m", "Hello, can you help me?", "Ultra-light general task"),
        ("smollm:135m", "What is 2+2?", "Ultra-light math"),
        ("smollm:360m", "Write a Python function to check if a number is even.", "Small model coding"),
        ("smollm:360m", "Explain what an API is in one sentence.", "Small model explanation"),
        ("tinyllama:1.1b", "Create a simple task list for a web developer.", "Medium model planning"),
        ("tinyllama:1.1b", "How do I debug a Python error?", "Medium model help"),
    ]
    
    results = []
    total_time = 0
    
    for model, prompt, description in test_cases:
        success, duration = test_model(model, prompt, description)
        results.append({
            'model': model,
            'description': description,
            'success': success,
            'duration': duration
        })
        total_time += duration
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Successful tests: {successful_tests}/{total_tests}")
    print(f"Total test time: {total_time:.2f} seconds")
    print(f"Average response time: {total_time/total_tests:.2f} seconds")
    
    print("\n📊 Model Performance:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['model']}: {result['duration']:.2f}s - {result['description']}")
    
    print(f"\n💡 Recommendations:")
    print("- smollm:135m (91MB): Ultra-light for basic responses")
    print("- smollm:360m (229MB): Best balance of size vs capability")
    print("- tinyllama:1.1b (637MB): When you need better reasoning")
    print("- Keep qwen2.5:3b (1.9GB) as fallback for complex tasks")

if __name__ == "__main__":
    main()