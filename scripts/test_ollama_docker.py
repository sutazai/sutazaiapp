#!/usr/bin/env python3
"""
Alternative Ollama setup using Docker for better compatibility
"""

import subprocess
import time
import requests
import sys
import os

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def start_ollama_docker():
    """Start Ollama using Docker"""
    try:
        # Check if ollama container already exists
        result = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=ollama', '--format', '{{.Names}}'], 
                              capture_output=True, text=True)
        
        if 'ollama' in result.stdout:
            print("Stopping existing ollama container...")
            subprocess.run(['docker', 'stop', 'ollama'], capture_output=True)
            subprocess.run(['docker', 'rm', 'ollama'], capture_output=True)
        
        print("Starting Ollama in Docker...")
        cmd = [
            'docker', 'run', '-d',
            '--name', 'ollama',
            '-p', '11434:11434',
            '-v', '/opt/sutazaiapp/models/ollama:/root/.ollama',
            'ollama/ollama'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama Docker container started")
            
            # Wait for service to be ready
            for i in range(30):
                try:
                    response = requests.get('http://127.0.0.1:11434/api/tags', timeout=2)
                    if response.status_code == 200:
                        print("✅ Ollama API is ready")
                        return True
                except:
                    pass
                time.sleep(2)
                print(f"Waiting for Ollama... ({i+1}/30)")
            
            print("⚠️ Ollama started but API not responding")
            return False
        else:
            print(f"❌ Failed to start Ollama Docker: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Ollama Docker: {e}")
        return False

def pull_model():
    """Pull llama3-chatqa model"""
    try:
        print("Pulling llama3-chatqa model...")
        cmd = ['docker', 'exec', 'ollama', 'ollama', 'pull', 'llama3-chatqa']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model pulled successfully")
            return True
        else:
            print(f"⚠️ Model pull had issues: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def test_chat():
    """Test chat functionality"""
    try:
        payload = {
            "model": "llama3-chatqa",
            "prompt": "Hello, are you working?",
            "stream": False
        }
        
        response = requests.post('http://127.0.0.1:11434/api/generate', 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat test successful: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Chat test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🐳 Testing Ollama Docker setup...")
    
    if not check_docker():
        print("❌ Docker not available. Using fallback mode.")
        return False
    
    if start_ollama_docker():
        if pull_model():
            if test_chat():
                print("🎉 Ollama Docker setup complete and working!")
                return True
    
    print("⚠️ Ollama Docker setup incomplete, using fallback mode")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)