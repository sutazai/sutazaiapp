#!/usr/bin/env python3
"""
Ollama Optimization Script
Implements Phase 1 optimizations for the SutazAI system
"""

import os
import sys
import yaml
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

class OllamaOptimizer:
    """Optimize Ollama configuration for TinyLlama model"""
    
    def __init__(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.docker_compose_path = self.project_root / "docker-compose.yml"
        self.backup_path = self.project_root / "docker-compose.yml.backup"
        
    def backup_config(self):
        """Create backup of current docker-compose.yml"""
        print("Creating backup of docker-compose.yml...")
        subprocess.run([
            "cp", 
            str(self.docker_compose_path), 
            str(self.backup_path)
        ], check=True)
        print(f"Backup created at: {self.backup_path}")
        
    def load_compose_config(self) -> Dict[str, Any]:
        """Load current docker-compose configuration"""
        with open(self.docker_compose_path, 'r') as f:
            return yaml.safe_load(f)
            
    def optimize_ollama_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations to Ollama service configuration"""
        
        if 'services' not in config or 'ollama' not in config['services']:
            print("ERROR: Ollama service not found in docker-compose.yml")
            return config
            
        ollama = config['services']['ollama']
        
        print("Applying Ollama optimizations...")
        
        # Optimize resource limits
        if 'deploy' not in ollama:
            ollama['deploy'] = {}
        if 'resources' not in ollama['deploy']:
            ollama['deploy']['resources'] = {}
            
        # Update resource limits (reduced from 20G/10CPU)
        ollama['deploy']['resources']['limits'] = {
            'cpus': '4',     # Reduced from 10
            'memory': '4G'   # Reduced from 20G
        }
        
        ollama['deploy']['resources']['reservations'] = {
            'cpus': '2',     # Reduced from 4
            'memory': '2G'   # Reduced from 8G
        }
        
        # Optimize environment variables
        if 'environment' not in ollama:
            ollama['environment'] = {}
            
        # Update performance-related environment variables
        env_updates = {
            'OLLAMA_NUM_PARALLEL': '4',      # Reduced from 50
            'OLLAMA_NUM_THREADS': '4',       # Reduced from 10
            'OLLAMA_MAX_LOADED_MODELS': '1', # Reduced from 3
            'OLLAMA_KEEP_ALIVE': '5m',       # Reduced from 10m
            'OLLAMA_FLASH_ATTENTION': '0',   # Disable for CPU
            'OLLAMA_DEBUG': 'false',
            'OLLAMA_HOST': '0.0.0.0:11434',  # Use standard port internally
        }
        
        # Apply environment updates
        for key, value in env_updates.items():
            if isinstance(ollama['environment'], list):
                # Handle list format
                ollama['environment'] = {
                    item.split('=')[0]: item.split('=', 1)[1] 
                    for item in ollama['environment']
                }
            ollama['environment'][key] = value
            
        print("Optimizations applied:")
        print("  - Memory limit: 20G -> 4G")
        print("  - CPU limit: 10 -> 4")
        print("  - Parallel requests: 50 -> 4")
        print("  - Max loaded models: 3 -> 1")
        print("  - Keep alive: 10m -> 5m")
        
        return config
        
    def save_compose_config(self, config: Dict[str, Any]):
        """Save updated docker-compose configuration"""
        print("Saving optimized configuration...")
        with open(self.docker_compose_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print("Configuration saved successfully")
        
    def restart_ollama(self):
        """Restart Ollama service with new configuration"""
        print("\nRestarting Ollama service...")
        
        # Stop Ollama
        print("Stopping Ollama container...")
        subprocess.run([
            "docker-compose", 
            "-f", str(self.docker_compose_path),
            "stop", "ollama"
        ], check=True)
        
        # Remove old container
        print("Removing old container...")
        subprocess.run([
            "docker-compose",
            "-f", str(self.docker_compose_path),
            "rm", "-f", "ollama"
        ], check=True)
        
        # Start with new config
        print("Starting Ollama with optimized configuration...")
        subprocess.run([
            "docker-compose",
            "-f", str(self.docker_compose_path),
            "up", "-d", "ollama"
        ], check=True)
        
        print("Waiting for Ollama to be ready...")
        time.sleep(10)
        
    def verify_optimization(self):
        """Verify that optimizations are applied"""
        print("\nVerifying optimizations...")
        
        # Check container stats
        result = subprocess.run([
            "docker", "stats", "--no-stream",
            "--format", "json",
            "sutazai-ollama"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            stats = json.loads(result.stdout)
            print(f"Container stats:")
            print(f"  - Memory usage: {stats.get('MemUsage', 'N/A')}")
            print(f"  - CPU usage: {stats.get('CPUPerc', 'N/A')}")
            
        # Check if model is still loaded
        result = subprocess.run([
            "docker", "exec", "sutazai-ollama",
            "ollama", "list"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\nLoaded models:")
            print(result.stdout)
            
        # Test generation
        print("\nTesting text generation...")
        result = subprocess.run([
            "curl", "-s", "-X", "POST",
            "http://127.0.0.1:10104/api/generate",
            "-d", json.dumps({
                "model": "tinyllama",
                "prompt": "Hello, how are you?",
                "stream": False,
                "options": {
                    "num_predict": 20
                }
            })
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                print(f"Generation successful: {response.get('response', '')[:100]}...")
            except:
                print("Generation test failed")
        
    def rollback(self):
        """Rollback to previous configuration"""
        print("\nRolling back to previous configuration...")
        if self.backup_path.exists():
            subprocess.run([
                "cp",
                str(self.backup_path),
                str(self.docker_compose_path)
            ], check=True)
            print("Configuration rolled back")
            self.restart_ollama()
        else:
            print("No backup found")
            
    def run(self):
        """Execute optimization process"""
        print("=" * 60)
        print("OLLAMA OPTIMIZATION FOR SUTAZAI")
        print("=" * 60)
        
        try:
            # Step 1: Backup
            self.backup_config()
            
            # Step 2: Load and optimize
            config = self.load_compose_config()
            optimized_config = self.optimize_ollama_config(config)
            
            # Step 3: Save
            self.save_compose_config(optimized_config)
            
            # Step 4: Restart
            self.restart_ollama()
            
            # Step 5: Verify
            self.verify_optimization()
            
            print("\n" + "=" * 60)
            print("OPTIMIZATION COMPLETE!")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Monitor memory usage: docker stats sutazai-ollama")
            print("2. Check logs: docker logs -f sutazai-ollama")
            print("3. If issues occur, rollback: python3 optimize_ollama.py --rollback")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Rolling back changes...")
            self.rollback()
            sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize Ollama for SutazAI")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous config")
    args = parser.parse_args()
    
    optimizer = OllamaOptimizer()
    
    if args.rollback:
        optimizer.rollback()
    else:
        optimizer.run()