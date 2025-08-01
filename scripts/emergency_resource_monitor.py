#!/usr/bin/env python3
"""
Emergency Resource Monitor
Prevents system freezing by monitoring and controlling resource usage
"""

import time
import subprocess
import requests
import json
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmergencyResourceMonitor:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.memory_threshold = 80  # Percentage
        self.emergency_threshold = 90  # Critical threshold
        self.small_models = ["tinyllama:1.1b", "qwen2.5:3b"]
        self.preferred_model = "tinyllama:1.1b"
        
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            result = subprocess.run(['free'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            mem_line = lines[1].split()
            total = int(mem_line[1])
            used = int(mem_line[2])
            return (used / total) * 100
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")
            return []
    
    def unload_all_models(self):
        """Unload all models from memory"""
        try:
            models = self.get_loaded_models()
            for model in models:
                self.unload_model(model)
            logger.info("All models unloaded")
        except Exception as e:
            logger.error(f"Failed to unload all models: {e}")
    
    def unload_model(self, model_name: str):
        """Unload a specific model"""
        try:
            payload = {
                "model": model_name,
                "prompt": "",
                "stream": False,
                "keep_alive": 0
            }
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            if response.status_code == 200:
                logger.info(f"Unloaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
    
    def load_emergency_model(self):
        """Load the smallest emergency model"""
        try:
            payload = {
                "model": self.preferred_model,
                "prompt": "System ready",
                "stream": False,
                "keep_alive": "30s"
            }
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            if response.status_code == 200:
                logger.info(f"Emergency model loaded: {self.preferred_model}")
        except Exception as e:
            logger.error(f"Failed to load emergency model: {e}")
    
    def emergency_resource_cleanup(self):
        """Emergency cleanup to prevent system freezing"""
        logger.warning("EMERGENCY: Performing resource cleanup to prevent system freeze")
        
        # 1. Unload all models immediately
        self.unload_all_models()
        
        # 2. Wait for memory to clear
        time.sleep(5)
        
        # 3. Load only the smallest model
        self.load_emergency_model()
        
        # 4. Stop non-essential containers if memory is still high
        memory_usage = self.get_memory_usage()
        if memory_usage > self.emergency_threshold:
            self.stop_non_essential_containers()
    
    def stop_non_essential_containers(self):
        """Stop non-essential containers to free resources"""
        non_essential = [
            "sutazai-agentgpt", "sutazai-langflow", "sutazai-flowise",
            "sutazai-dify-api", "sutazai-bigagi", "sutazai-crewaigui"
        ]
        
        for container in non_essential:
            try:
                subprocess.run(['docker', 'stop', container], 
                             capture_output=True, check=False)
                logger.info(f"Stopped non-essential container: {container}")
            except Exception as e:
                logger.error(f"Failed to stop container {container}: {e}")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting emergency resource monitor...")
        
        while True:
            try:
                memory_usage = self.get_memory_usage()
                logger.info(f"Memory usage: {memory_usage:.1f}%")
                
                if memory_usage > self.emergency_threshold:
                    self.emergency_resource_cleanup()
                elif memory_usage > self.memory_threshold:
                    logger.warning(f"High memory usage detected: {memory_usage:.1f}%")
                    # Preemptively unload larger models
                    loaded_models = self.get_loaded_models()
                    for model in loaded_models:
                        if model not in self.small_models:
                            self.unload_model(model)
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    monitor = EmergencyResourceMonitor()
    monitor.monitor_loop()