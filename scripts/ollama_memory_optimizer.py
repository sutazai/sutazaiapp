#!/usr/bin/env python3
"""
Ollama Memory Optimizer
Prevents system freezing by managing Ollama models and memory usage
"""

import os
import sys
import time
import json
import psutil
import requests
import logging
import threading
from typing import Dict, List, Optional
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaMemoryOptimizer:
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/ollama_optimization.yaml"):
        """Initialize the Ollama Memory Optimizer"""
        self.config = self.load_config(config_path)
        self.ollama_url = "http://localhost:11434"
        self.running = True
        self.last_activity = {}
        
        # Memory thresholds
        self.memory_warning_threshold = self.config['monitoring']['alert_thresholds']['memory_warning']
        self.memory_critical_threshold = self.config['monitoring']['alert_thresholds']['memory_critical']
        self.emergency_threshold = self.config['ollama_config']['emergency_settings']['emergency_unload_threshold']
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration if config file is not available"""
        return {
            'ollama_config': {
                'memory_limits': {'max_model_memory': '6G'},
                'model_management': {'max_loaded_models': 1, 'unload_after_idle': '60s'},
                'emergency_settings': {'emergency_unload_threshold': 92}
            },
            'monitoring': {
                'memory_check_interval': 10,
                'alert_thresholds': {'memory_warning': 80, 'memory_critical': 90}
            }
        }
    
    def get_system_memory_info(self) -> Dict:
        """Get current system memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent': memory.percent,
            'free_gb': round(memory.free / (1024**3), 2)
        }
    
    def get_loaded_models(self) -> List[Dict]:
        """Get currently loaded models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")
        return []
    
    def get_available_models(self) -> List[Dict]:
        """Get all available models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
        return []
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model"""
        try:
            payload = {"name": model_name, "keep_alive": 0}
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Successfully unloaded model: {model_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
        return False
    
    def unload_all_models(self) -> int:
        """Unload all loaded models"""
        loaded_models = self.get_loaded_models()
        unloaded_count = 0
        
        for model in loaded_models:
            model_name = model.get('name', model.get('model', ''))
            if model_name and self.unload_model(model_name):
                unloaded_count += 1
                time.sleep(2)  # Wait between unloads
                
        return unloaded_count
    
    def get_model_memory_usage(self, model_info: Dict) -> float:
        """Estimate model memory usage in GB"""
        # Extract size from model info
        size_bytes = model_info.get('size', 0)
        if isinstance(size_bytes, (int, float)):
            return round(size_bytes / (1024**3), 2)
        return 0.0
    
    def should_unload_model(self, model: Dict) -> bool:
        """Determine if a model should be unloaded based on activity"""
        model_name = model.get('name', model.get('model', ''))
        if not model_name:
            return True
            
        # Check last activity
        last_active = self.last_activity.get(model_name, time.time())
        idle_time = time.time() - last_active
        
        # Get idle timeout from config
        idle_timeout_str = self.config['ollama_config']['model_management']['unload_after_idle']
        idle_timeout = self.parse_time_string(idle_timeout_str)
        
        return idle_time > idle_timeout
    
    def parse_time_string(self, time_str: str) -> int:
        """Parse time string like '60s', '5m' to seconds"""
        if time_str.endswith('s'):
            return int(time_str[:-1])
        elif time_str.endswith('m'):
            return int(time_str[:-1]) * 60
        elif time_str.endswith('h'):
            return int(time_str[:-1]) * 3600
        else:
            return 60  # Default to 60 seconds
    
    def optimize_memory_usage(self):
        """Main memory optimization logic"""
        memory_info = self.get_system_memory_info()
        loaded_models = self.get_loaded_models()
        
        logger.info(f"Memory usage: {memory_info['percent']:.1f}% "
                   f"({memory_info['used_gb']:.1f}GB/{memory_info['total_gb']:.1f}GB)")
        
        if loaded_models:
            logger.info(f"Loaded models: {len(loaded_models)}")
            for model in loaded_models:
                model_name = model.get('name', model.get('model', 'unknown'))
                model_size = self.get_model_memory_usage(model)
                logger.info(f"  - {model_name}: {model_size:.1f}GB")
        
        # Emergency unload if memory is critically high
        if memory_info['percent'] >= self.emergency_threshold:
            logger.warning(f"Emergency: Memory usage at {memory_info['percent']:.1f}%! "
                          "Unloading all models.")
            unloaded = self.unload_all_models()
            logger.warning(f"Emergency unloaded {unloaded} models")
            
            # Force garbage collection
            self.force_garbage_collection()
            return
        
        # Regular optimization
        if memory_info['percent'] >= self.memory_critical_threshold:
            logger.warning(f"High memory usage: {memory_info['percent']:.1f}%")
            
            # Unload idle models
            models_to_unload = [m for m in loaded_models if self.should_unload_model(m)]
            for model in models_to_unload:
                model_name = model.get('name', model.get('model', ''))
                if self.unload_model(model_name):
                    logger.info(f"Unloaded idle model: {model_name}")
                    time.sleep(1)
        
        elif memory_info['percent'] >= self.memory_warning_threshold:
            logger.info(f"Moderate memory usage: {memory_info['percent']:.1f}%")
            
            # Check if we have too many models loaded
            max_models = self.config['ollama_config']['model_management']['max_loaded_models']
            if len(loaded_models) > max_models:
                models_to_unload = loaded_models[max_models:]
                for model in models_to_unload:
                    model_name = model.get('name', model.get('model', ''))
                    if self.unload_model(model_name):
                        logger.info(f"Unloaded excess model: {model_name}")
                        time.sleep(1)
    
    def force_garbage_collection(self):
        """Force garbage collection in Python and system"""
        import gc
        gc.collect()
        
        # Try to trigger system memory cleanup
        try:
            os.system("sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true")
        except:
            pass
        
        logger.info("Forced garbage collection and memory cleanup")
    
    def monitor_ollama_requests(self):
        """Monitor Ollama API requests to track model activity"""
        # This would be implemented with request intercepting
        # For now, we'll update activity when we detect loaded models
        loaded_models = self.get_loaded_models()
        current_time = time.time()
        
        for model in loaded_models:
            model_name = model.get('name', model.get('model', ''))
            if model_name:
                self.last_activity[model_name] = current_time
    
    def run_optimization_cycle(self):
        """Run a single optimization cycle"""
        try:
            self.monitor_ollama_requests()
            self.optimize_memory_usage()
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
    
    def start_monitoring(self):
        """Start the continuous monitoring loop"""
        logger.info("Starting Ollama Memory Optimizer")
        check_interval = self.config['monitoring']['memory_check_interval']
        
        while self.running:
            try:
                self.run_optimization_cycle()
                time.sleep(check_interval)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(check_interval)
    
    def stop(self):
        """Stop the optimizer"""
        self.running = False
        logger.info("Ollama Memory Optimizer stopped")

def main():
    """Main function"""
    optimizer = OllamaMemoryOptimizer()
    
    try:
        optimizer.start_monitoring()
    except KeyboardInterrupt:
        optimizer.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()