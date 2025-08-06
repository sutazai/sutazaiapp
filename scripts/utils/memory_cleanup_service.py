#!/usr/bin/env python3
"""
Memory Cleanup Service
Automatic memory management and garbage collection for SutazAI system using small models
"""

import os
import sys
import time
import psutil
import logging
import signal
import subprocess
from threading import Thread, Event
from typing import Dict, List
import requests
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryCleanupService:
    def __init__(self):
        """Initialize Memory Cleanup Service optimized for small models"""
        self.running = True
        self.cleanup_interval = int(os.getenv('CLEANUP_INTERVAL', 30))  # 30 seconds
        self.memory_warning_threshold = int(os.getenv('MEMORY_WARNING_THRESHOLD', 80))
        self.memory_critical_threshold = int(os.getenv('MEMORY_CRITICAL_THRESHOLD', 90))
        self.emergency_threshold = int(os.getenv('EMERGENCY_THRESHOLD', 95))
        
        # Small model configuration
        self.small_models = ['gpt-oss2.5:3b', 'gpt-oss.2:3b', 'gpt-oss2.5-coder:3b']
        self.max_model_memory_gb = 3.0  # Max memory per small model
        self.system_reserved_gb = 4.0   # Reserve 4GB for system
        
        self.cleanup_event = Event()
        
        # Ollama connection
        self.ollama_url = "http://localhost:11434"
        
        logger.info("Memory Cleanup Service initialized for small model system")
        logger.info(f"Small models configured: {', '.join(self.small_models)}")
    
    def get_memory_info(self) -> Dict:
        """Get detailed memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'free_gb': round(memory.free / (1024**3), 2),
            'percent': memory.percent,
            'cached_gb': round(memory.cached / (1024**3), 2),
            'buffers_gb': round(memory.buffers / (1024**3), 2)
        }
    
    def get_process_memory_usage(self) -> List[Dict]:
        """Get top memory consuming processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'memory_info']):
            try:
                pinfo = proc.info
                if pinfo['memory_percent'] > 1.0:  # Only processes using >1% memory
                    processes.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'],
                        'memory_percent': round(pinfo['memory_percent'], 2),
                        'memory_mb': round(pinfo['memory_info'].rss / (1024**2), 2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:10]
    
    def get_ollama_status(self) -> Dict:
        """Get Ollama model status optimized for small models"""
        try:
            # Get loaded models
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            loaded_models = []
            if response.status_code == 200:
                data = response.json()
                loaded_models = data.get('models', [])
            
            # Get available models
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            available_models = []
            if response.status_code == 200:
                data = response.json()
                available_models = data.get('models', [])
            
            # Calculate memory usage for small models
            total_model_memory = 0
            for model in loaded_models:
                model_name = model.get('name', '')
                # Estimate memory for small models (3B models ~ 2GB)
                if any(sm in model_name for sm in self.small_models):
                    total_model_memory += 2.0  # 2GB per small model
                else:
                    total_model_memory += 4.0  # 4GB for other models
            
            return {
                'loaded_models': loaded_models,
                'loaded_count': len(loaded_models),
                'available_models': len(available_models),
                'estimated_memory_gb': total_model_memory,
                'small_models_loaded': [m for m in loaded_models 
                                      if any(sm in m.get('name', '') for sm in self.small_models)]
            }
        except Exception as e:
            logger.error(f"Failed to get Ollama status: {e}")
            return {'loaded_models': [], 'loaded_count': 0, 'estimated_memory_gb': 0.0}
    
    def unload_non_small_models(self) -> int:
        """Unload models that are not small models"""
        ollama_status = self.get_ollama_status()
        unloaded_count = 0
        
        for model in ollama_status['loaded_models']:
            model_name = model.get('name', '')
            
            # Keep small models, unload others
            if not any(sm in model_name for sm in self.small_models):
                try:
                    payload = {"name": model_name, "keep_alive": 0}
                    response = requests.post(f"{self.ollama_url}/api/generate", 
                                           json=payload, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"Unloaded non-small model: {model_name}")
                        unloaded_count += 1
                        time.sleep(2)  # Wait between unloads
                except Exception as e:
                    logger.error(f"Failed to unload model {model_name}: {e}")
        
        return unloaded_count
    
    def force_garbage_collection(self):
        """Force garbage collection and system cleanup"""
        try:
            # Python garbage collection
            import gc
            collected = gc.collect()
            logger.info(f"Python GC collected {collected} objects")
            
            # System memory cleanup
            try:
                subprocess.run(['sync'], check=False, timeout=5)
                # Drop caches (requires appropriate permissions)
                subprocess.run(['sh', '-c', 'echo 1 > /proc/sys/vm/drop_caches'], 
                             check=False, timeout=5)
                logger.info("System memory caches dropped")
            except Exception as e:
                logger.debug(f"Could not drop system caches: {e}")
            
            # Docker system cleanup
            try:
                subprocess.run(['docker', 'system', 'prune', '-f'], 
                             check=False, timeout=30, capture_output=True)
                logger.info("Docker system cleanup completed")
            except Exception as e:
                logger.debug(f"Docker cleanup failed: {e}")
                
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
    
    def emergency_memory_cleanup(self):
        """Emergency memory cleanup when system is critically low"""
        logger.warning("EMERGENCY: Performing aggressive memory cleanup for small model system")
        
        # 1. Unload all non-small models immediately
        unloaded = self.unload_non_small_models()
        logger.warning(f"Emergency unloaded {unloaded} non-small models")
        
        # 2. Force garbage collection
        self.force_garbage_collection()
        
        # 3. Kill memory-heavy non-essential processes
        self.kill_memory_heavy_processes()
        
        # 4. Wait and check if memory improved
        time.sleep(5)
        memory_info = self.get_memory_info()
        logger.warning(f"After emergency cleanup: Memory usage {memory_info['percent']:.1f}%")
    
    def kill_memory_heavy_processes(self):
        """Kill non-essential memory-heavy processes"""
        processes = self.get_process_memory_usage()
        killed_count = 0
        
        # Kill processes using >10% memory that are not essential
        essential_processes = ['ollama', 'postgres', 'redis', 'systemd', 'kernel', 'init']
        
        for proc in processes:
            if proc['memory_percent'] > 10.0 and killed_count < 3:
                proc_name = proc['name'].lower()
                
                # Don't kill essential processes
                if not any(essential in proc_name for essential in essential_processes):
                    try:
                        os.kill(proc['pid'], signal.SIGTERM)
                        logger.warning(f"Killed memory-heavy process: {proc['name']} "
                                     f"(PID: {proc['pid']}, Memory: {proc['memory_percent']:.1f}%)")
                        killed_count += 1
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Failed to kill process {proc['name']}: {e}")
        
        return killed_count
    
    def optimize_for_small_models(self):
        """Optimize system specifically for small models"""
        ollama_status = self.get_ollama_status()
        
        # Ensure only small models are loaded
        if ollama_status['loaded_count'] > 1:
            logger.info("Multiple models loaded, optimizing for single small model")
            self.unload_non_small_models()
        
        # Check if loaded models are actually small models
        non_small_loaded = [m for m in ollama_status['loaded_models'] 
                           if not any(sm in m.get('name', '') for sm in self.small_models)]
        
        if non_small_loaded:
            logger.info(f"Non-small models detected: {[m.get('name') for m in non_small_loaded]}")
            self.unload_non_small_models()
    
    def run_cleanup_cycle(self):
        """Run a complete cleanup cycle"""
        memory_info = self.get_memory_info()
        ollama_status = self.get_ollama_status()
        
        logger.info(f"Memory: {memory_info['percent']:.1f}% "
                   f"({memory_info['used_gb']:.1f}GB/{memory_info['total_gb']:.1f}GB), "
                   f"Models loaded: {ollama_status['loaded_count']}, "
                   f"Est. model memory: {ollama_status['estimated_memory_gb']:.1f}GB")
        
        # Emergency cleanup
        if memory_info['percent'] >= self.emergency_threshold:
            self.emergency_memory_cleanup()
            return
        
        # Critical memory management
        if memory_info['percent'] >= self.memory_critical_threshold:
            logger.warning(f"Critical memory usage: {memory_info['percent']:.1f}%")
            
            # Unload non-small models
            self.unload_non_small_models()
            
            # Force garbage collection
            self.force_garbage_collection()
            
            # Optimize for small models
            self.optimize_for_small_models()
        
        # Warning level cleanup
        elif memory_info['percent'] >= self.memory_warning_threshold:
            logger.info(f"High memory usage: {memory_info['percent']:.1f}%")
            
            # Ensure we're using small models only
            self.optimize_for_small_models()
            
            # Light garbage collection
            import gc
            gc.collect()
        
        # Regular maintenance
        else:
            logger.debug(f"Normal memory usage: {memory_info['percent']:.1f}%")
            
            # Still ensure we're optimized for small models
            if ollama_status['loaded_count'] > 1:
                self.optimize_for_small_models()
    
    def start_service(self):
        """Start the memory cleanup service"""
        logger.info("Starting Memory Cleanup Service for small model system")
        
        try:
            while self.running:
                self.run_cleanup_cycle()
                
                # Wait for next cycle or stop event
                if self.cleanup_event.wait(timeout=self.cleanup_interval):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping service...")
        except Exception as e:
            logger.error(f"Unexpected error in service: {e}")
        finally:
            self.stop_service()
    
    def stop_service(self):
        """Stop the service"""
        self.running = False
        self.cleanup_event.set()
        logger.info("Memory Cleanup Service stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main function"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start service
    service = MemoryCleanupService()
    service.start_service()

if __name__ == "__main__":
    main()