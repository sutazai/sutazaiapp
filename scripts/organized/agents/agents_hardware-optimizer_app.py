#!/usr/bin/env python3
"""
Hardware Resource Optimizer Agent
Dynamically manages system resources and optimizes container allocation
"""

import os
import time
import logging
import psutil
from flask import Flask, jsonify
from threading import Thread
import schedule
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class HardwareResourceOptimizer:
    def __init__(self):
        try:
            import docker
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False
        
        # Small model system configuration
        self.max_memory_mb = int(os.getenv('MAX_MEMORY_MB', 6144))  # 6GB for small models
        self.cpu_threshold = int(os.getenv('CPU_THRESHOLD', 75))    # Lower threshold for safety
        self.memory_threshold = int(os.getenv('MEMORY_THRESHOLD', 80))  # Conservative threshold
        self.optimization_interval = int(os.getenv('OPTIMIZATION_INTERVAL', 30))  # More frequent
        
        # Small model defaults
        self.small_model_mode = os.getenv('SMALL_MODEL_MODE', 'true').lower() == 'true'
        self.default_models = os.getenv('DEFAULT_MODELS', 'qwen2.5:3b,llama3.2:3b').split(',')
        self.max_model_memory_gb = 3.0  # Max memory per small model
        
        # Ollama connection for small model management
        self.ollama_url = "http://localhost:11434"
        
        logger.info(f"Hardware Resource Optimizer initialized for small model system")
        logger.info(f"Small model mode: {self.small_model_mode}")
        logger.info(f"Default models: {', '.join(self.default_models)}")
        
    def get_system_resources(self):
        """Get current system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk_percent,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return None
    
    def get_container_resources(self):
        """Get resource usage for all containers"""
        container_stats = []
        
        if not self.docker_available or not self.docker_client:
            logger.warning("Docker client not available, skipping container stats")
            return container_stats
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage safely
                    cpu_percent = 0.0
                    if ('cpu_stats' in stats and 'precpu_stats' in stats and
                        'cpu_usage' in stats['cpu_stats'] and 'cpu_usage' in stats['precpu_stats']):
                        
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                   stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                        
                        if system_delta > 0:
                            cpu_count = len(stats['cpu_stats']['cpu_usage'].get('percpu_usage', [psutil.cpu_count()]))
                            cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0
                    
                    # Calculate memory usage safely
                    memory_usage = 0
                    memory_percent = 0.0
                    if 'memory_stats' in stats and 'usage' in stats['memory_stats']:
                        memory_usage = stats['memory_stats']['usage']
                        memory_limit = stats['memory_stats'].get('limit', 0)
                        if memory_limit > 0:
                            memory_percent = (memory_usage / memory_limit) * 100.0
                    
                    container_stats.append({
                        'name': container.name,
                        'id': container.short_id,
                        'cpu_percent': min(cpu_percent, 100.0),  # Cap at 100%
                        'memory_usage_mb': memory_usage / (1024**2),
                        'memory_percent': min(memory_percent, 100.0),  # Cap at 100%
                        'status': container.status
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to get stats for {container.name}: {e}")
                    # Add basic container info even if stats fail
                    container_stats.append({
                        'name': container.name,
                        'id': container.short_id,
                        'cpu_percent': 0.0,
                        'memory_usage_mb': 0.0,
                        'memory_percent': 0.0,
                        'status': container.status
                    })
                    
        except Exception as e:
            logger.error(f"Failed to get container resources: {e}")
            
        return container_stats
    
    def optimize_container_resources(self):
        """Optimize container resource allocation"""
        system_resources = self.get_system_resources()
        container_stats = self.get_container_resources()
        
        if not system_resources or not container_stats:
            return
        
        # Check if system is under pressure
        if (system_resources['cpu_percent'] > self.cpu_threshold or 
            system_resources['memory_percent'] > self.memory_threshold):
            
            logger.warning(f"System under pressure - CPU: {system_resources['cpu_percent']:.1f}%, "
                          f"Memory: {system_resources['memory_percent']:.1f}%")
            
            # Find high-resource containers
            high_cpu_containers = [c for c in container_stats if c['cpu_percent'] > 50]
            high_memory_containers = [c for c in container_stats if c['memory_percent'] > 70]
            
            # Implement resource throttling for non-essential containers
            self.throttle_non_essential_containers(high_cpu_containers, high_memory_containers)
        
        # Log optimization results
        logger.info(f"Resource optimization - System CPU: {system_resources['cpu_percent']:.1f}%, "
                   f"Memory: {system_resources['memory_percent']:.1f}%, "
                   f"Active containers: {len(container_stats)}")
    
    def throttle_non_essential_containers(self, high_cpu_containers, high_memory_containers):
        """Throttle non-essential containers to free up resources"""
        non_essential_patterns = ['tier2', 'tier3', 'test', 'dev']
        
        for container_info in high_cpu_containers + high_memory_containers:
            container_name = container_info['name'].lower()
            
            # Check if container is non-essential
            if any(pattern in container_name for pattern in non_essential_patterns):
                try:
                    container = self.docker_client.containers.get(container_info['id'])
                    
                    # Update container with stricter resource limits
                    container.update(
                        cpu_period=100000,
                        cpu_quota=50000,  # Limit to 50% CPU
                        mem_limit=f"{int(container_info['memory_usage_mb'] * 0.8)}m"
                    )
                    
                    logger.info(f"Throttled container {container_name} due to high resource usage")
                    
                except Exception as e:
                    logger.error(f"Failed to throttle container {container_name}: {e}")
    
    def scale_containers_based_on_load(self):
        """Scale container replicas based on system load"""
        system_resources = self.get_system_resources()
        
        if not system_resources:
            return
        
        # If system has plenty of resources, consider scaling up essential services
        if (system_resources['cpu_percent'] < 30 and 
            system_resources['memory_percent'] < 50):
            
            logger.info("System has abundant resources - considering scale up")
            # Implementation for scaling up would go here
            
        # If system is overloaded, scale down non-essential services
        elif (system_resources['cpu_percent'] > 90 or 
              system_resources['memory_percent'] > 95):
            
            logger.warning("System critically overloaded - scaling down")
            self.emergency_scale_down()
    
    def emergency_scale_down(self):
        """Emergency scale down of non-essential containers"""
        try:
            containers = self.docker_client.containers.list()
            non_essential_containers = [
                c for c in containers 
                if any(pattern in c.name.lower() for pattern in ['tier3', 'test', 'dev', 'optional'])
            ]
            
            for container in non_essential_containers[:3]:  # Stop up to 3 containers
                logger.warning(f"Emergency stopping container: {container.name}")
                container.stop()
                
        except Exception as e:
            logger.error(f"Failed emergency scale down: {e}")
    
    def get_ollama_status(self):
        """Get Ollama model status for small model optimization"""
        try:
            import requests
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                # Categorize models
                small_models = []
                large_models = []
                
                for model in models:
                    model_name = model.get('name', '')
                    if any(default in model_name for default in self.default_models):
                        small_models.append(model)
                    else:
                        large_models.append(model)
                
                return {
                    'loaded_models': models,
                    'small_models': small_models,
                    'large_models': large_models,
                    'total_loaded': len(models),
                    'memory_efficient': len(large_models) == 0
                }
        except Exception as e:
            logger.error(f"Failed to get Ollama status: {e}")
            
        return {
            'loaded_models': [],
            'small_models': [],
            'large_models': [],
            'total_loaded': 0,
            'memory_efficient': True
        }
    
    def optimize_ollama_for_small_models(self):
        """Optimize Ollama specifically for small models"""
        if not self.small_model_mode:
            return
            
        ollama_status = self.get_ollama_status()
        
        # Unload large models if any are loaded
        if ollama_status['large_models']:
            logger.warning(f"Large models detected: {[m.get('name') for m in ollama_status['large_models']]}")
            self.unload_large_models(ollama_status['large_models'])
        
        # Ensure only one small model is loaded at a time
        if len(ollama_status['small_models']) > 1:
            logger.info("Multiple small models loaded, keeping only the first one")
            models_to_unload = ollama_status['small_models'][1:]
            for model in models_to_unload:
                self.unload_model(model.get('name', ''))
        
        # Log optimization results
        if ollama_status['memory_efficient']:
            logger.info("Ollama optimized for small models - memory efficient mode active")
        else:
            logger.warning("Ollama not optimized - large models still loaded")
    
    def unload_large_models(self, large_models):
        """Unload large models to free memory"""
        for model in large_models:
            model_name = model.get('name', '')
            if self.unload_model(model_name):
                logger.info(f"Unloaded large model: {model_name}")
                time.sleep(2)  # Wait between unloads
    
    def unload_model(self, model_name):
        """Unload a specific model from Ollama"""
        try:
            import requests
            payload = {"name": model_name, "keep_alive": 0}
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def run_optimization_cycle(self):
        """Run a complete optimization cycle for small model system"""
        # Small model optimization first
        if self.small_model_mode:
            self.optimize_ollama_for_small_models()
        
        # Standard container optimization
        self.optimize_container_resources()
        self.scale_containers_based_on_load()

# Global instance
optimizer = HardwareResourceOptimizer()

@app.route('/health')
def health():
    """Health check endpoint"""
    system_resources = optimizer.get_system_resources()
    
    return jsonify({
        'status': 'healthy',
        'system_resources': system_resources,
        'timestamp': time.time()
    })

@app.route('/resources')
def resources():
    """Get detailed resource information"""
    return jsonify({
        'system': optimizer.get_system_resources(),
        'containers': optimizer.get_container_resources()
    })

@app.route('/optimize')
def optimize():
    """Trigger optimization cycle"""
    try:
        optimizer.run_optimization_cycle()
        return jsonify({'status': 'success', 'message': 'Optimization cycle completed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/emergency-scale-down')
def emergency_scale_down():
    """Trigger emergency scale down"""
    try:
        optimizer.emergency_scale_down()
        return jsonify({'status': 'success', 'message': 'Emergency scale down completed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/ollama-status')
def ollama_status():
    """Get Ollama model status for small model optimization"""
    status = optimizer.get_ollama_status()
    return jsonify({
        'status': 'success',
        'ollama_status': status,
        'small_model_mode': optimizer.small_model_mode,
        'default_models': optimizer.default_models,
        'timestamp': time.time()
    })

@app.route('/optimize-small-models')
def optimize_small_models():
    """Trigger small model optimization"""
    try:
        optimizer.optimize_ollama_for_small_models()
        return jsonify({'status': 'success', 'message': 'Small model optimization completed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/unload-large-models')
def unload_large_models():
    """Unload all large models"""
    try:
        ollama_status = optimizer.get_ollama_status()
        if ollama_status['large_models']:
            optimizer.unload_large_models(ollama_status['large_models'])
            return jsonify({
                'status': 'success', 
                'message': f"Unloaded {len(ollama_status['large_models'])} large models"
            })
        else:
            return jsonify({'status': 'success', 'message': 'No large models to unload'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/system-summary')
def system_summary():
    """Get comprehensive system summary for small model operation"""
    try:
        system_resources = optimizer.get_system_resources()
        container_stats = optimizer.get_container_resources()
        ollama_status = optimizer.get_ollama_status()
        
        # Calculate memory efficiency
        memory_efficient = (
            system_resources['memory_percent'] < optimizer.memory_threshold and
            ollama_status['memory_efficient'] and
            len(ollama_status['loaded_models']) <= 1
        )
        
        return jsonify({
            'status': 'success',
            'system_resources': system_resources,
            'container_count': len(container_stats),
            'ollama_status': ollama_status,
            'memory_efficient': memory_efficient,
            'small_model_mode': optimizer.small_model_mode,
            'optimization_needed': system_resources['memory_percent'] > optimizer.memory_threshold,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def run_scheduler():
    """Run scheduled optimization tasks"""
    schedule.every(1).minutes.do(optimizer.run_optimization_cycle)
    
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == '__main__':
    # Start scheduler in background
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    app.run(host='0.0.0.0', port=8523, debug=False)