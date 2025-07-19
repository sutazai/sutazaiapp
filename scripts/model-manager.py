#!/usr/bin/env python3
"""
SutazAI Model Manager
Automated model management, cleanup, and optimization
"""

import os
import json
import time
import requests
import logging
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/sutazai-model-manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SutazAI-ModelManager')

class ModelManager:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.models_config = {
            # Priority models (always keep loaded)
            "priority": [
                {
                    "name": "llama3.2:1b",
                    "memory_mb": 1200,
                    "description": "Lightweight general model",
                    "use_cases": ["chat", "general"]
                },
                {
                    "name": "qwen2.5-coder:1.5b",
                    "memory_mb": 1500,
                    "description": "Efficient code model",
                    "use_cases": ["code", "programming"]
                }
            ],
            # Secondary models (load on demand)
            "secondary": [
                {
                    "name": "starcoder2:3b",
                    "memory_mb": 2500,
                    "description": "Advanced code model",
                    "use_cases": ["code", "architecture"]
                },
                {
                    "name": "deepseek-r1:8b",
                    "memory_mb": 6000,
                    "description": "Advanced reasoning model",
                    "use_cases": ["reasoning", "analysis"]
                }
            ],
            # Optional models (only for specific tasks)
            "optional": [
                {
                    "name": "qwen2.5:7b",
                    "memory_mb": 5000,
                    "description": "General purpose large model",
                    "use_cases": ["complex_tasks"]
                }
            ]
        }
        
        self.memory_thresholds = {
            "low": 2048,      # MB - when to load priority models only
            "medium": 4096,   # MB - when to load secondary models
            "high": 8192      # MB - when to load optional models
        }
        
        self.model_usage_stats = {}
        self.last_cleanup = datetime.now()

    def get_available_memory(self) -> int:
        """Get available system memory in MB"""
        memory = psutil.virtual_memory()
        return memory.available // (1024 * 1024)

    def get_ollama_status(self) -> Dict:
        """Get Ollama service status"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return {"status": "running", "models": response.json().get("models", [])}
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "offline", "message": str(e)}

    def get_loaded_models(self) -> List[Dict]:
        """Get currently loaded models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                return []
        except requests.exceptions.RequestException:
            return []

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        for category in ["priority", "secondary", "optional"]:
            for model in self.models_config[category]:
                if model["name"] == model_name:
                    return model
        return None

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available in Ollama"""
        status = self.get_ollama_status()
        if status["status"] == "running":
            return any(model["name"] == model_name for model in status["models"])
        return False

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                timeout=600  # 10 minutes timeout
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    def load_model(self, model_name: str) -> bool:
        """Load a model into memory"""
        try:
            # Check if model is available
            if not self.is_model_available(model_name):
                logger.info(f"Model {model_name} not available, pulling...")
                if not self.pull_model(model_name):
                    return False
            
            # Load model by making a simple request
            logger.info(f"Loading model: {model_name}")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                logger.info(f"Model {model_name} loaded successfully")
                return True
            else:
                logger.error(f"Failed to load model {model_name}: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        try:
            logger.info(f"Unloading model: {model_name}")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "keep_alive": 0
                },
                timeout=30
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False

    def unload_all_models(self):
        """Unload all models to free memory"""
        logger.info("Unloading all models")
        loaded_models = self.get_loaded_models()
        
        for model in loaded_models:
            model_name = model.get("name", "")
            if model_name:
                self.unload_model(model_name)

    def calculate_memory_strategy(self) -> Dict:
        """Calculate which models to load based on available memory"""
        available_memory = self.get_available_memory()
        strategy = {
            "available_memory": available_memory,
            "to_load": [],
            "to_unload": [],
            "strategy": "conservative"
        }
        
        # Determine strategy based on available memory
        if available_memory >= self.memory_thresholds["high"]:
            strategy["strategy"] = "aggressive"
            # Load priority + secondary + some optional
            for model in self.models_config["priority"]:
                strategy["to_load"].append(model["name"])
            for model in self.models_config["secondary"]:
                strategy["to_load"].append(model["name"])
            # Load one optional model if enough memory
            if available_memory >= 10240:  # 10GB
                strategy["to_load"].append(self.models_config["optional"][0]["name"])
                
        elif available_memory >= self.memory_thresholds["medium"]:
            strategy["strategy"] = "balanced"
            # Load priority + one secondary
            for model in self.models_config["priority"]:
                strategy["to_load"].append(model["name"])
            if self.models_config["secondary"]:
                strategy["to_load"].append(self.models_config["secondary"][0]["name"])
                
        elif available_memory >= self.memory_thresholds["low"]:
            strategy["strategy"] = "conservative"
            # Load only priority models
            for model in self.models_config["priority"]:
                strategy["to_load"].append(model["name"])
        else:
            strategy["strategy"] = "minimal"
            # Load only the smallest priority model
            if self.models_config["priority"]:
                smallest_model = min(
                    self.models_config["priority"],
                    key=lambda m: m["memory_mb"]
                )
                strategy["to_load"].append(smallest_model["name"])
        
        # Determine which models to unload
        loaded_models = self.get_loaded_models()
        loaded_names = [model.get("name", "") for model in loaded_models]
        
        for loaded_name in loaded_names:
            if loaded_name and loaded_name not in strategy["to_load"]:
                strategy["to_unload"].append(loaded_name)
        
        return strategy

    def apply_memory_strategy(self):
        """Apply memory management strategy"""
        logger.info("Applying memory management strategy")
        
        strategy = self.calculate_memory_strategy()
        logger.info(f"Memory strategy: {strategy['strategy']} "
                   f"(Available: {strategy['available_memory']}MB)")
        
        # Unload unnecessary models first
        for model_name in strategy["to_unload"]:
            self.unload_model(model_name)
            time.sleep(2)  # Brief pause between operations
        
        # Load required models
        for model_name in strategy["to_load"]:
            if not any(m.get("name") == model_name for m in self.get_loaded_models()):
                available_memory = self.get_available_memory()
                model_info = self.get_model_info(model_name)
                
                if model_info and available_memory >= model_info["memory_mb"]:
                    if self.load_model(model_name):
                        logger.info(f"Successfully loaded {model_name}")
                    else:
                        logger.warning(f"Failed to load {model_name}")
                else:
                    logger.warning(f"Insufficient memory for {model_name}")
                
                time.sleep(5)  # Pause between model loads

    def cleanup_models(self):
        """Clean up unused models and optimize storage"""
        logger.info("Starting model cleanup")
        
        # Get model usage statistics
        loaded_models = self.get_loaded_models()
        
        # Unload models that haven't been used recently
        current_time = datetime.now()
        for model in loaded_models:
            model_name = model.get("name", "")
            last_used = self.model_usage_stats.get(model_name, current_time)
            
            # Unload if not used in last 30 minutes
            if isinstance(last_used, str):
                last_used = datetime.fromisoformat(last_used)
            
            if current_time - last_used > timedelta(minutes=30):
                model_info = self.get_model_info(model_name)
                # Don't unload priority models
                if not model_info or model_info not in self.models_config["priority"]:
                    logger.info(f"Unloading unused model: {model_name}")
                    self.unload_model(model_name)

    def record_model_usage(self, model_name: str):
        """Record model usage for optimization"""
        self.model_usage_stats[model_name] = datetime.now().isoformat()

    def get_model_recommendations(self) -> Dict:
        """Get model recommendations based on current state"""
        available_memory = self.get_available_memory()
        loaded_models = [m.get("name", "") for m in self.get_loaded_models()]
        
        recommendations = {
            "memory_status": "good" if available_memory > 4096 else "low",
            "loaded_models": loaded_models,
            "suggested_actions": []
        }
        
        if available_memory < 2048:
            recommendations["suggested_actions"].append(
                "Critical memory low - consider unloading non-essential models"
            )
        elif available_memory < 4096:
            recommendations["suggested_actions"].append(
                "Memory low - consider loading smaller models"
            )
        
        # Check for missing priority models
        for model in self.models_config["priority"]:
            if model["name"] not in loaded_models:
                if available_memory >= model["memory_mb"]:
                    recommendations["suggested_actions"].append(
                        f"Consider loading priority model: {model['name']}"
                    )
        
        return recommendations

    def emergency_cleanup(self):
        """Emergency cleanup when memory is critically low"""
        logger.warning("Performing emergency memory cleanup")
        
        # Unload all non-priority models
        loaded_models = self.get_loaded_models()
        priority_names = [m["name"] for m in self.models_config["priority"]]
        
        for model in loaded_models:
            model_name = model.get("name", "")
            if model_name and model_name not in priority_names:
                logger.warning(f"Emergency unload: {model_name}")
                self.unload_model(model_name)

    def monitor_and_optimize(self):
        """Main monitoring and optimization routine"""
        logger.info("Running model monitoring and optimization")
        
        # Check Ollama status
        status = self.get_ollama_status()
        if status["status"] != "running":
            logger.error(f"Ollama not running: {status.get('message', 'Unknown error')}")
            return
        
        # Check system memory
        available_memory = self.get_available_memory()
        memory_percent = psutil.virtual_memory().percent
        
        logger.info(f"Available memory: {available_memory}MB ({100-memory_percent:.1f}% free)")
        
        # Emergency cleanup if memory is critically low
        if memory_percent > 95:
            self.emergency_cleanup()
            return
        
        # Regular optimization
        if memory_percent > 85:
            self.cleanup_models()
        
        # Apply memory strategy
        self.apply_memory_strategy()
        
        # Log current state
        loaded_models = self.get_loaded_models()
        logger.info(f"Currently loaded models: {[m.get('name') for m in loaded_models]}")

    def generate_report(self) -> Dict:
        """Generate model management report"""
        available_memory = self.get_available_memory()
        memory_info = psutil.virtual_memory()
        loaded_models = self.get_loaded_models()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_memory": {
                "total_mb": memory_info.total // (1024 * 1024),
                "available_mb": available_memory,
                "used_percent": memory_info.percent
            },
            "ollama_status": self.get_ollama_status(),
            "loaded_models": loaded_models,
            "model_strategy": self.calculate_memory_strategy(),
            "recommendations": self.get_model_recommendations(),
            "usage_stats": self.model_usage_stats
        }
        
        return report

    def run(self):
        """Main run loop"""
        logger.info("SutazAI Model Manager started")
        
        # Schedule regular tasks
        schedule.every(5).minutes.do(self.monitor_and_optimize)
        schedule.every(30).minutes.do(self.cleanup_models)
        
        # Initial optimization
        self.monitor_and_optimize()
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)
                
                # Generate and save report every hour
                if datetime.now().minute == 0:
                    report = self.generate_report()
                    with open('/var/log/sutazai-model-report.json', 'w') as f:
                        json.dump(report, f, indent=2)
                
            except KeyboardInterrupt:
                logger.info("Model manager stopped")
                break
            except Exception as e:
                logger.error(f"Model manager error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    manager = ModelManager()
    manager.run()