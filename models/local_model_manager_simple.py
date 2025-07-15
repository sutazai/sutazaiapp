"""
Simplified Local Model Manager for validation
Basic model management without external dependencies
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class LocalModelManager:
    """Simplified local model manager for validation"""
    
    def __init__(self):
        self.models = {
            "llama3.1": {
                "name": "Llama 3.1",
                "type": "chat",
                "status": "available",
                "loaded": False
            },
            "codellama": {
                "name": "Code Llama",
                "type": "code",
                "status": "available",
                "loaded": False
            }
        }
        self.initialized = True
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models"""
        return [
            {
                "name": model_id,
                "type": info["type"],
                "status": info["status"],
                "loaded": info["loaded"]
            }
            for model_id, info in self.models.items()
        ]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "total_models": len(self.models),
            "loaded_models": sum(1 for m in self.models.values() if m["loaded"]),
            "ollama_status": {
                "status": "available",
                "host": "http://localhost:11434"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        if model_name in self.models:
            return {
                "name": model_name,
                "info": self.models[model_name],
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def load_model(self, model_name: str):
        """Load a model"""
        if model_name in self.models:
            self.models[model_name]["loaded"] = True
            return {"status": "loaded", "model": model_name}
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def unload_model(self, model_name: str):
        """Unload a model"""
        if model_name in self.models:
            self.models[model_name]["loaded"] = False
            return {"status": "unloaded", "model": model_name}
        else:
            raise ValueError(f"Model {model_name} not found")