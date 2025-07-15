"""Model Registry for Local Models"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    LANGUAGE_MODEL = "language_model"
    CODE_MODEL = "code_model"
    EMBEDDING_MODEL = "embedding_model"
    CHAT_MODEL = "chat_model"

@dataclass
class ModelConfig:
    model_id: str
    name: str
    model_type: ModelType
    description: str
    size: str
    capabilities: List[str]
    status: str = "available"
    local_path: Optional[str] = None

class ModelRegistry:
    def __init__(self):
        self.models = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default models"""
        default_models = [
            ModelConfig(
                model_id="llama2-7b",
                name="Llama 2 7B",
                model_type=ModelType.LANGUAGE_MODEL,
                description="Meta's Llama 2 7B parameter model",
                size="3.8GB",
                capabilities=["text_generation", "conversation"]
            ),
            ModelConfig(
                model_id="codellama-7b",
                name="Code Llama 7B",
                model_type=ModelType.CODE_MODEL,
                description="Code-specialized Llama model",
                size="3.8GB",
                capabilities=["code_generation", "code_completion"]
            )
        ]
        
        for model in default_models:
            self.models[model.model_id] = model
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[ModelConfig]:
        """List all models"""
        return list(self.models.values())
    
    def add_model(self, model: ModelConfig):
        """Add new model"""
        self.models[model.model_id] = model

# Global registry
model_registry = ModelRegistry()
