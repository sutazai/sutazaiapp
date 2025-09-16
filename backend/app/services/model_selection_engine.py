"""
Model Selection Engine for Intelligent LLM Management
Implements automated decision-making for model selection based on hardware capabilities
Complies with Rule 16: Local LLM Operations
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import httpx
import aiohttp

from app.services.hardware_intelligence_system import (
    HardwareIntelligenceSystem, 
    ResourceStatus, 
    ResourcePrediction
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    INTENSIVE = "intensive"


class ModelType(Enum):
    """Available local models via Ollama"""
    TINYLLAMA = "tinyllama:latest"
    GPT_OSS_20B = "gpt-oss:20b"
    MISTRAL_7B = "mistral:7b-instruct"
    LLAMA2_7B = "llama2:7b"
    LLAMA2_13B = "llama2:13b"


@dataclass
class ModelProfile:
    """Model resource requirements and capabilities"""
    name: str
    model_type: ModelType
    memory_required_gb: float
    cpu_cores_optimal: int
    gpu_required: bool
    context_window: int
    quality_score: float  # 0-1 scale
    speed_score: float  # 0-1 scale
    capabilities: List[str] = field(default_factory=list)


@dataclass
class ModelDecision:
    """Model selection decision result"""
    selected_model: ModelType
    reason: str
    confidence: float
    resource_impact: str
    estimated_duration: float
    monitoring_required: bool
    auto_shutoff_time: Optional[float] = None
    fallback_model: Optional[ModelType] = None
    warnings: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)


class ModelSelectionEngine:
    """
    Intelligent model selection engine with hardware awareness
    Makes automated decisions based on multiple factors
    """
    
    def __init__(self, hardware_system: Optional[HardwareIntelligenceSystem] = None):
        self.hardware = hardware_system or HardwareIntelligenceSystem()
        self.model_profiles = self._initialize_model_profiles()
        self.decision_matrix = self._build_decision_matrix()
        self.safety_limits = self._establish_safety_limits()
        self.current_model = None
        self.model_switch_history = []
        self.performance_metrics = {}
        self.ollama_url = f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}"
        
    def _initialize_model_profiles(self) -> Dict[ModelType, ModelProfile]:
        """Initialize profiles for available models"""
        return {
            ModelType.TINYLLAMA: ModelProfile(
                name="TinyLlama 1.1B",
                model_type=ModelType.TINYLLAMA,
                memory_required_gb=1.5,
                cpu_cores_optimal=2,
                gpu_required=False,
                context_window=2048,
                quality_score=0.65,
                speed_score=0.95,
                capabilities=["chat", "basic_reasoning", "simple_tasks"]
            ),
            ModelType.MISTRAL_7B: ModelProfile(
                name="Mistral 7B Instruct",
                model_type=ModelType.MISTRAL_7B,
                memory_required_gb=6.0,
                cpu_cores_optimal=4,
                gpu_required=False,
                context_window=8192,
                quality_score=0.80,
                speed_score=0.75,
                capabilities=["chat", "reasoning", "code", "analysis"]
            ),
            ModelType.LLAMA2_7B: ModelProfile(
                name="Llama 2 7B",
                model_type=ModelType.LLAMA2_7B,
                memory_required_gb=7.0,
                cpu_cores_optimal=4,
                gpu_required=False,
                context_window=4096,
                quality_score=0.82,
                speed_score=0.70,
                capabilities=["chat", "reasoning", "creative", "analysis"]
            ),
            ModelType.LLAMA2_13B: ModelProfile(
                name="Llama 2 13B",
                model_type=ModelType.LLAMA2_13B,
                memory_required_gb=13.0,
                cpu_cores_optimal=6,
                gpu_required=True,
                context_window=4096,
                quality_score=0.88,
                speed_score=0.60,
                capabilities=["chat", "reasoning", "creative", "code", "analysis", "complex_tasks"]
            ),
            ModelType.GPT_OSS_20B: ModelProfile(
                name="GPT-OSS 20B",
                model_type=ModelType.GPT_OSS_20B,
                memory_required_gb=20.0,
                cpu_cores_optimal=8,
                gpu_required=True,
                context_window=8192,
                quality_score=0.92,
                speed_score=0.40,
                capabilities=["chat", "reasoning", "creative", "code", "analysis", "complex_tasks", "research"]
            )
        }
    
    def _build_decision_matrix(self) -> Dict[str, Any]:
        """Build decision matrix for model selection"""
        return {
            TaskComplexity.SIMPLE: {
                'preferred_models': [ModelType.TINYLLAMA],
                'max_memory_gb': 2,
                'max_cpu_usage': 30,
                'timeout_seconds': 30
            },
            TaskComplexity.MODERATE: {
                'preferred_models': [ModelType.MISTRAL_7B, ModelType.LLAMA2_7B, ModelType.TINYLLAMA],
                'max_memory_gb': 8,
                'max_cpu_usage': 50,
                'timeout_seconds': 60
            },
            TaskComplexity.COMPLEX: {
                'preferred_models': [ModelType.LLAMA2_13B, ModelType.LLAMA2_7B, ModelType.MISTRAL_7B],
                'max_memory_gb': 16,
                'max_cpu_usage': 70,
                'timeout_seconds': 120
            },
            TaskComplexity.INTENSIVE: {
                'preferred_models': [ModelType.GPT_OSS_20B, ModelType.LLAMA2_13B],
                'max_memory_gb': 32,
                'max_cpu_usage': 85,
                'timeout_seconds': 300
            }
        }
    
    def _establish_safety_limits(self) -> Dict[str, Any]:
        """Establish safety limits for resource usage"""
        return {
            'max_memory_percent': 85,
            'max_cpu_percent': 90,
            'max_gpu_percent': 95,
            'max_temperature_c': 85,
            'min_free_memory_gb': 2,
            'emergency_shutoff_temperature_c': 95,
            'max_runtime_seconds': {
                ModelType.TINYLLAMA: None,  # No limit for TinyLlama
                ModelType.MISTRAL_7B: 300,
                ModelType.LLAMA2_7B: 300,
                ModelType.LLAMA2_13B: 180,
                ModelType.GPT_OSS_20B: 120
            }
        }
    
    async def make_model_decision(
        self, 
        task_complexity: TaskComplexity,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelDecision:
        """
        Make intelligent model selection based on multiple factors
        """
        # Real-time system assessment
        current_resources = await self._get_current_resource_state()
        system_health = await self.hardware.assess_system_health()
        thermal_status = self.hardware.check_thermal_health()
        
        # Task analysis
        resource_prediction = await self._predict_resource_requirements(task_complexity, user_request)
        expected_duration = self._estimate_task_duration(task_complexity, len(user_request))
        
        # Safety validation
        safety_check = self._validate_safety_conditions(
            current_resources, 
            resource_prediction, 
            thermal_status
        )
        
        # Get available models from Ollama
        available_models = await self._get_available_ollama_models()
        
        # Make decision
        decision = self._make_final_decision(
            task_complexity,
            current_resources,
            safety_check,
            available_models,
            expected_duration
        )
        
        # Log decision
        self._log_decision(decision, task_complexity, current_resources)
        
        return decision
    
    async def _get_current_resource_state(self) -> Dict[str, Any]:
        """Get current system resource state"""
        # Refresh hardware profile
        self.hardware.hardware_profile = self.hardware.detect_hardware_capabilities()
        
        return {
            'cpu_usage': self.hardware.hardware_profile.cpu.get('current_utilization', 0),
            'memory_available_gb': self.hardware.hardware_profile.memory.get('available_gb', 0),
            'memory_used_percent': self.hardware.hardware_profile.memory.get('used_percent', 0),
            'gpu_available': self.hardware.hardware_profile.gpu.get('gpu_present', False),
            'gpu_usage': self.hardware.hardware_profile.gpu.get('gpu_utilization', 0) if self.hardware.hardware_profile.gpu.get('gpu_present') else 0,
            'temperature': self.hardware.hardware_profile.thermal.get('current_temperature', 0),
            'can_handle_intensive_operation': self._can_handle_intensive_operation()
        }
    
    def _can_handle_intensive_operation(self) -> bool:
        """Check if system can handle intensive operations"""
        profile = self.hardware.hardware_profile
        
        # Check minimum requirements
        if profile.memory.get('available_gb', 0) < self.safety_limits['min_free_memory_gb']:
            return False
        
        if profile.cpu.get('current_utilization', 100) > self.safety_limits['max_cpu_percent']:
            return False
        
        if profile.thermal.get('current_temperature', 100) > self.safety_limits['max_temperature_c']:
            return False
        
        if profile.thermal.get('thermal_throttling', False):
            return False
        
        return True
    
    async def _predict_resource_requirements(
        self, 
        task_complexity: TaskComplexity,
        user_request: str
    ) -> ResourcePrediction:
        """Predict resource requirements for the task"""
        # Use hardware system's prediction
        return await self.hardware.predict_resource_requirements(
            task_complexity.value,
            len(user_request)
        )
    
    def _estimate_task_duration(self, complexity: TaskComplexity, request_length: int) -> float:
        """Estimate task duration in seconds"""
        base_duration = {
            TaskComplexity.SIMPLE: 5,
            TaskComplexity.MODERATE: 15,
            TaskComplexity.COMPLEX: 30,
            TaskComplexity.INTENSIVE: 60
        }
        
        # Adjust for request length
        length_factor = min(request_length / 1000, 2)
        
        return base_duration.get(complexity, 15) * (1 + length_factor * 0.5)
    
    def _validate_safety_conditions(
        self,
        current_resources: Dict[str, Any],
        resource_prediction: ResourcePrediction,
        thermal_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate safety conditions for model operation"""
        safety_result = {
            'safe_for_tinyllama': True,
            'safe_for_7b': False,
            'safe_for_13b': False,
            'safe_for_gpt_oss': False,
            'confidence_score': 0.0,
            'limiting_factors': []
        }
        
        # Check for TinyLlama (always safe unless critical)
        if current_resources['cpu_usage'] < 95 and current_resources['memory_used_percent'] < 95:
            safety_result['safe_for_tinyllama'] = True
        
        # Check for 7B models
        if (current_resources['memory_available_gb'] > 8 and 
            current_resources['cpu_usage'] < 70 and
            thermal_status['temperature'] < 80):
            safety_result['safe_for_7b'] = True
        
        # Check for 13B models
        if (current_resources['memory_available_gb'] > 15 and
            current_resources['cpu_usage'] < 60 and
            thermal_status['temperature'] < 75 and
            (current_resources['gpu_available'] or current_resources['cpu_usage'] < 50)):
            safety_result['safe_for_13b'] = True
        
        # Check for GPT-OSS 20B
        if (current_resources['memory_available_gb'] > 24 and
            current_resources['cpu_usage'] < 50 and
            thermal_status['temperature'] < 70 and
            current_resources['gpu_available'] and
            current_resources['gpu_usage'] < 50 and
            not thermal_status.get('throttling', False)):
            safety_result['safe_for_gpt_oss'] = True
        
        # Calculate confidence score
        if safety_result['safe_for_gpt_oss']:
            safety_result['confidence_score'] = 0.95
        elif safety_result['safe_for_13b']:
            safety_result['confidence_score'] = 0.85
        elif safety_result['safe_for_7b']:
            safety_result['confidence_score'] = 0.75
        else:
            safety_result['confidence_score'] = 0.65
        
        # Identify limiting factors
        if current_resources['memory_available_gb'] < 8:
            safety_result['limiting_factors'].append("Low memory")
        if current_resources['cpu_usage'] > 70:
            safety_result['limiting_factors'].append("High CPU usage")
        if thermal_status['temperature'] > 80:
            safety_result['limiting_factors'].append("High temperature")
        if not current_resources['gpu_available']:
            safety_result['limiting_factors'].append("No GPU available")
        
        return safety_result
    
    async def _get_available_ollama_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        logger.info(f"Available Ollama models: {models}")
                        return models
                    else:
                        logger.warning(f"Failed to get Ollama models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            # Return default model that should always be available
            return ["tinyllama:latest"]
    
    def _make_final_decision(
        self,
        task_complexity: TaskComplexity,
        current_resources: Dict[str, Any],
        safety_check: Dict[str, Any],
        available_models: List[str],
        expected_duration: float
    ) -> ModelDecision:
        """Make final model selection decision"""
        
        # Get preferred models for task complexity
        matrix = self.decision_matrix[task_complexity]
        preferred_models = matrix['preferred_models']
        
        # Filter by safety and availability
        viable_models = []
        for model_type in preferred_models:
            model_name = model_type.value
            model_profile = self.model_profiles[model_type]
            
            # Check if model is available in Ollama
            if model_name not in available_models:
                continue
            
            # Check safety conditions
            if model_type == ModelType.TINYLLAMA and safety_check['safe_for_tinyllama']:
                viable_models.append(model_type)
            elif model_type in [ModelType.MISTRAL_7B, ModelType.LLAMA2_7B] and safety_check['safe_for_7b']:
                viable_models.append(model_type)
            elif model_type == ModelType.LLAMA2_13B and safety_check['safe_for_13b']:
                viable_models.append(model_type)
            elif model_type == ModelType.GPT_OSS_20B and safety_check['safe_for_gpt_oss']:
                viable_models.append(model_type)
        
        # Select best viable model or default to TinyLlama
        if viable_models:
            selected_model = viable_models[0]  # First viable is best for complexity
        else:
            selected_model = ModelType.TINYLLAMA  # Always fallback to TinyLlama
        
        # Determine resource impact
        model_profile = self.model_profiles[selected_model]
        if model_profile.memory_required_gb < 2:
            resource_impact = "low"
        elif model_profile.memory_required_gb < 8:
            resource_impact = "moderate"
        elif model_profile.memory_required_gb < 16:
            resource_impact = "high"
        else:
            resource_impact = "very_high"
        
        # Determine monitoring requirements
        monitoring_required = selected_model != ModelType.TINYLLAMA
        
        # Calculate auto-shutoff time for resource-intensive models
        auto_shutoff_time = self.safety_limits['max_runtime_seconds'].get(selected_model)
        
        # Build warnings
        warnings = []
        if safety_check['limiting_factors']:
            warnings.extend(safety_check['limiting_factors'])
        if current_resources['temperature'] > 75:
            warnings.append("System running warm - monitoring thermal status")
        if selected_model == ModelType.GPT_OSS_20B:
            warnings.append("Resource-intensive model - automatic shutoff enabled")
        
        # Build optimizations
        optimizations = []
        if current_resources['gpu_available'] and selected_model in [ModelType.LLAMA2_13B, ModelType.GPT_OSS_20B]:
            optimizations.append("GPU acceleration enabled")
        if selected_model == ModelType.TINYLLAMA:
            optimizations.append("Fast inference mode")
        
        # Determine reason
        if selected_model == ModelType.TINYLLAMA:
            if task_complexity == TaskComplexity.SIMPLE:
                reason = "Optimal model for simple task complexity"
            else:
                reason = f"Resource constraints require lightweight model (Limits: {', '.join(safety_check['limiting_factors'])})"
        elif selected_model in [ModelType.MISTRAL_7B, ModelType.LLAMA2_7B]:
            reason = f"Balanced model for {task_complexity.value} complexity with available resources"
        elif selected_model == ModelType.LLAMA2_13B:
            reason = "Enhanced model for complex task with sufficient resources"
        else:  # GPT_OSS_20B
            reason = "Maximum capability model selected for intensive task with optimal resources"
        
        # Determine fallback model
        fallback_model = ModelType.TINYLLAMA if selected_model != ModelType.TINYLLAMA else None
        
        return ModelDecision(
            selected_model=selected_model,
            reason=reason,
            confidence=safety_check['confidence_score'],
            resource_impact=resource_impact,
            estimated_duration=expected_duration,
            monitoring_required=monitoring_required,
            auto_shutoff_time=auto_shutoff_time,
            fallback_model=fallback_model,
            warnings=warnings,
            optimizations=optimizations
        )
    
    def _log_decision(
        self, 
        decision: ModelDecision, 
        task_complexity: TaskComplexity,
        resources: Dict[str, Any]
    ) -> None:
        """Log model selection decision"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'task_complexity': task_complexity.value,
            'selected_model': decision.selected_model.value,
            'reason': decision.reason,
            'confidence': decision.confidence,
            'resource_state': resources,
            'warnings': decision.warnings
        }
        
        logger.info(f"Model decision: {decision.selected_model.value} - {decision.reason}")
        if decision.warnings:
            logger.warning(f"Decision warnings: {', '.join(decision.warnings)}")
        
        # Add to history
        self.model_switch_history.append(log_entry)
    
    async def switch_model(self, target_model: ModelType) -> bool:
        """Switch to a different model"""
        try:
            # Check if model is available
            available = await self._get_available_ollama_models()
            if target_model.value not in available:
                logger.error(f"Model {target_model.value} not available in Ollama")
                return False
            
            # Unload current model if exists
            if self.current_model:
                await self._unload_model(self.current_model)
            
            # Load new model
            success = await self._load_model(target_model)
            
            if success:
                self.current_model = target_model
                logger.info(f"Successfully switched to model: {target_model.value}")
                return True
            else:
                logger.error(f"Failed to switch to model: {target_model.value}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    async def _load_model(self, model_type: ModelType) -> bool:
        """Load a model in Ollama"""
        try:
            # Pull model if not available
            async with aiohttp.ClientSession() as session:
                # First try to pull the model
                pull_data = {"name": model_type.value}
                async with session.post(f"{self.ollama_url}/api/pull", json=pull_data) as response:
                    if response.status == 200:
                        logger.info(f"Model {model_type.value} pulled successfully")
                    # Even if pull fails, model might already exist
                
                # Load the model
                load_data = {"model": model_type.value}
                async with session.post(f"{self.ollama_url}/api/generate", 
                                       json={**load_data, "prompt": "test", "stream": False}) as response:
                    if response.status == 200:
                        logger.info(f"Model {model_type.value} loaded successfully")
                        return True
                    else:
                        logger.error(f"Failed to load model {model_type.value}: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error loading model {model_type.value}: {e}")
            return False
    
    async def _unload_model(self, model_type: ModelType) -> bool:
        """Unload a model from memory"""
        # Ollama doesn't have explicit unload, but we can track it
        logger.info(f"Model {model_type.value} marked for unload")
        return True
    
    async def emergency_shutdown(self, reason: str) -> None:
        """Emergency shutdown of current model"""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        # Force switch to TinyLlama
        if self.current_model != ModelType.TINYLLAMA:
            await self.switch_model(ModelType.TINYLLAMA)
        
        # Log emergency event
        emergency_log = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'previous_model': self.current_model.value if self.current_model else None,
            'action': 'Forced switch to TinyLlama'
        }
        
        logger.critical(f"Emergency action taken: {emergency_log}")
    
    def analyze_task_complexity(self, user_request: str) -> TaskComplexity:
        """Analyze and classify task complexity from user request"""
        request_lower = user_request.lower()
        request_length = len(user_request)
        
        # Keywords for different complexity levels
        simple_keywords = ["hello", "hi", "thanks", "simple", "basic", "what is", "tell me"]
        moderate_keywords = ["explain", "describe", "how to", "compare", "list", "summarize"]
        complex_keywords = ["analyze", "design", "implement", "optimize", "debug", "architect"]
        intensive_keywords = ["research", "comprehensive", "detailed analysis", "full implementation", 
                             "enterprise", "production", "scale", "benchmark"]
        
        # Check for keywords
        has_simple = any(keyword in request_lower for keyword in simple_keywords)
        has_moderate = any(keyword in request_lower for keyword in moderate_keywords)
        has_complex = any(keyword in request_lower for keyword in complex_keywords)
        has_intensive = any(keyword in request_lower for keyword in intensive_keywords)
        
        # Classify based on keywords and length
        if has_intensive or request_length > 2000:
            return TaskComplexity.INTENSIVE
        elif has_complex or request_length > 1000:
            return TaskComplexity.COMPLEX
        elif has_moderate or request_length > 500:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report of model selection decisions"""
        if not self.model_switch_history:
            return {"status": "No history available"}
        
        # Analyze history
        model_usage = {}
        complexity_distribution = {}
        
        for entry in self.model_switch_history:
            # Count model usage
            model = entry['selected_model']
            model_usage[model] = model_usage.get(model, 0) + 1
            
            # Count complexity distribution
            complexity = entry['task_complexity']
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
        
        return {
            'total_decisions': len(self.model_switch_history),
            'model_usage': model_usage,
            'complexity_distribution': complexity_distribution,
            'current_model': self.current_model.value if self.current_model else None,
            'last_decision': self.model_switch_history[-1] if self.model_switch_history else None
        }


# Create global instance
model_selection_engine = ModelSelectionEngine()