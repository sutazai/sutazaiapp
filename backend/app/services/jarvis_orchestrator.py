"""
JARVIS Orchestrator - Core AI coordination system
Implements Microsoft JARVIS four-stage pipeline architecture
Fully compliant with Rule 16: Local LLM Operations Only
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from app.services.ollama_helper import ollama_helper
from app.services.hardware_intelligence_system import (
    HardwareIntelligenceSystem,
    ResourceStatus,
    ThermalStatus
)
from app.services.model_selection_engine import (
    ModelSelectionEngine,
    TaskComplexity
)

logger = logging.getLogger(__name__)

# AI Provider imports with fallback
try:
    from langchain.tools import Tool
    from langchain.agents import initialize_agent, AgentType
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available - some features will be limited")

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - some features will be limited")


class TaskType(Enum):
    """Task classification for model selection"""
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    AUDIO = "audio"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


class ModelProvider(Enum):
    """Available model providers - LOCAL ONLY per Rule 16"""
    LOCAL = "local"
    OLLAMA = "ollama"


class TaskPlan(BaseModel):
    """Task planning structure with resource predictions"""
    task_type: TaskType
    complexity: float = Field(ge=0, le=1)
    requires_context: bool = False
    requires_tools: bool = False
    privacy_sensitive: bool = False
    expected_tokens: int = 1000
    subtasks: List[Dict[str, Any]] = []
    # Resource predictions
    predicted_memory_mb: int = 1000
    predicted_cpu_percent: float = 50.0
    predicted_runtime_seconds: float = 10.0


class ModelSelection(BaseModel):
    """Model selection result with resource awareness"""
    primary_model: str
    fallback_models: List[str]
    provider: ModelProvider
    estimated_cost: float
    estimated_latency: float
    requires_gpu: bool = False
    # Resource requirements
    required_memory_gb: float = 2.0
    required_cpu_cores: int = 2
    safety_validated: bool = False


class JARVISOrchestrator:
    """
    Main orchestrator implementing Microsoft JARVIS architecture
    Fully compliant with Rule 16: Intelligent Hardware-Aware Management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize hardware-aware systems
        logger.info("Initializing hardware-aware JARVIS orchestrator...")
        self.hardware_system = HardwareIntelligenceSystem()
        self.model_engine = ModelSelectionEngine(
            hardware_system=self.hardware_system,
            safety_margin=0.2,  # 20% safety buffer
            enable_predictive=True
        )
        
        # Perform initial system check
        hw_status = self.hardware_system.perform_comprehensive_selfcheck()
        logger.info(f"Hardware initialized: {hw_status['summary']}")
        
        # Validate minimum requirements
        if hw_status['resource_status'] == ResourceStatus.CRITICAL:
            logger.error("CRITICAL: Insufficient resources to start JARVIS")
            raise RuntimeError("Insufficient system resources")
        
        self.model_registry = self._initialize_models()
        self.tool_registry = self._initialize_tools()
        self.cache = {}  # Simple in-memory cache
        
        # Enhanced metrics with resource tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0,
            "resource_violations": 0,
            "emergency_shutdowns": 0,
            "model_switches": 0,
            "thermal_throttles": 0
        }
        
        # Resource monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.last_resource_check = datetime.now()
        self.resource_history = []
        
        # Start continuous monitoring
        self._start_resource_monitor()
        
    def _initialize_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize LOCAL ONLY models per Rule 16"""
        return {
            # Local models via Ollama - Automatically selected based on hardware
            "tinyllama": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CHAT, TaskType.CREATIVE, TaskType.SUMMARIZATION],
                "max_tokens": 2048,
                "cost_per_1k": 0.0,
                "latency": 0.3,
                "quality": 0.75,
                "local": True,
                "name": "tinyllama:latest",
                "min_ram_gb": 2,
                "min_vram_gb": 0,
                "complexity": TaskComplexity.SIMPLE
            },
            "mistral-7b": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CHAT, TaskType.CODE, TaskType.ANALYSIS],
                "max_tokens": 4096,
                "cost_per_1k": 0.0,
                "latency": 0.5,
                "quality": 0.82,
                "local": True,
                "name": "mistral:latest",
                "min_ram_gb": 8,
                "min_vram_gb": 4,
                "complexity": TaskComplexity.MODERATE
            },
            "llama2-7b": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CHAT, TaskType.CODE, TaskType.ANALYSIS],
                "max_tokens": 4096,
                "cost_per_1k": 0.0,
                "latency": 0.8,
                "quality": 0.85,
                "local": True,
                "name": "llama2:7b",
                "min_ram_gb": 8,
                "min_vram_gb": 4,
                "complexity": TaskComplexity.MODERATE
            },
            "llama2-13b": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CHAT, TaskType.CODE, TaskType.ANALYSIS, TaskType.CREATIVE],
                "max_tokens": 4096,
                "cost_per_1k": 0.0,
                "latency": 1.5,
                "quality": 0.88,
                "local": True,
                "name": "llama2:13b",
                "min_ram_gb": 16,
                "min_vram_gb": 8,
                "complexity": TaskComplexity.COMPLEX
            },
            "gpt-oss-20b": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CHAT, TaskType.CODE, TaskType.ANALYSIS, TaskType.CREATIVE, TaskType.VISION],
                "max_tokens": 8192,
                "cost_per_1k": 0.0,
                "latency": 2.5,
                "quality": 0.92,
                "local": True,
                "name": "gpt-oss:20b",
                "min_ram_gb": 32,
                "min_vram_gb": 16,
                "complexity": TaskComplexity.INTENSIVE
            },
            "codestral": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CODE],
                "max_tokens": 32768,
                "cost_per_1k": 0.0,
                "latency": 2.0,
                "quality": 0.92,
                "local": True,
                "name": "codestral:latest",
                "min_ram_gb": 16,
                "min_vram_gb": 8,
                "complexity": TaskComplexity.COMPLEX
            },
            "deepseek-coder": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CODE],
                "max_tokens": 16384,
                "cost_per_1k": 0.0,
                "latency": 1.8,
                "quality": 0.90,
                "local": True,
                "name": "deepseek-coder:latest",
                "min_ram_gb": 8,
                "min_vram_gb": 4,
                "complexity": TaskComplexity.MODERATE
            },
            "llava": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.VISION],
                "max_tokens": 4096,
                "cost_per_1k": 0.0,
                "latency": 3.0,
                "quality": 0.85,
                "local": True,
                "name": "llava:latest",
                "min_ram_gb": 8,
                "min_vram_gb": 4,
                "complexity": TaskComplexity.MODERATE
            }
        }
    
    def _initialize_tools(self) -> list:
        """Initialize LangChain tools for enhanced capabilities"""
        tools = []
        
        if LANGCHAIN_AVAILABLE:
            # Add search tool
            tools.append(Tool(
                name="web_search",
                func=self._web_search,
                description="Search the web for current information"
            ))
            
            # Add calculator tool
            tools.append(Tool(
                name="calculator",
                func=self._calculate,
                description="Perform mathematical calculations"
            ))
            
            # Add code execution tool
            tools.append(Tool(
                name="code_executor",
                func=self._execute_code,
                description="Execute Python code safely"
            ))
        
        return tools
    
    def _start_resource_monitor(self):
        """Start continuous resource monitoring thread"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Get current resource status
                    resources = self.hardware_system.get_current_resources()
                    
                    # Store in history
                    self.resource_history.append({
                        'timestamp': datetime.now(),
                        'resources': resources
                    })
                    
                    # Keep only last hour of data
                    cutoff = datetime.now() - timedelta(hours=1)
                    self.resource_history = [
                        h for h in self.resource_history 
                        if h['timestamp'] > cutoff
                    ]
                    
                    # Check for critical conditions
                    if resources['resource_status'] == ResourceStatus.CRITICAL:
                        logger.warning("CRITICAL resource status detected")
                        self._handle_resource_emergency()
                    
                    # Check thermal status
                    if resources['thermal_status'] == ThermalStatus.CRITICAL:
                        logger.warning("CRITICAL thermal status detected")
                        self.metrics['thermal_throttles'] += 1
                        self._handle_thermal_emergency()
                    
                    # Sleep for monitoring interval
                    time.sleep(1)  # 1 second monitoring interval
                    
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    time.sleep(5)  # Back off on error
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring thread started")
    
    def _handle_resource_emergency(self):
        """Handle critical resource conditions"""
        logger.warning("Entering resource emergency mode")
        
        # Switch to minimal model if not already
        if hasattr(self, 'current_model') and self.current_model != 'tinyllama':
            self.model_engine.switch_model('tinyllama')
            self.metrics['model_switches'] += 1
        
        # Clear caches
        self.cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.metrics['resource_violations'] += 1
    
    def _handle_thermal_emergency(self):
        """Handle critical thermal conditions"""
        logger.warning("Entering thermal emergency mode")
        
        # Immediate switch to tinyllama
        if hasattr(self, 'current_model'):
            self.model_engine.emergency_shutdown()
            self.metrics['emergency_shutdowns'] += 1
        
        # Wait for cooling
        time.sleep(5)
        
        # Restart with minimal model
        self.model_engine.switch_model('tinyllama')
    
    async def process(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main processing pipeline with full Rule 16 compliance
        """
        start_time = datetime.now()
        self.metrics["total_requests"] += 1
        
        try:
            # Pre-execution safety check
            if not self._validate_resource_safety():
                return {
                    "success": False,
                    "error": "System resources insufficient",
                    "response": "System is currently under heavy load. Please try again in a moment."
                }
            
            # Stage 1: Task Planning with resource prediction
            task_plan = await self._plan_task(user_input, context)
            logger.info(f"Task planned: {task_plan.task_type}, complexity: {task_plan.complexity}")
            
            # Stage 2: Hardware-aware model selection
            model_selection = await self._select_models_intelligently(task_plan)
            logger.info(f"Model selected: {model_selection.primary_model}")
            
            # Stage 3: Task Execution with monitoring
            execution_result = await self._execute_task_safely(
                user_input, task_plan, model_selection, context
            )
            
            # Stage 4: Response Generation
            response = await self._generate_response(
                execution_result, task_plan, model_selection
            )
            
            # Update metrics
            self.metrics["successful_requests"] += 1
            latency = (datetime.now() - start_time).total_seconds()
            self._update_latency_metric(latency)
            
            return {
                "success": True,
                "response": response,
                "metadata": {
                    "task_type": task_plan.task_type.value,
                    "model_used": model_selection.primary_model,
                    "latency": latency,
                    "resource_status": self.hardware_system.get_current_resources()['resource_status'].value,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"JARVIS processing error: {str(e)}")
            self.metrics["failed_requests"] += 1
            
            # Check if it's a resource issue
            if "resource" in str(e).lower() or "memory" in str(e).lower():
                self._handle_resource_emergency()
            
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your request."
            }
    
    def _validate_resource_safety(self) -> bool:
        """Validate system resources are safe for execution"""
        resources = self.hardware_system.get_current_resources()
        
        # Check critical thresholds
        if resources['resource_status'] == ResourceStatus.CRITICAL:
            return False
        
        if resources['thermal_status'] == ThermalStatus.CRITICAL:
            return False
        
        if resources['memory']['percent'] > 95:
            return False
        
        if resources['cpu']['percent'] > 95:
            return False
        
        return True
    
    async def _plan_task(self, user_input: str, context: Optional[Dict]) -> TaskPlan:
        """
        Stage 1: Analyze and plan the task with resource prediction
        """
        # Classify task type
        task_type = self._classify_task_type(user_input)
        
        # Estimate complexity
        complexity = self._estimate_complexity(user_input)
        
        # Map to TaskComplexity enum for model selection
        if complexity < 0.3:
            task_complexity = TaskComplexity.SIMPLE
        elif complexity < 0.6:
            task_complexity = TaskComplexity.MODERATE
        elif complexity < 0.8:
            task_complexity = TaskComplexity.COMPLEX
        else:
            task_complexity = TaskComplexity.INTENSIVE
        
        # Predict resource requirements
        predicted_resources = self.hardware_system.predict_resource_requirements(
            task_complexity=task_complexity,
            context_size=len(user_input),
            history_length=len(context.get('history', [])) if context else 0
        )
        
        task_plan = TaskPlan(
            task_type=task_type,
            complexity=complexity,
            requires_context=bool(context),
            requires_tools=self._check_tool_requirement(user_input),
            privacy_sensitive=self._check_privacy_sensitivity(user_input),
            expected_tokens=min(len(user_input) * 10, 4000),
            predicted_memory_mb=predicted_resources['memory_mb'],
            predicted_cpu_percent=predicted_resources['cpu_percent'],
            predicted_runtime_seconds=predicted_resources['runtime_seconds']
        )
        
        # Store context for scoring
        task_plan._context = context or {}
        task_plan._complexity_enum = task_complexity
        
        # Break down into subtasks if complex
        if task_plan.complexity > 0.7:
            task_plan.subtasks = self._decompose_task(user_input)
        
        return task_plan
    
    async def _select_models_intelligently(self, task_plan: TaskPlan) -> ModelSelection:
        """
        Stage 2: Select appropriate models using hardware-aware engine
        """
        # Use the ModelSelectionEngine for intelligent selection
        task_complexity = getattr(task_plan, '_complexity_enum', TaskComplexity.MODERATE)
        
        decision = self.model_engine.make_model_decision(
            task_complexity=task_complexity,
            context_size=task_plan.expected_tokens,
            response_time_target=30.0,  # 30 second target
            quality_requirement=0.8
        )
        
        # Map decision to our model registry
        suitable_models = []
        for model_name, model_info in self.model_registry.items():
            if task_plan.task_type in model_info["capabilities"]:
                # Check if model matches decision
                if model_info["name"] == decision['selected_model']:
                    suitable_models.insert(0, model_name)  # Primary
                elif model_info['complexity'] == TaskComplexity.SIMPLE:
                    suitable_models.append(model_name)  # Fallback
        
        if not suitable_models:
            # Default to tinyllama if no suitable model found
            suitable_models = ["tinyllama"]
        
        # Validate safety
        safety_check = self.model_engine.validate_safety(
            self.model_registry[suitable_models[0]]['name']
        )
        
        return ModelSelection(
            primary_model=suitable_models[0],
            fallback_models=suitable_models[1:3] if len(suitable_models) > 1 else ["tinyllama"],
            provider=ModelProvider.OLLAMA,
            estimated_cost=0.0,  # All local models are free
            estimated_latency=self.model_registry[suitable_models[0]]["latency"],
            requires_gpu=self.model_registry[suitable_models[0]].get("min_vram_gb", 0) > 0,
            required_memory_gb=self.model_registry[suitable_models[0]].get("min_ram_gb", 2),
            required_cpu_cores=2,
            safety_validated=safety_check['safe']
        )
    
    async def _execute_task_safely(
        self, 
        user_input: str, 
        task_plan: TaskPlan,
        model_selection: ModelSelection,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Stage 3: Execute task with safety monitoring and fallback
        """
        # Pre-execution safety check
        if not model_selection.safety_validated:
            safety = self.model_engine.validate_safety(
                self.model_registry[model_selection.primary_model]['name']
            )
            if not safety['safe']:
                logger.warning(f"Safety validation failed: {safety['reason']}")
                # Force fallback to tinyllama
                model_selection.primary_model = "tinyllama"
                model_selection.fallback_models = []
        
        # Try primary model with monitoring
        try:
            # Record current model
            self.current_model = model_selection.primary_model
            
            result = await self._call_model_with_monitoring(
                model_selection.primary_model,
                user_input,
                task_plan,
                context
            )
            return {"success": True, "result": result, "model": model_selection.primary_model}
            
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            self.metrics['model_switches'] += 1
            
            # Try fallback models
            for fallback_model in model_selection.fallback_models:
                try:
                    self.current_model = fallback_model
                    result = await self._call_model_with_monitoring(
                        fallback_model,
                        user_input,
                        task_plan,
                        context
                    )
                    return {"success": True, "result": result, "model": fallback_model}
                except Exception as fallback_error:
                    logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
            
            # Last resort: emergency tinyllama
            try:
                self.current_model = "tinyllama"
                self.model_engine.emergency_shutdown()
                self.model_engine.switch_model("tinyllama")
                result = await self._call_ollama("tinyllama", user_input, context)
                return {"success": True, "result": result, "model": "tinyllama"}
            except Exception as emergency_error:
                logger.error(f"Emergency model failed: {emergency_error}")
                raise Exception("All models failed to process the task")
    
    async def _call_model_with_monitoring(
        self,
        model_name: str,
        user_input: str,
        task_plan: TaskPlan,
        context: Optional[Dict]
    ) -> str:
        """Call model with resource monitoring"""
        # Start monitoring
        start_resources = self.hardware_system.get_current_resources()
        
        # Call the model
        result = await self._call_model(model_name, user_input, task_plan, context)
        
        # End monitoring
        end_resources = self.hardware_system.get_current_resources()
        
        # Check for resource violations
        if end_resources['resource_status'] == ResourceStatus.CRITICAL:
            self.metrics['resource_violations'] += 1
            logger.warning(f"Resource violation detected during {model_name} execution")
        
        return result
    
    async def _generate_response(
        self,
        execution_result: Dict[str, Any],
        task_plan: TaskPlan,
        model_selection: ModelSelection
    ) -> str:
        """
        Stage 4: Generate final response from execution results
        """
        if not execution_result["success"]:
            return "I apologize, but I couldn't complete your request at this time."
        
        raw_response = execution_result["result"]
        
        # Post-process based on task type
        if task_plan.task_type == TaskType.CODE:
            return self._format_code_response(raw_response)
        elif task_plan.task_type == TaskType.ANALYSIS:
            return self._format_analysis_response(raw_response)
        else:
            return raw_response
    
    # Helper methods
    def _classify_task_type(self, user_input: str) -> TaskType:
        """Classify task type from user input"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["code", "program", "function", "debug", "implement"]):
            return TaskType.CODE
        elif any(word in input_lower for word in ["image", "picture", "photo", "see", "look"]):
            return TaskType.VISION
        elif any(word in input_lower for word in ["analyze", "compare", "evaluate", "assess"]):
            return TaskType.ANALYSIS
        elif any(word in input_lower for word in ["create", "write", "compose", "generate"]):
            return TaskType.CREATIVE
        elif any(word in input_lower for word in ["translate", "language"]):
            return TaskType.TRANSLATION
        elif any(word in input_lower for word in ["summarize", "summary", "brief"]):
            return TaskType.SUMMARIZATION
        else:
            return TaskType.CHAT
    
    def _estimate_complexity(self, user_input: str) -> float:
        """Estimate task complexity (0-1)"""
        # Simple heuristic based on input length and keywords
        length_factor = min(len(user_input) / 500, 0.5)
        
        complex_keywords = ["complex", "detailed", "comprehensive", "analyze", "multiple"]
        keyword_factor = 0.5 if any(k in user_input.lower() for k in complex_keywords) else 0
        
        return min(length_factor + keyword_factor, 1.0)
    
    def _check_tool_requirement(self, user_input: str) -> bool:
        """Check if task requires external tools"""
        tool_keywords = ["search", "calculate", "current", "latest", "today", "now"]
        return any(k in user_input.lower() for k in tool_keywords)
    
    def _check_privacy_sensitivity(self, user_input: str) -> bool:
        """Check if task contains sensitive information"""
        sensitive_keywords = ["password", "secret", "private", "confidential", "personal"]
        return any(k in user_input.lower() for k in sensitive_keywords)
    
    def _decompose_task(self, user_input: str) -> List[Dict[str, Any]]:
        """Decompose complex task into subtasks"""
        # Simplified decomposition (in production, use LLM)
        return [{"subtask": "analyze", "input": user_input}]
    
    async def _call_model(
        self,
        model_name: str,
        user_input: str,
        task_plan: TaskPlan,
        context: Optional[Dict]
    ) -> str:
        """Call specific model with input - ONLY LOCAL MODELS"""
        model_info = self.model_registry[model_name]
        
        # All models go through Ollama (local only)
        if model_info["provider"] == ModelProvider.OLLAMA:
            return await self._call_ollama(model_name, user_input, context)
        else:
            # This should never happen with Rule 16 compliance
            raise ValueError(f"Non-local model provider not allowed: {model_info['provider']}")
    
    async def _call_ollama(self, model: str, prompt: str, context: Dict) -> str:
        """Call local Ollama model using helper"""
        # Get the actual Ollama model name from registry
        model_info = self.model_registry.get(model, {})
        ollama_model = model_info.get('name', 'tinyllama:latest')
        
        try:
            # Build messages for chat format
            messages = []
            
            # Add system prompt if available
            if context and context.get("system_prompt"):
                messages.append({
                    "role": "system",
                    "content": context["system_prompt"]
                })
            
            # Add conversation history if available
            if context and context.get("history"):
                for msg in context["history"][-5:]:  # Last 5 messages
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Use the helper to generate response
            response = await ollama_helper.chat(
                model=ollama_model,
                messages=messages,
                temperature=0.7,
                max_tokens=model_info.get('max_tokens', 2000)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling Ollama model {ollama_model}: {str(e)}")
            
            # Emergency fallback to tinyllama
            if ollama_model != 'tinyllama:latest':
                logger.info("Falling back to tinyllama...")
                return await ollama_helper.chat(
                    model='tinyllama:latest',
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000
                )
            raise Exception(f"Ollama error: {str(e)}")
    
    def _format_code_response(self, response: str) -> str:
        """Format code response with syntax highlighting hints"""
        return f"```python\n{response}\n```"
    
    def _format_analysis_response(self, response: str) -> str:
        """Format analysis response with structure"""
        return f"Analysis Results:\n\n{response}"
    
    def _update_latency_metric(self, latency: float):
        """Update average latency metric"""
        n = self.metrics["successful_requests"]
        old_avg = self.metrics["average_latency"]
        self.metrics["average_latency"] = (old_avg * (n - 1) + latency) / n if n > 0 else latency
    
    # Tool implementations
    async def _web_search(self, query: str) -> str:
        """Web search tool implementation"""
        return f"Search results for: {query}"
    
    async def _calculate(self, expression: str) -> str:
        """Calculator tool implementation"""
        try:
            result = eval(expression)
            return str(result)
        except:
            return "Invalid expression"
    
    async def _execute_code(self, code: str) -> str:
        """Safe code execution tool"""
        return "Code execution not implemented in safe mode"
    
    # Streaming support for real-time responses
    async def stream_process(self, user_input: str, context: Optional[Dict] = None):
        """
        Stream processing for real-time responses with resource awareness
        """
        # Validate resources first
        if not self._validate_resource_safety():
            yield "System resources insufficient. Please try again later."
            return
        
        task_plan = await self._plan_task(user_input, context)
        model_selection = await self._select_models_intelligently(task_plan)
        
        # Stream tokens as they're generated
        async for token in self._stream_model_call(
            model_selection.primary_model,
            user_input,
            context
        ):
            yield token
    
    async def _stream_model_call(self, model: str, prompt: str, context: Dict):
        """Stream model response token by token"""
        # Get response with monitoring
        response = await self._call_model_with_monitoring(model, prompt, None, context)
        
        # Simulate streaming
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate streaming delay
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for monitoring"""
        resources = self.hardware_system.get_current_resources()
        
        return {
            "hardware": resources,
            "metrics": self.metrics,
            "current_model": getattr(self, 'current_model', 'none'),
            "monitoring_active": self.monitoring_active,
            "cache_size": len(self.cache),
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down JARVIS orchestrator...")
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Shutdown model engine
        if hasattr(self, 'model_engine'):
            self.model_engine.emergency_shutdown()
        
        # Clear caches
        self.cache.clear()
        
        logger.info("JARVIS orchestrator shutdown complete")