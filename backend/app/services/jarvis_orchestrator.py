"""
JARVIS Orchestrator - Core AI coordination system
Implements Microsoft JARVIS four-stage pipeline architecture
with best practices from all analyzed repositories
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import httpx
from pydantic import BaseModel, Field

# AI Provider imports
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)


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
    """Available model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    OLLAMA = "ollama"


class TaskPlan(BaseModel):
    """Task planning structure"""
    task_type: TaskType
    complexity: float = Field(ge=0, le=1)
    requires_context: bool = False
    requires_tools: bool = False
    privacy_sensitive: bool = False
    expected_tokens: int = 1000
    subtasks: List[Dict[str, Any]] = []


class ModelSelection(BaseModel):
    """Model selection result"""
    primary_model: str
    fallback_models: List[str]
    provider: ModelProvider
    estimated_cost: float
    estimated_latency: float
    requires_gpu: bool = False


class JARVISOrchestrator:
    """
    Main orchestrator implementing Microsoft JARVIS architecture
    with enhancements from other repositories
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_registry = self._initialize_models()
        self.tool_registry = self._initialize_tools()
        self.cache = {}  # Simple in-memory cache
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0
        }
        
    def _initialize_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available models with capabilities"""
        return {
            # Chat models
            "gpt-4": {
                "provider": ModelProvider.OPENAI,
                "capabilities": [TaskType.CHAT, TaskType.CODE, TaskType.ANALYSIS],
                "max_tokens": 8192,
                "cost_per_1k": 0.03,
                "latency": 1.5,
                "quality": 0.95
            },
            "claude-3-opus": {
                "provider": ModelProvider.ANTHROPIC,
                "capabilities": [TaskType.CHAT, TaskType.CODE, TaskType.CREATIVE],
                "max_tokens": 200000,
                "cost_per_1k": 0.015,
                "latency": 1.2,
                "quality": 0.95
            },
            "gemini-pro": {
                "provider": ModelProvider.GOOGLE,
                "capabilities": [TaskType.CHAT, TaskType.VISION, TaskType.ANALYSIS],
                "max_tokens": 32768,
                "cost_per_1k": 0.001,
                "latency": 0.8,
                "quality": 0.90
            },
            # Code models
            "codestral": {
                "provider": ModelProvider.HUGGINGFACE,
                "capabilities": [TaskType.CODE],
                "max_tokens": 32768,
                "cost_per_1k": 0.0,
                "latency": 2.0,
                "quality": 0.92
            },
            "deepseek-coder": {
                "provider": ModelProvider.HUGGINGFACE,
                "capabilities": [TaskType.CODE],
                "max_tokens": 16384,
                "cost_per_1k": 0.0,
                "latency": 1.8,
                "quality": 0.90
            },
            # Vision models
            "llava": {
                "provider": ModelProvider.HUGGINGFACE,
                "capabilities": [TaskType.VISION],
                "max_tokens": 4096,
                "cost_per_1k": 0.0,
                "latency": 3.0,
                "quality": 0.85
            },
            # Local models (via Ollama)
            "llama-3-70b": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CHAT, TaskType.CODE],
                "max_tokens": 8192,
                "cost_per_1k": 0.0,
                "latency": 2.5,
                "quality": 0.88,
                "local": True
            },
            "mistral-7b": {
                "provider": ModelProvider.OLLAMA,
                "capabilities": [TaskType.CHAT],
                "max_tokens": 4096,
                "cost_per_1k": 0.0,
                "latency": 0.5,
                "quality": 0.82,
                "local": True
            }
        }
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize LangChain tools for enhanced capabilities"""
        tools = []
        
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
    
    async def process(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main processing pipeline following Microsoft JARVIS architecture
        """
        start_time = datetime.now()
        self.metrics["total_requests"] += 1
        
        try:
            # Stage 1: Task Planning
            task_plan = await self._plan_task(user_input, context)
            logger.info(f"Task planned: {task_plan.task_type}, complexity: {task_plan.complexity}")
            
            # Stage 2: Model Selection
            model_selection = await self._select_models(task_plan)
            logger.info(f"Model selected: {model_selection.primary_model}")
            
            # Stage 3: Task Execution
            execution_result = await self._execute_task(
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
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"JARVIS processing error: {str(e)}")
            self.metrics["failed_requests"] += 1
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your request."
            }
    
    async def _plan_task(self, user_input: str, context: Optional[Dict]) -> TaskPlan:
        """
        Stage 1: Analyze and plan the task
        Inspired by Microsoft JARVIS task planning
        """
        # Use a lightweight model for task analysis
        prompt = f"""Analyze this user request and classify it:
        User Input: {user_input}
        Context: {json.dumps(context) if context else 'None'}
        
        Determine:
        1. Task type (chat/code/vision/audio/analysis/creative/translation/summarization)
        2. Complexity (0-1 scale)
        3. Whether it requires context
        4. Whether it requires tools
        5. Privacy sensitivity
        6. Estimated tokens needed
        
        Respond in JSON format.
        """
        
        # For now, use simple heuristics (in production, use an LLM)
        task_plan = TaskPlan(
            task_type=self._classify_task_type(user_input),
            complexity=self._estimate_complexity(user_input),
            requires_context=bool(context),
            requires_tools=self._check_tool_requirement(user_input),
            privacy_sensitive=self._check_privacy_sensitivity(user_input),
            expected_tokens=min(len(user_input) * 10, 4000)
        )
        
        # Break down into subtasks if complex
        if task_plan.complexity > 0.7:
            task_plan.subtasks = self._decompose_task(user_input)
        
        return task_plan
    
    async def _select_models(self, task_plan: TaskPlan) -> ModelSelection:
        """
        Stage 2: Select appropriate models based on task requirements
        Dynamic selection inspired by Microsoft JARVIS
        """
        suitable_models = []
        
        # Filter models by capability
        for model_name, model_info in self.model_registry.items():
            if task_plan.task_type in model_info["capabilities"]:
                # Score the model
                score = self._score_model(model_info, task_plan)
                suitable_models.append((model_name, score, model_info))
        
        # Sort by score
        suitable_models.sort(key=lambda x: x[1], reverse=True)
        
        if not suitable_models:
            raise ValueError(f"No suitable model found for task type: {task_plan.task_type}")
        
        # Select primary and fallback models
        primary = suitable_models[0]
        fallbacks = [m[0] for m in suitable_models[1:3]]
        
        return ModelSelection(
            primary_model=primary[0],
            fallback_models=fallbacks,
            provider=primary[2]["provider"],
            estimated_cost=primary[2]["cost_per_1k"] * task_plan.expected_tokens / 1000,
            estimated_latency=primary[2]["latency"],
            requires_gpu=primary[2].get("requires_gpu", False)
        )
    
    async def _execute_task(
        self, 
        user_input: str, 
        task_plan: TaskPlan,
        model_selection: ModelSelection,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Stage 3: Execute the task with selected models
        Implements retry logic and fallback mechanisms
        """
        # Try primary model
        try:
            result = await self._call_model(
                model_selection.primary_model,
                user_input,
                task_plan,
                context
            )
            return {"success": True, "result": result, "model": model_selection.primary_model}
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            
            # Try fallback models
            for fallback_model in model_selection.fallback_models:
                try:
                    result = await self._call_model(
                        fallback_model,
                        user_input,
                        task_plan,
                        context
                    )
                    return {"success": True, "result": result, "model": fallback_model}
                except Exception as fallback_error:
                    logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
            
            raise Exception("All models failed to process the task")
    
    async def _generate_response(
        self,
        execution_result: Dict[str, Any],
        task_plan: TaskPlan,
        model_selection: ModelSelection
    ) -> str:
        """
        Stage 4: Generate final response from execution results
        Formats and enhances the raw model output
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
    
    def _score_model(self, model_info: Dict, task_plan: TaskPlan) -> float:
        """Score model suitability for task"""
        score = model_info["quality"] * 100
        
        # Adjust for privacy requirements
        if task_plan.privacy_sensitive and model_info.get("local"):
            score += 50
        elif task_plan.privacy_sensitive and not model_info.get("local"):
            score -= 30
        
        # Adjust for complexity
        score += (1 - abs(task_plan.complexity - model_info["quality"])) * 20
        
        # Adjust for cost (if not local)
        if model_info["cost_per_1k"] > 0:
            score -= model_info["cost_per_1k"] * 10
        
        # Adjust for latency requirements
        score -= model_info["latency"] * 5
        
        return score
    
    async def _call_model(
        self,
        model_name: str,
        user_input: str,
        task_plan: TaskPlan,
        context: Optional[Dict]
    ) -> str:
        """Call specific model with input"""
        model_info = self.model_registry[model_name]
        
        # Route to appropriate provider
        if model_info["provider"] == ModelProvider.OPENAI:
            return await self._call_openai(model_name, user_input, context)
        elif model_info["provider"] == ModelProvider.ANTHROPIC:
            return await self._call_anthropic(model_name, user_input, context)
        elif model_info["provider"] == ModelProvider.OLLAMA:
            return await self._call_ollama(model_name, user_input, context)
        else:
            return f"Model {model_name} execution not implemented yet"
    
    async def _call_openai(self, model: str, prompt: str, context: Dict) -> str:
        """Call OpenAI model"""
        # Placeholder - implement actual API call
        return f"OpenAI {model} response to: {prompt}"
    
    async def _call_anthropic(self, model: str, prompt: str, context: Dict) -> str:
        """Call Anthropic model"""
        # Placeholder - implement actual API call
        return f"Anthropic {model} response to: {prompt}"
    
    async def _call_ollama(self, model: str, prompt: str, context: Dict) -> str:
        """Call local Ollama model"""
        # Placeholder - implement actual API call
        return f"Ollama {model} response to: {prompt}"
    
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
        Stream processing for real-time responses
        Inspired by danilofalcao's WebSocket implementation
        """
        task_plan = await self._plan_task(user_input, context)
        model_selection = await self._select_models(task_plan)
        
        # Stream tokens as they're generated
        async for token in self._stream_model_call(
            model_selection.primary_model,
            user_input,
            context
        ):
            yield token
    
    async def _stream_model_call(self, model: str, prompt: str, context: Dict):
        """Stream model response token by token"""
        # Placeholder for streaming implementation
        response = await self._call_model(model, prompt, None, context)
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate streaming delay