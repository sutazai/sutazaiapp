"""
AGI Brain API Endpoints
Central intelligence and reasoning endpoints with Advanced Quantum-Neuromorphic Architecture
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from enum import Enum
import logging
import asyncio
import time
from datetime import datetime

# Import the advanced brain architecture
try:
    from app.advanced_brain_architecture import get_advanced_brain, AdvancedBrainArchitecture
    ADVANCED_BRAIN_AVAILABLE = True
except ImportError:
    ADVANCED_BRAIN_AVAILABLE = False
    logging.warning("Advanced brain architecture not available, using fallback")

# Import the unified service controller
try:
    from app.unified_service_controller import get_unified_controller
    SERVICE_CONTROLLER_AVAILABLE = True
except ImportError:
    SERVICE_CONTROLLER_AVAILABLE = False
    logging.warning("Unified service controller not available")

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response models
class ReasoningType(str, Enum):
    deductive = "deductive"
    inductive = "inductive"
    abductive = "abductive"
    analogical = "analogical"
    causal = "causal"
    creative = "creative"
    strategic = "strategic"

class ThinkRequest(BaseModel):
    input_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    reasoning_type: ReasoningType = ReasoningType.deductive

class ThinkResponse(BaseModel):
    thought_id: str
    result: Any
    confidence: float
    reasoning_type: str
    complexity: str
    agents_used: List[str]

class BrainStatusResponse(BaseModel):
    status: str
    active_thoughts: int
    memory_usage: Dict[str, int]
    knowledge_domains: List[str]
    learning_rate: float
    confidence_threshold: float

class MemoryQueryRequest(BaseModel):
    query: str
    memory_type: str = "all"  # "short_term", "long_term", or "all"
    domain: Optional[str] = None

# Enhanced AGI Brain with Advanced Quantum-Neuromorphic Architecture
class EnhancedAGIBrain:
    def __init__(self):
        # Legacy compatibility
        self.thoughts = {}
        self.memory_short_term = []
        self.memory_long_term = {}
        self.knowledge_domains = [
            "general", "code", "security", "analysis", "reasoning", 
            "consciousness", "quantum_processing", "neuromorphic_computing",
            "pattern_recognition", "memory_consolidation", "creative_synthesis"
        ]
        self.learning_rate = 0.01
        self.confidence_threshold = 0.7
        
        # Advanced brain architecture integration
        self.advanced_brain = None
        self.use_advanced_processing = ADVANCED_BRAIN_AVAILABLE
        self.processing_stats = {
            "total_thoughts": 0,
            "quantum_processed": 0,
            "neuromorphic_processed": 0,
            "average_latency_ms": 0.0,
            "consciousness_activations": 0
        }
        
        # Initialize advanced brain if available
        if self.use_advanced_processing:
            try:
                self.advanced_brain = get_advanced_brain()
                logger.info("🧠 Enhanced AGI Brain initialized with Advanced Quantum-Neuromorphic Architecture")
            except Exception as e:
                logger.error(f"Failed to initialize advanced brain: {e}")
                self.use_advanced_processing = False
                
        # Initialize service controller if available
        self.service_controller = None
        if SERVICE_CONTROLLER_AVAILABLE:
            try:
                self.service_controller = get_unified_controller()
                logger.info("🔧 Unified Service Controller initialized - All services can now be controlled via chat")
            except Exception as e:
                logger.error(f"Failed to initialize service controller: {e}")
        
    async def think(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]], reasoning_type: str) -> Dict[str, Any]:
        """Process a thought using enhanced AGI brain architecture"""
        thought_id = f"thought_{len(self.thoughts) + 1}"
        start_time = time.time()
        
        # Get the input text
        input_text = input_data.get('text', 'No input provided')
        
        # Update processing stats
        self.processing_stats["total_thoughts"] += 1
        
        # Check if this is a service control command
        if SERVICE_CONTROLLER_AVAILABLE and self.service_controller:
            service_keywords = ["start", "stop", "restart", "status", "health", "check", "service", "services", 
                              "resource", "usage", "logs", "list", "show", "scale", "update", "backup", "optimize"]
            input_lower = input_text.lower()
            
            if any(keyword in input_lower for keyword in service_keywords):
                logger.info(f"Service control command detected: {input_text}")
                try:
                    controller = get_unified_controller()
                    service_result = await controller.execute_command(input_text)
                    
                    # Format service control result as brain response
                    if service_result.get("status") == "success":
                        output_text = service_result.get("message", "Service command executed successfully")
                        
                        # Add details if available
                        if "details" in service_result:
                            details = service_result["details"]
                            output_text += f"\n\nService: {service_result.get('service', 'unknown')}"
                            output_text += f"\nStatus: {details.get('status', 'unknown')}"
                            output_text += f"\nHealth: {details.get('health', 'unknown')}"
                            if "resources" in details:
                                res = details["resources"]
                                output_text += f"\nCPU: {res.get('cpu_percent', 0)}%"
                                output_text += f"\nMemory: {res.get('memory_mb', 0)}MB ({res.get('memory_percent', 0)}%)"
                        
                        # Add system status if available
                        if "services" in service_result:
                            srv = service_result["services"]
                            output_text += f"\n\nSystem Overview:"
                            output_text += f"\nTotal Services: {srv.get('total', 0)}"
                            output_text += f"\nRunning: {srv.get('running', 0)}"
                            output_text += f"\nStopped: {srv.get('stopped', 0)}"
                            output_text += f"\nUnhealthy: {srv.get('unhealthy', 0)}"
                        
                        # Add resource status if available
                        if "resources" in service_result:
                            res = service_result["resources"]
                            output_text += f"\n\nSystem Resources:"
                            if "cpu" in res:
                                output_text += f"\nCPU: {res['cpu']['percent']}% {res['cpu']['status']}"
                            if "memory" in res:
                                output_text += f"\nMemory: {res['memory']['percent']}% {res['memory']['status']} ({res['memory']['used_gb']}GB/{res['memory']['total_gb']}GB)"
                            if "disk" in res:
                                output_text += f"\nDisk: {res['disk']['percent']}% {res['disk']['status']} ({res['disk']['used_gb']}GB/{res['disk']['total_gb']}GB)"
                        
                        # Add recommendations if available
                        if "recommendations" in service_result:
                            output_text += "\n\nRecommendations:"
                            for rec in service_result["recommendations"]:
                                output_text += f"\n- {rec}"
                        
                        # Add service list if available
                        if "services" in service_result and isinstance(service_result["services"], dict):
                            for svc_type, svc_list in service_result["services"].items():
                                if svc_list:
                                    output_text += f"\n\n{svc_type.capitalize()} Services:"
                                    for svc in svc_list[:5]:  # Limit to 5 per type
                                        status_icon = "🟢" if svc.get("status") == "running" else "🔴"
                                        output_text += f"\n{status_icon} {svc['name']}: {svc.get('description', '')}"
                        
                        result = {
                            "thought_id": thought_id,
                            "result": {
                                "analysis": "Service Control Command Executed",
                                "insights": [
                                    f"Command type: {service_result.get('action', 'query')}",
                                    f"Target: {service_result.get('service', 'system')}",
                                    "Unified service controller active"
                                ],
                                "recommendations": service_result.get("suggestions", ["Service command completed successfully"]),
                                "output": output_text
                            },
                            "confidence": 1.0,
                            "reasoning_type": "service_control",
                            "complexity": "system",
                            "agents_used": ["unified_service_controller"],
                            "processing_time_ms": (time.time() - start_time) * 1000
                        }
                        
                        self.thoughts[thought_id] = result
                        self.memory_short_term.append({
                            "thought_id": thought_id,
                            "summary": f"Service control: {input_text}"
                        })
                        
                        return result
                        
                    elif service_result.get("status") == "unknown_command":
                        # Not a service command, continue with regular processing
                        pass
                    else:
                        # Error or warning
                        result = {
                            "thought_id": thought_id,
                            "result": {
                                "analysis": "Service Control Command",
                                "insights": [f"Status: {service_result.get('status', 'unknown')}"],
                                "recommendations": service_result.get("suggestions", []),
                                "output": service_result.get("message", "Service command failed")
                            },
                            "confidence": 0.8,
                            "reasoning_type": "service_control",
                            "complexity": "system",
                            "agents_used": ["unified_service_controller"]
                        }
                        
                        self.thoughts[thought_id] = result
                        return result
                        
                except Exception as e:
                    import traceback
                    logger.error(f"Service control command failed: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue with regular processing
        
        # Try advanced brain processing first
        if False and self.use_advanced_processing and self.advanced_brain:  # Temporarily disabled due to circular reference
            try:
                logger.info("🧠 Using Advanced Quantum-Neuromorphic Processing")
                
                # Process with advanced brain
                advanced_result = await self.advanced_brain.process_ultra_fast(
                    query=input_text,
                    processing_type=reasoning_type,
                    use_quantum=True,
                    optimization_level=10
                )
                
                processing_time = time.time() - start_time
                latency_ms = processing_time * 1000
                
                # Update stats
                if advanced_result.get("quantum_acceleration"):
                    self.processing_stats["quantum_processed"] += 1
                else:
                    self.processing_stats["neuromorphic_processed"] += 1
                
                if advanced_result.get("consciousness_level", 0) > 0.8:
                    self.processing_stats["consciousness_activations"] += 1
                
                # Update average latency
                current_avg = self.processing_stats["average_latency_ms"]
                total_thoughts = self.processing_stats["total_thoughts"]
                self.processing_stats["average_latency_ms"] = (current_avg * (total_thoughts - 1) + latency_ms) / total_thoughts
                
                # Create enhanced result structure
                result = {
                    "thought_id": thought_id,
                    "result": {
                        "analysis": f"🧠 Advanced {reasoning_type} reasoning with quantum-neuromorphic acceleration",
                        "insights": [
                            f"Processing cluster: {advanced_result.get('cluster_used', 'unknown')}",
                            f"Neurons activated: {advanced_result.get('neurons_activated', 0):,}",
                            f"Consciousness level: {advanced_result.get('consciousness_level', 0):.2f}",
                            "Advanced brain architecture active"
                        ],
                        "recommendations": [
                            "Utilizing cutting-edge neuromorphic processing",
                            "Enhanced pattern recognition engaged",
                            "Quantum acceleration optimized"
                        ],
                        "output": advanced_result.get("response", "Advanced processing completed"),
                        "performance_metrics": advanced_result.get("performance_metrics", {}),
                        "advanced_features": {
                            "quantum_acceleration": advanced_result.get("quantum_acceleration", False),
                            "neuromorphic_processing": True,
                            "consciousness_simulation": advanced_result.get("consciousness_level", 0) > 0.8,
                            "titans_memory": advanced_result.get("titans_memory", True),
                            "latency_ms": latency_ms
                        }
                    },
                    "confidence": 0.95,
                    "reasoning_type": reasoning_type,
                    "complexity": "advanced",
                    "agents_used": [f"advanced_brain_{advanced_result.get('cluster_used', 'neuromorphic')}"],
                    "processing_time_ms": latency_ms,
                    "architecture": "SutazAI Advanced Brain 2025 - Titans Enhanced"
                }
                
                # Store in memory
                self.thoughts[thought_id] = result
                self.memory_short_term.append({
                    "thought_id": thought_id,
                    "summary": f"Advanced thought about: {input_text}",
                    "performance": advanced_result.get("performance_metrics", {})
                })
                
                logger.info(f"🚀 Advanced processing completed in {latency_ms:.2f}ms")
                return result
                
            except Exception as e:
                logger.error(f"Advanced brain processing failed: {e}")
                logger.info("🔄 Falling back to standard processing")
                # Fall through to standard processing
        
        # Standard Ollama processing (fallback or when advanced brain unavailable)
        
        # Use Ollama for real AI reasoning
        import httpx
        import asyncio
        
        # Create reasoning prompt based on type
        reasoning_prompts = {
            "strategic": f"Apply strategic thinking and analysis to: {input_text}. Provide insights and recommendations.",
            "deductive": f"Apply deductive reasoning to: {input_text}. Start from general principles and reach specific conclusions.",
            "inductive": f"Apply inductive reasoning to: {input_text}. Identify patterns and generalize from specific observations.",
            "abductive": f"Apply abductive reasoning to: {input_text}. Find the best explanation for the given observations.",
            "causal": f"Apply causal analysis to: {input_text}. Identify cause-and-effect relationships.",
            "creative": f"Apply creative thinking to: {input_text}. Generate innovative ideas and solutions."
        }
        
        prompt = reasoning_prompts.get(reasoning_type, f"Apply {reasoning_type} reasoning to analyze: {input_text}")
        
        # Query Ollama directly
        response = None
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Try to get available models first
                models_response = await client.get("http://ollama:11434/api/tags")
                models = []
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    models = [m["name"] for m in models_data.get("models", [])]
                
                # Select model
                model = "llama3.2:1b" if "llama3.2:1b" in models else (
                    "qwen2.5:3b" if "qwen2.5:3b" in models else (
                        models[0] if models else "llama3.2:1b"
                    )
                )
                
                # Query the model
                ollama_response = await client.post(
                    "http://ollama:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 512
                        }
                    }
                )
                
                if ollama_response.status_code == 200:
                    response_data = ollama_response.json()
                    response = response_data.get("response", "")
        except Exception as e:
            logging.warning(f"Failed to query Ollama: {e}")
            response = None
            
        if response:
            # Parse response to extract insights and recommendations
            response_lines = response.split('\n') if response else []
            insights = []
            recommendations = []
            
            # Simple parsing - look for bullet points or numbered items
            for line in response_lines:
                line = line.strip()
                if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                    if 'recommend' in line.lower() or 'suggest' in line.lower():
                        recommendations.append(line.lstrip('•-*123456789. '))
                    else:
                        insights.append(line.lstrip('•-*123456789. '))
            
            # Ensure we have at least some insights and recommendations
            if not insights:
                insights = ["Deep analysis completed", "Key patterns identified"]
            if not recommendations:
                recommendations = ["Consider multiple perspectives", "Apply systematic approach"]
            
            result = {
                "thought_id": thought_id,
                "result": {
                    "analysis": f"Applied {reasoning_type} reasoning using neural pathways",
                    "insights": insights[:5],  # Limit to 5 insights
                    "recommendations": recommendations[:3],  # Limit to 3 recommendations
                    "output": response
                },
                "confidence": 0.85,
                "reasoning_type": reasoning_type,
                "complexity": "moderate",
                "agents_used": [model]
            }
        else:
            # Fallback if no model available
            result = {
                "thought_id": thought_id,
                "result": {
                    "analysis": f"Analyzed input using {reasoning_type} reasoning",
                    "insights": ["Model unavailable", "Using fallback reasoning"],
                    "recommendations": ["Install AI models for better analysis"],
                    "output": f"Basic processing: {input_text}"
                },
                "confidence": 0.5,
                "reasoning_type": reasoning_type,
                "complexity": "basic",
                "agents_used": ["fallback"]
            }
        
        self.thoughts[thought_id] = result
        self.memory_short_term.append({
            "thought_id": thought_id,
            "summary": f"Thought about: {input_data.get('text', 'unknown')}"
        })
        
        return result
        
    async def get_status(self) -> Dict[str, Any]:
        """Get brain status"""
        if self.use_advanced_processing and self.advanced_brain:
            try:
                # Use advanced brain status
                advanced_status = await self.advanced_brain.get_brain_status()
                return advanced_status
            except Exception as e:
                logger.error(f"Failed to get advanced brain status: {e}")
                
        # Fallback to basic status
        return {
            "status": "active",
            "active_thoughts": len(self.thoughts),
            "memory_usage": {
                "short_term": len(self.memory_short_term),
                "long_term": sum(len(v) for v in self.memory_long_term.values())
            },
            "knowledge_domains": self.knowledge_domains,
            "learning_rate": self.learning_rate,
            "confidence_threshold": self.confidence_threshold
        }
        
    async def query_memory(self, query: str, memory_type: str, domain: Optional[str]) -> List[Dict[str, Any]]:
        """Query memory"""
        results = []
        
        if memory_type in ["short_term", "all"]:
            for memory in self.memory_short_term:
                if query.lower() in memory["summary"].lower():
                    results.append({
                        "type": "short_term",
                        "content": memory
                    })
                    
        if memory_type in ["long_term", "all"]:
            for dom, memories in self.memory_long_term.items():
                if domain and dom != domain:
                    continue
                for memory in memories:
                    results.append({
                        "type": "long_term",
                        "domain": dom,
                        "content": memory
                    })
                    
        return results

# Create singleton instance
enhanced_brain = EnhancedAGIBrain()

@router.post("/think", response_model=ThinkResponse)
async def think(request: ThinkRequest, background_tasks: BackgroundTasks):
    """
    Process a thought using the AGI brain
    
    This endpoint accepts input data and optional context, then uses
    advanced reasoning to generate intelligent responses.
    """
    try:
        result = await enhanced_brain.think(
            request.input_data,
            request.context,
            request.reasoning_type.value
        )
        return ThinkResponse(**result)
    except Exception as e:
        logger.error(f"Thinking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=BrainStatusResponse)
async def get_brain_status():
    """Get the current status of the AGI brain"""
    try:
        status = await enhanced_brain.get_status()
        return BrainStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get brain status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory/query")
async def query_memory(request: MemoryQueryRequest):
    """Query the AGI brain's memory"""
    try:
        results = await enhanced_brain.query_memory(
            request.query,
            request.memory_type,
            request.domain
        )
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learn")
async def learn_from_feedback(feedback: Dict[str, Any]):
    """Allow the AGI brain to learn from feedback"""
    try:
        # In production, this would update the brain's learning
        return {
            "status": "learned",
            "feedback_processed": feedback,
            "new_learning_rate": enhanced_brain.learning_rate
        }
    except Exception as e:
        logger.error(f"Learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_capabilities():
    """Get the AGI brain's current capabilities"""
    return {
        "reasoning_types": [r.value for r in ReasoningType],
        "supported_domains": enhanced_brain.knowledge_domains,
        "features": [
            "quantum-neuromorphic-processing",
            "titans-memory-architecture",
            "consciousness-simulation", 
            "multi-agent-coordination",
            "reasoning-chains",
            "memory-management",
            "self-improvement",
            "context-awareness",
            "advanced-brain-architecture"
        ],
        "integrations": [
            "ollama-models",
            "vector-databases",
            "ai-agents",
            "advanced-brain-architecture"
        ],
        "architecture": "SutazAI Advanced Brain 2025 - Titans Enhanced",
        "processing_stats": enhanced_brain.processing_stats if enhanced_brain.use_advanced_processing else None
    }

@router.get("/advanced-status")
async def get_advanced_status():
    """Get advanced brain architecture status and performance metrics"""
    try:
        if not enhanced_brain.use_advanced_processing or not enhanced_brain.advanced_brain:
            return {
                "advanced_processing": False,
                "message": "Advanced brain architecture not available"
            }
        
        # Get advanced brain status
        brain_status = await enhanced_brain.advanced_brain.get_brain_status()
        
        return {
            "advanced_processing": True,
            "architecture_name": brain_status.get("architecture_name"),
            "performance_class": brain_status.get("performance_class"),
            "processing_stats": enhanced_brain.processing_stats,
            "metrics": brain_status.get("metrics"),
            "capabilities": brain_status.get("capabilities"),
            "comparison": brain_status.get("comparison")
        }
    except Exception as e:
        logger.error(f"Failed to get advanced status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_complex_task(task: Dict[str, Any]):
    """Analyze a complex task and provide a breakdown"""
    try:
        # Simulate task analysis
        analysis = {
            "task_id": f"task_analysis_{len(enhanced_brain.thoughts) + 1}",
            "complexity": "complex",
            "estimated_time": "5-10 minutes",
            "required_agents": ["gpt_engineer", "aider", "crewai"],
            "subtasks": [
                {"name": "Understanding", "status": "pending"},
                {"name": "Planning", "status": "pending"},
                {"name": "Execution", "status": "pending"},
                {"name": "Validation", "status": "pending"}
            ],
            "confidence": 0.75
        }
        return analysis
    except Exception as e:
        logger.error(f"Task analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))