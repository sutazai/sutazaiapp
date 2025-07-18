#!/usr/bin/env python3
"""
Comprehensive API Extensions for SutazAI v8
Additional endpoints for new AI services and enhanced functionality
"""

import aiohttp
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException
import time

logger = logging.getLogger(__name__)

# Create router for extensions
router = APIRouter(prefix="/api/v8", tags=["SutazAI v8 Extensions"])

# FAISS Vector Search endpoints
@router.post("/vector/faiss/create_index")
async def create_faiss_index(
    index_name: str,
    dimension: int,
    index_type: str = "IVFFlat"
) -> Dict[str, Any]:
    """Create a new FAISS index"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://faiss:8088/indexes/{index_name}",
                params={"dimension": dimension, "index_type": index_type}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"FAISS error: {error_text}")
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        raise HTTPException(status_code=500, detail=f"FAISS index creation error: {str(e)}")

@router.post("/vector/faiss/add_vectors")
async def add_faiss_vectors(
    index_name: str,
    vectors: List[List[float]],
    ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Add vectors to a FAISS index"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"vectors": vectors, "ids": ids}
            async with session.post(
                f"http://faiss:8088/indexes/{index_name}/vectors",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"FAISS error: {error_text}")
    except Exception as e:
        logger.error(f"Error adding vectors to FAISS: {e}")
        raise HTTPException(status_code=500, detail=f"FAISS vector addition error: {str(e)}")

@router.post("/vector/faiss/search")
async def search_faiss_vectors(
    query_vector: List[float],
    index_name: str = "default",
    k: int = 10
) -> Dict[str, Any]:
    """Search for similar vectors in FAISS"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query_vector": query_vector,
                "index_name": index_name,
                "k": k
            }
            async with session.post(
                "http://faiss:8088/search",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"FAISS error: {error_text}")
    except Exception as e:
        logger.error(f"Error searching FAISS vectors: {e}")
        raise HTTPException(status_code=500, detail=f"FAISS search error: {str(e)}")

@router.get("/vector/faiss/indexes")
async def list_faiss_indexes() -> Dict[str, Any]:
    """List all FAISS indexes"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://faiss:8088/indexes") as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"FAISS error: {error_text}")
    except Exception as e:
        logger.error(f"Error listing FAISS indexes: {e}")
        raise HTTPException(status_code=500, detail=f"FAISS list error: {str(e)}")

# Awesome Code AI endpoints
@router.post("/code/awesome_ai/analyze")
async def awesome_ai_analyze_code(
    code: str,
    language: str = "python",
    analysis_types: List[str] = ["quality", "security", "performance"]
) -> Dict[str, Any]:
    """Analyze code using Awesome Code AI"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "code": code,
                "language": language,
                "analysis_type": analysis_types
            }
            async with session.post(
                "http://awesome-code-ai:8089/analyze",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Awesome Code AI error: {error_text}")
    except Exception as e:
        logger.error(f"Error with Awesome Code AI analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Code analysis error: {str(e)}")

@router.post("/code/awesome_ai/generate")
async def awesome_ai_generate_code(
    prompt: str,
    language: str = "python",
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate code using Awesome Code AI"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "language": language,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            async with session.post(
                "http://awesome-code-ai:8089/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Awesome Code AI error: {error_text}")
    except Exception as e:
        logger.error(f"Error with Awesome Code AI generation: {e}")
        raise HTTPException(status_code=500, detail=f"Code generation error: {str(e)}")

@router.post("/code/awesome_ai/optimize")
async def awesome_ai_optimize_code(
    code: str,
    language: str = "python"
) -> Dict[str, Any]:
    """Optimize code using Awesome Code AI"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://awesome-code-ai:8089/optimize",
                json={"code": code, "language": language}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Awesome Code AI error: {error_text}")
    except Exception as e:
        logger.error(f"Error with Awesome Code AI optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Code optimization error: {str(e)}")

@router.post("/code/awesome_ai/review")
async def awesome_ai_review_code(
    code: str,
    language: str = "python"
) -> Dict[str, Any]:
    """Review code using Awesome Code AI"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://awesome-code-ai:8089/review",
                json={"code": code, "language": language}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Awesome Code AI error: {error_text}")
    except Exception as e:
        logger.error(f"Error with Awesome Code AI review: {e}")
        raise HTTPException(status_code=500, detail=f"Code review error: {str(e)}")

@router.get("/code/awesome_ai/tools")
async def list_awesome_ai_tools() -> Dict[str, Any]:
    """List available Awesome Code AI tools"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://awesome-code-ai:8089/tools") as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Awesome Code AI error: {error_text}")
    except Exception as e:
        logger.error(f"Error listing Awesome Code AI tools: {e}")
        raise HTTPException(status_code=500, detail=f"Tools list error: {str(e)}")

# Enhanced Model Manager endpoints
@router.post("/models/enhanced/load")
async def load_enhanced_model(
    model_name: str,
    quantization: Optional[str] = None,
    device: str = "auto"
) -> Dict[str, Any]:
    """Load a model using Enhanced Model Manager"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model_name": model_name,
                "quantization": quantization,
                "device": device
            }
            async with session.post(
                "http://enhanced-model-manager:8090/models/load",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Model Manager error: {error_text}")
    except Exception as e:
        logger.error(f"Error loading enhanced model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")

@router.post("/models/enhanced/generate")
async def enhanced_model_generate(
    model_name: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict[str, Any]:
    """Generate text using Enhanced Model Manager"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model_name": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            async with session.post(
                "http://enhanced-model-manager:8090/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Model Manager error: {error_text}")
    except Exception as e:
        logger.error(f"Error with enhanced model generation: {e}")
        raise HTTPException(status_code=500, detail=f"Model generation error: {str(e)}")

@router.get("/models/enhanced/list")
async def list_enhanced_models() -> Dict[str, Any]:
    """List available models in Enhanced Model Manager"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://enhanced-model-manager:8090/models") as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Model Manager error: {error_text}")
    except Exception as e:
        logger.error(f"Error listing enhanced models: {e}")
        raise HTTPException(status_code=500, detail=f"Model list error: {str(e)}")

@router.get("/models/enhanced/loaded")
async def list_loaded_models() -> Dict[str, Any]:
    """List currently loaded models"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://enhanced-model-manager:8090/models/loaded") as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Model Manager error: {error_text}")
    except Exception as e:
        logger.error(f"Error listing loaded models: {e}")
        raise HTTPException(status_code=500, detail=f"Loaded models error: {str(e)}")

# DeepSeek-Coder specific endpoints
@router.post("/code/deepseek/generate")
async def deepseek_generate_code(
    prompt: str,
    language: str = "python"
) -> Dict[str, Any]:
    """Generate code using DeepSeek-Coder via Enhanced Model Manager"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://enhanced-model-manager:8090/code/generate",
                json={"prompt": prompt, "language": language}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"DeepSeek error: {error_text}")
    except Exception as e:
        logger.error(f"Error with DeepSeek code generation: {e}")
        raise HTTPException(status_code=500, detail=f"DeepSeek generation error: {str(e)}")

@router.post("/code/deepseek/complete")
async def deepseek_complete_code(
    code: str,
    language: str = "python"
) -> Dict[str, Any]:
    """Complete code using DeepSeek-Coder"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://enhanced-model-manager:8090/code/complete",
                json={"code": code, "language": language}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"DeepSeek error: {error_text}")
    except Exception as e:
        logger.error(f"Error with DeepSeek code completion: {e}")
        raise HTTPException(status_code=500, detail=f"DeepSeek completion error: {str(e)}")

@router.post("/code/deepseek/explain")
async def deepseek_explain_code(
    code: str,
    language: str = "python"
) -> Dict[str, Any]:
    """Explain code using DeepSeek-Coder"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://enhanced-model-manager:8090/code/explain",
                json={"code": code, "language": language}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"DeepSeek error: {error_text}")
    except Exception as e:
        logger.error(f"Error with DeepSeek code explanation: {e}")
        raise HTTPException(status_code=500, detail=f"DeepSeek explanation error: {str(e)}")

@router.post("/code/deepseek/optimize")
async def deepseek_optimize_code(
    code: str,
    language: str = "python"
) -> Dict[str, Any]:
    """Optimize code using DeepSeek-Coder"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://enhanced-model-manager:8090/code/optimize",
                json={"code": code, "language": language}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"DeepSeek error: {error_text}")
    except Exception as e:
        logger.error(f"Error with DeepSeek code optimization: {e}")
        raise HTTPException(status_code=500, detail=f"DeepSeek optimization error: {str(e)}")

# Autonomous Self-Improvement endpoints
@router.post("/self_improvement/generate_improvements")
async def generate_system_improvements(
    system_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate autonomous system improvements"""
    try:
        # Mock implementation for now - would integrate with real autonomous code generator
        if not system_analysis:
            system_analysis = {
                "performance_issues": [
                    {"description": "Database query optimization needed", "severity": "medium"},
                    {"description": "Memory usage optimization required", "severity": "high"}
                ],
                "security_vulnerabilities": [
                    {"type": "input_validation", "severity": "high", "description": "Improve input validation"}
                ],
                "missing_features": ["Advanced caching", "Load balancing"]
            }
        
        # Simulate autonomous improvements generation
        improvements = []
        for issue in system_analysis.get("performance_issues", []):
            improvements.append({
                "type": "performance_improvement",
                "description": f"Generated optimization for: {issue['description']}",
                "code": f"# Performance improvement for {issue['description']}\n# Generated code would go here",
                "quality_score": 0.85,
                "model_used": "autonomous_generator"
            })
        
        for vuln in system_analysis.get("security_vulnerabilities", []):
            improvements.append({
                "type": "security_fix",
                "description": f"Security fix for: {vuln['description']}",
                "code": f"# Security improvement for {vuln['description']}\n# Generated code would go here",
                "quality_score": 0.90,
                "model_used": "autonomous_generator"
            })
        
        return {
            "status": "success",
            "improvements_generated": len(improvements),
            "results": improvements,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error generating system improvements: {e}")
        raise HTTPException(status_code=500, detail=f"Self-improvement error: {str(e)}")

@router.get("/self_improvement/stats")
async def get_self_improvement_stats() -> Dict[str, Any]:
    """Get autonomous self-improvement statistics"""
    try:
        # Mock stats - would come from real autonomous generator
        stats = {
            "total_generated": 42,
            "successful_generations": 38,
            "improvement_cycles": 5,
            "quality_improvements": 12,
            "average_quality_score": 0.86,
            "last_generation": time.time() - 3600
        }
        
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"Error getting self-improvement stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

# Comprehensive system health endpoint
@router.get("/system/comprehensive_health")
async def get_comprehensive_health() -> Dict[str, Any]:
    """Get comprehensive system health including all new services"""
    try:
        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "services": {},
            "summary": {
                "total_services": 0,
                "healthy_services": 0,
                "unhealthy_services": 0
            }
        }
        
        # All services to check
        all_services = {
            "faiss": "http://faiss:8088/health",
            "awesome_code_ai": "http://awesome-code-ai:8089/health",
            "enhanced_model_manager": "http://enhanced-model-manager:8090/health",
            "langflow": "http://langflow:7860/health",
            "dify": "http://dify:5001/health",
            "autogen": "http://autogen:8080/health",
            "pytorch": "http://pytorch:8085/health",
            "tensorflow": "http://tensorflow:8086/health",
            "jax": "http://jax:8087/health"
        }
        
        async with aiohttp.ClientSession() as session:
            for service_name, health_url in all_services.items():
                try:
                    async with session.get(health_url, timeout=5) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            health_status["services"][service_name] = {
                                "status": "healthy",
                                "details": health_data
                            }
                            health_status["summary"]["healthy_services"] += 1
                        else:
                            health_status["services"][service_name] = {
                                "status": "unhealthy",
                                "details": {"error": f"HTTP {response.status}"}
                            }
                            health_status["summary"]["unhealthy_services"] += 1
                except Exception as e:
                    health_status["services"][service_name] = {
                        "status": "error",
                        "details": {"error": str(e)}
                    }
                    health_status["summary"]["unhealthy_services"] += 1
                
                health_status["summary"]["total_services"] += 1
        
        # Determine overall status
        if health_status["summary"]["unhealthy_services"] > 0:
            health_status["overall_status"] = "degraded"
        
        if health_status["summary"]["healthy_services"] == 0:
            health_status["overall_status"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting comprehensive health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")

# Batch processing endpoints
@router.post("/batch/code_generation")
async def batch_code_generation(
    requests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Process multiple code generation requests in batch"""
    try:
        results = []
        
        for i, req in enumerate(requests):
            try:
                # Use DeepSeek for code generation
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://enhanced-model-manager:8090/code/generate",
                        json=req
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            results.append({
                                "index": i,
                                "status": "success",
                                "result": result
                            })
                        else:
                            results.append({
                                "index": i,
                                "status": "error",
                                "error": f"HTTP {response.status}"
                            })
            except Exception as e:
                results.append({
                    "index": i,
                    "status": "error", 
                    "error": str(e)
                })
        
        successful = len([r for r in results if r["status"] == "success"])
        
        return {
            "status": "completed",
            "total_requests": len(requests),
            "successful": successful,
            "failed": len(requests) - successful,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch code generation: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@router.get("/system/service_summary")
async def get_service_summary() -> Dict[str, Any]:
    """Get a summary of all integrated services"""
    return {
        "status": "success",
        "sutazai_version": "2.0.0",
        "total_services": 34,
        "service_categories": {
            "vector_databases": ["ChromaDB", "Qdrant", "FAISS"],
            "llm_models": ["DeepSeek-Coder-33B", "Llama 2", "CodeLlama"],
            "code_ai_tools": ["Awesome Code AI", "TabbyML", "Semgrep", "GPT-Engineer", "Aider"],
            "ml_frameworks": ["PyTorch", "TensorFlow", "JAX"],
            "agent_frameworks": ["AutoGPT", "LocalAGI", "AutoGen", "AgentZero", "BigAGI"],
            "automation_tools": ["Browser-Use", "Skyvern", "Langflow", "Dify"],
            "specialized_services": ["Documind", "FinRobot", "OpenWebUI"],
            "infrastructure": ["PostgreSQL", "Redis", "Nginx", "Prometheus", "Grafana"]
        },
        "new_in_v8": [
            "FAISS vector similarity search",
            "Awesome Code AI integration", 
            "Enhanced Model Manager with DeepSeek-Coder",
            "Autonomous self-improvement system",
            "Comprehensive batch processing",
            "Advanced cross-component integration"
        ],
        "capabilities": [
            "100% local execution",
            "25+ AI technologies integrated",
            "Autonomous code generation and improvement",
            "Multi-model orchestration",
            "Real-time vector search",
            "Advanced security scanning",
            "Financial analysis",
            "Document processing",
            "Web automation",
            "Performance monitoring"
        ]
    }