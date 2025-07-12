from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from uuid import uuid4

from api.auth import get_current_user, require_admin
from api.database import db_manager
from tools.ml_frameworks import (
    ml_framework_manager, 
    process_text, 
    analyze_code, 
    generate_text,
    get_ml_status
)
from agents.ml_agent import ml_analysis_agent
from memory import vector_memory

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/status")
async def get_ml_framework_status(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get status of all ML frameworks and capabilities."""
    try:
        status = await get_ml_status()
        
        return {
            "status": status,
            "capabilities": {
                "text_processing": "spacy" in status.get("available_pipelines", []),
                "sentiment_analysis": "sentiment" in status.get("available_pipelines", []),
                "text_generation": "text_generation" in status.get("available_pipelines", []),
                "code_analysis": True,
                "onnx_support": status.get("frameworks", {}).get("onnx", False)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting ML status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/text")
async def analyze_text(
    analysis_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Analyze text using comprehensive NLP frameworks."""
    try:
        text = analysis_request.get("text", "")
        include_embeddings = analysis_request.get("include_embeddings", False)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text content is required")
        
        # Process text with ML frameworks
        result = await process_text(text)
        
        # Log the analysis
        await db_manager.log_system_event(
            "info", "ml_analysis", "Text analyzed",
            {
                "user": current_user.get("username"),
                "text_length": len(text),
                "entities_found": len(result.entities)
            }
        )
        
        response_data = {
            "analysis_id": str(uuid4()),
            "text_length": len(text),
            "tokens": result.tokens[:100],  # Limit for response size
            "entities": result.entities,
            "sentiment": result.sentiment,
            "keywords": result.keywords,
            "language": result.language,
            "summary": result.summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Include embeddings if requested
        if include_embeddings and result.embeddings is not None:
            response_data["embeddings"] = {
                "shape": result.embeddings.shape,
                "sample": result.embeddings[:5].tolist()  # First 5 dimensions only
            }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/code")
async def analyze_code_endpoint(
    analysis_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Analyze code using ML-powered code analysis."""
    try:
        code = analysis_request.get("code", "")
        language = analysis_request.get("language", "python")
        detailed = analysis_request.get("detailed", False)
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Use ML agent for comprehensive analysis
        task = {
            "type": "analyze_code",
            "code": code,
            "language": language
        }
        
        analysis_result = await ml_analysis_agent.execute_task(task)
        
        if not analysis_result.get("success"):
            raise HTTPException(status_code=500, detail=analysis_result.get("error"))
        
        # Log the analysis
        await db_manager.log_system_event(
            "info", "ml_analysis", "Code analyzed",
            {
                "user": current_user.get("username"),
                "language": language,
                "code_length": len(code)
            }
        )
        
        response_data = {
            "analysis_id": str(uuid4()),
            "language": language,
            "code_length": len(code),
            "analysis": analysis_result["analysis"],
            "recommendations": analysis_result["recommendations"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Include detailed NLP insights if requested
        if detailed:
            response_data["nlp_insights"] = analysis_result.get("nlp_insights", {})
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/text")
async def generate_text_endpoint(
    generation_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate text using ML frameworks."""
    try:
        prompt = generation_request.get("prompt", "")
        max_length = generation_request.get("max_length", 100)
        framework = generation_request.get("framework", "transformers")
        
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Generate text
        generated_text = await generate_text(prompt, max_length)
        
        # Log the generation
        await db_manager.log_system_event(
            "info", "ml_analysis", "Text generated",
            {
                "user": current_user.get("username"),
                "prompt_length": len(prompt),
                "framework": framework
            }
        )
        
        return {
            "generation_id": str(uuid4()),
            "prompt": prompt,
            "generated_text": generated_text,
            "framework_used": framework,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/security")
async def analyze_security(
    security_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Analyze code security using ML techniques."""
    try:
        code = security_request.get("code", "")
        language = security_request.get("language", "python")
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Use ML agent for security analysis
        task = {
            "type": "assess_security",
            "code": code,
            "language": language
        }
        
        security_result = await ml_analysis_agent.execute_task(task)
        
        if not security_result.get("success"):
            raise HTTPException(status_code=500, detail=security_result.get("error"))
        
        # Log the security analysis
        await db_manager.log_system_event(
            "info", "ml_analysis", "Security analysis performed",
            {
                "user": current_user.get("username"),
                "language": language,
                "security_score": security_result.get("security_score", 0)
            }
        )
        
        return {
            "analysis_id": str(uuid4()),
            "language": language,
            "security_score": security_result["security_score"],
            "vulnerabilities": security_result["vulnerabilities"],
            "recommendations": security_result["recommendations"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing security: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize/performance")
async def optimize_performance(
    optimization_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Optimize code performance using ML insights."""
    try:
        code = optimization_request.get("code", "")
        language = optimization_request.get("language", "python")
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Use ML agent for performance optimization
        task = {
            "type": "optimize_performance",
            "code": code,
            "language": language
        }
        
        optimization_result = await ml_analysis_agent.execute_task(task)
        
        if not optimization_result.get("success"):
            raise HTTPException(status_code=500, detail=optimization_result.get("error"))
        
        # Log the optimization
        await db_manager.log_system_event(
            "info", "ml_analysis", "Performance optimization performed",
            {
                "user": current_user.get("username"),
                "language": language,
                "optimizations_count": len(optimization_result.get("optimizations", []))
            }
        )
        
        return {
            "optimization_id": str(uuid4()),
            "language": language,
            "current_analysis": optimization_result["current_analysis"],
            "optimizations": optimization_result["optimizations"],
            "estimated_improvement": optimization_result["estimated_improvement"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/documentation")
async def generate_documentation(
    doc_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate documentation using ML analysis."""
    try:
        code = doc_request.get("code", "")
        doc_type = doc_request.get("doc_type", "api")
        language = doc_request.get("language", "python")
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Use ML agent for documentation generation
        task = {
            "type": "generate_documentation",
            "code": code,
            "doc_type": doc_type,
            "language": language
        }
        
        doc_result = await ml_analysis_agent.execute_task(task)
        
        if not doc_result.get("success"):
            raise HTTPException(status_code=500, detail=doc_result.get("error"))
        
        # Log the documentation generation
        await db_manager.log_system_event(
            "info", "ml_analysis", "Documentation generated",
            {
                "user": current_user.get("username"),
                "doc_type": doc_type,
                "language": language
            }
        )
        
        return {
            "doc_id": str(uuid4()),
            "doc_type": doc_type,
            "language": language,
            "documentation": doc_result["documentation"],
            "analysis_used": doc_result.get("analysis_used", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating documentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/benchmark")
async def benchmark_frameworks(
    benchmark_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Benchmark different ML frameworks on the same task."""
    try:
        test_text = benchmark_request.get("test_text", "This is a sample text for benchmarking ML frameworks.")
        
        # Run benchmarks
        benchmarks = await ml_framework_manager.benchmark_frameworks(test_text)
        
        # Log the benchmark
        await db_manager.log_system_event(
            "info", "ml_analysis", "Framework benchmark performed",
            {
                "user": current_user.get("username"),
                "test_text_length": len(test_text)
            }
        )
        
        return {
            "benchmark_id": str(uuid4()),
            "test_text_length": len(test_text),
            "benchmarks": benchmarks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error benchmarking frameworks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/semantic")
async def semantic_search(
    search_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Perform semantic search using ML embeddings."""
    try:
        query = search_request.get("query", "")
        limit = search_request.get("limit", 10)
        similarity_threshold = search_request.get("similarity_threshold", 0.7)
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query is required")
        
        # Process query with NLP
        query_analysis = await process_text(query)
        
        # Perform semantic search in vector memory
        search_results = await vector_memory.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        # Log the search
        await db_manager.log_system_event(
            "info", "ml_analysis", "Semantic search performed",
            {
                "user": current_user.get("username"),
                "query": query,
                "results_count": len(search_results)
            }
        )
        
        return {
            "search_id": str(uuid4()),
            "query": query,
            "query_analysis": {
                "entities": query_analysis.entities,
                "keywords": query_analysis.keywords
            },
            "results": search_results,
            "results_count": len(search_results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/available")
async def get_available_models(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get information about available ML models and frameworks."""
    try:
        status = await get_ml_status()
        
        available_models = {
            "spacy_models": ["en_core_web_sm"] if "spacy_en" in status.get("available_pipelines", []) else [],
            "transformers_models": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "gpt2",
                "sentence-transformers/all-MiniLM-L6-v2"
            ] if status.get("frameworks", {}).get("transformers") else [],
            "frameworks": status.get("frameworks", {}),
            "capabilities": [
                "text_processing",
                "sentiment_analysis", 
                "entity_extraction",
                "code_analysis",
                "text_generation",
                "semantic_search",
                "security_analysis",
                "performance_optimization"
            ]
        }
        
        return {
            "available_models": available_models,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/onnx/convert")
async def convert_to_onnx(
    conversion_request: Dict[str, Any],
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """Convert PyTorch model to ONNX format (admin only)."""
    try:
        model_name = conversion_request.get("model_name", "")
        input_shape = conversion_request.get("input_shape", [1, 784])
        
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        # This would require a pre-trained PyTorch model
        # For now, return a placeholder response
        success = False  # await ml_framework_manager.create_onnx_model(model, input_shape, model_name)
        
        # Log the conversion attempt
        await db_manager.log_system_event(
            "info", "ml_analysis", "ONNX conversion attempted",
            {
                "user": current_user.get("username"),
                "model_name": model_name,
                "success": success
            }
        )
        
        return {
            "conversion_id": str(uuid4()),
            "model_name": model_name,
            "success": success,
            "message": "ONNX conversion feature requires pre-trained PyTorch models",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error converting to ONNX: {e}")
        raise HTTPException(status_code=500, detail=str(e))