from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from uuid import uuid4
import os
import tempfile
from pathlib import Path

from api.auth import get_current_user, require_admin
from api.database import db_manager
from tools.advanced_frameworks import (
    advanced_framework_manager,
    process_image,
    analyze_text_advanced,
    get_advanced_capabilities,
    create_fast_nn
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/capabilities")
async def get_capabilities(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get advanced AI framework capabilities and status."""
    try:
        capabilities = await get_advanced_capabilities()
        
        return {
            "capabilities": capabilities,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vision/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    operations: List[str] = ["detect_faces", "extract_features"],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Analyze image using computer vision frameworks."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process image
            result = await process_image(tmp_file_path, operations)
            
            # Log the analysis
            await db_manager.log_system_event(
                "info", "advanced_ai", "Image analyzed",
                {
                    "user": current_user.get("username"),
                    "filename": file.filename,
                    "operations": operations,
                    "file_size": len(content)
                }
            )
            
            return {
                "analysis_id": str(uuid4()),
                "filename": file.filename,
                "file_size": len(content),
                "operations": operations,
                "results": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nlp/advanced")
async def advanced_nlp_analysis(
    analysis_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Advanced NLP analysis using specialized frameworks."""
    try:
        text = analysis_request.get("text", "")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text content is required")
        
        if len(text) > 100000:  # 100KB limit
            raise HTTPException(status_code=400, detail="Text too long (max 100KB)")
        
        # Perform advanced NLP analysis
        result = await analyze_text_advanced(text)
        
        # Log the analysis
        await db_manager.log_system_event(
            "info", "advanced_ai", "Advanced NLP analysis performed",
            {
                "user": current_user.get("username"),
                "text_length": len(text),
                "features_analyzed": len(result.get("analysis", {}))
            }
        )
        
        return {
            "analysis_id": str(uuid4()),
            "text_length": len(text),
            "analysis": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced NLP analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neural-network/fast")
async def create_fast_neural_network(
    network_config: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create and train a fast neural network using FANN."""
    try:
        name = network_config.get("name", f"network_{str(uuid4())[:8]}")
        layers = network_config.get("layers", [2, 3, 1])
        
        # Validate configuration
        if not isinstance(layers, list) or len(layers) < 2:
            raise HTTPException(status_code=400, detail="Invalid network layers configuration")
        
        if any(layer < 1 for layer in layers):
            raise HTTPException(status_code=400, detail="All layer sizes must be positive")
        
        # Create network
        result = await create_fast_nn(network_config)
        
        # Log the creation
        await db_manager.log_system_event(
            "info", "advanced_ai", "Fast neural network created",
            {
                "user": current_user.get("username"),
                "network_name": name,
                "layers": layers,
                "success": "error" not in result
            }
        )
        
        return {
            "network_id": str(uuid4()),
            "network_name": name,
            "configuration": {
                "layers": layers,
                "total_parameters": sum(layers[i] * layers[i+1] for i in range(len(layers)-1))
            },
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating fast neural network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/language/detect")
async def detect_language(
    detection_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Advanced language detection supporting 165+ languages."""
    try:
        text = detection_request.get("text", "")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text content is required")
        
        # Use advanced NLP analysis to get language detection
        analysis_result = await analyze_text_advanced(text)
        language_info = analysis_result.get("analysis", {}).get("language_detection", {})
        
        # Log the detection
        await db_manager.log_system_event(
            "info", "advanced_ai", "Language detection performed",
            {
                "user": current_user.get("username"),
                "text_length": len(text),
                "detected_language": language_info.get("language", "unknown")
            }
        )
        
        return {
            "detection_id": str(uuid4()),
            "text_length": len(text),
            "language_detection": language_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/entities/multilingual")
async def multilingual_entity_extraction(
    extraction_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Multilingual named entity recognition."""
    try:
        text = extraction_request.get("text", "")
        language = extraction_request.get("language")  # Optional language hint
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text content is required")
        
        # Use advanced NLP analysis to get entity extraction
        analysis_result = await analyze_text_advanced(text)
        entities_info = analysis_result.get("analysis", {}).get("entities", {})
        
        # Log the extraction
        await db_manager.log_system_event(
            "info", "advanced_ai", "Multilingual entity extraction performed",
            {
                "user": current_user.get("username"),
                "text_length": len(text),
                "entities_found": len(entities_info.get("entities", [])),
                "language_hint": language
            }
        )
        
        return {
            "extraction_id": str(uuid4()),
            "text_length": len(text),
            "language_hint": language,
            "entity_extraction": entities_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multilingual entity extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sentiment/advanced")
async def advanced_sentiment_analysis(
    sentiment_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Advanced sentiment analysis with word-level insights."""
    try:
        text = sentiment_request.get("text", "")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text content is required")
        
        # Use advanced NLP analysis to get sentiment
        analysis_result = await analyze_text_advanced(text)
        sentiment_info = analysis_result.get("analysis", {}).get("sentiment", {})
        
        # Log the analysis
        await db_manager.log_system_event(
            "info", "advanced_ai", "Advanced sentiment analysis performed",
            {
                "user": current_user.get("username"),
                "text_length": len(text),
                "sentiment_label": sentiment_info.get("sentiment_label", "unknown")
            }
        )
        
        return {
            "sentiment_id": str(uuid4()),
            "text_length": len(text),
            "sentiment_analysis": sentiment_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/benchmark")
async def benchmark_frameworks(
    benchmark_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Benchmark advanced AI frameworks performance."""
    try:
        test_data = benchmark_request.get("test_data", {})
        
        # Add default test data if not provided
        if not test_data.get("text"):
            test_data["text"] = "This is a sample text for benchmarking advanced AI frameworks."
        
        # Run benchmarks
        benchmark_results = await advanced_framework_manager.benchmark_advanced_frameworks(test_data)
        
        # Log the benchmark
        await db_manager.log_system_event(
            "info", "advanced_ai", "Framework benchmark performed",
            {
                "user": current_user.get("username"),
                "frameworks_tested": len(benchmark_results.get("benchmarks", {}))
            }
        )
        
        return {
            "benchmark_id": str(uuid4()),
            "test_configuration": test_data,
            "results": benchmark_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error benchmarking frameworks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/supported")
async def get_supported_models(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get information about supported models and frameworks."""
    try:
        capabilities = await get_advanced_capabilities()
        
        supported_models = {
            "computer_vision": [
                "OpenCV Haar Cascades",
                "MediaPipe",
                "Face Recognition",
                "Custom CNN models"
            ],
            "neural_networks": [
                "FANN Multi-layer Perceptron",
                "Chainer Dynamic Networks",
                "Custom architectures"
            ],
            "nlp": [
                "Polyglot (165+ languages)",
                "AllenNLP research models",
                "Multilingual BERT",
                "Custom language models"
            ],
            "deep_learning": [
                "Caffe models",
                "Darknet YOLO",
                "Custom PyTorch models",
                "TensorFlow models"
            ]
        }
        
        return {
            "supported_models": supported_models,
            "framework_status": capabilities.get("framework_status", {}),
            "capabilities": capabilities.get("capabilities", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting supported models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/advanced")
async def advanced_health_check(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Health check for advanced AI frameworks."""
    try:
        capabilities = await get_advanced_capabilities()
        framework_status = capabilities.get("framework_status", {})
        
        # Calculate health score
        available_frameworks = sum(1 for status in framework_status.values() if status)
        total_frameworks = len(framework_status)
        health_score = (available_frameworks / total_frameworks) * 100 if total_frameworks > 0 else 0
        
        health_status = "healthy" if health_score >= 50 else "degraded" if health_score >= 25 else "unhealthy"
        
        return {
            "status": health_status,
            "health_score": health_score,
            "available_frameworks": available_frameworks,
            "total_frameworks": total_frameworks,
            "framework_details": framework_status,
            "recommendations": [
                "Install missing frameworks for full functionality",
                "Check GPU availability for acceleration",
                "Verify model downloads are complete"
            ] if health_score < 100 else ["All frameworks operational"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in advanced health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))