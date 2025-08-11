"""
Streaming API Endpoints for SutazAI with XSS Protection
Provides real-time streaming responses using advanced model management
"""
import asyncio
import json
import html
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime

from app.services.consolidated_ollama_service import ConsolidatedOllamaService, StreamingResponse as ModelStreamingResponse
from app.core.dependencies import get_advanced_model_manager
from app.core.security import xss_protection

logger = logging.getLogger(__name__)

router = APIRouter()


class StreamingChatRequest(BaseModel):
    """Streaming chat request model with XSS protection"""
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    model: Optional[str] = Field(None, description="Model to use for generation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stream: bool = Field(True, description="Enable streaming")
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate and sanitize chat messages"""
        if not v:
            raise ValueError("Messages cannot be empty")
        
        sanitized_messages = []
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")
            
            # Validate required fields
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            
            # Sanitize content
            try:
                sanitized_content = xss_protection.validator.validate_input(msg['content'], "chat_message")
                sanitized_role = xss_protection.validator.validate_input(msg['role'], "text")
                
                sanitized_messages.append({
                    'role': sanitized_role,
                    'content': sanitized_content
                })
            except ValueError as e:
                raise ValueError(f"Invalid message content: {str(e)}")
        
        return sanitized_messages
    
    @validator('model')
    def validate_model(cls, v):
        from app.utils.validation import validate_model_name
        return validate_model_name(v)


class StreamingTextRequest(BaseModel):
    """Streaming text generation request model with XSS protection"""
    prompt: str = Field(..., description="Text prompt for generation")
    model: Optional[str] = Field(None, description="Model to use for generation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stream: bool = Field(True, description="Enable streaming")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate and sanitize prompt"""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            return xss_protection.validator.validate_input(v, "text")
        except ValueError as e:
            raise ValueError(f"Invalid prompt content: {str(e)}")
    
    @validator('model')
    def validate_model(cls, v):
        from app.utils.validation import validate_model_name
        return validate_model_name(v)


class BatchRequest(BaseModel):
    """Batch processing request model"""
    requests: List[Dict[str, Any]] = Field(..., description="List of requests to process in batch")
    model: Optional[str] = Field(None, description="Model to use for all requests")
    batch_size: Optional[int] = Field(8, ge=1, le=32, description="Batch size for processing")


class CacheManagementRequest(BaseModel):
    """Cache management request model"""
    action: str = Field(..., description="Cache action: save, load, clear, status")
    models: Optional[List[str]] = Field(None, description="Specific models to manage")


@router.post("/chat/stream", 
    summary="Streaming Chat",
    description="Generate streaming chat responses for real-time conversation")
async def stream_chat(
    request: StreamingChatRequest,
    advanced_model_manager: ConsolidatedOllamaService = Depends(get_advanced_model_manager)
):
    """
    Stream chat responses in real-time
    
    This endpoint provides Server-Sent Events (SSE) for real-time chat streaming.
    Each chunk contains a JSON object with the response data.
    """
    try:
        async def generate_stream():
            """Generate streaming chat responses"""
            try:
                # Add SSE headers
                yield "data: " + json.dumps({
                    "type": "start",
                    "timestamp": datetime.now().isoformat(),
                    "model": request.model or advanced_model_manager.default_model
                }) + "\n\n"
                
                # Stream the actual chat response
                async for chunk in advanced_model_manager.chat_streaming(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p
                ):
                    # Sanitize chunk content before streaming
                    safe_content = chunk.chunk
                    try:
                        safe_content = xss_protection.validator.validate_input(chunk.chunk, "text")
                    except ValueError:
                        safe_content = "[Content filtered for security]"
                        logger.warning("Streaming content blocked by XSS protection")
                    
                    chunk_data = {
                        "type": "chunk",
                        "request_id": chunk.request_id,
                        "content": safe_content,
                        "is_final": chunk.is_final,
                        "metadata": chunk.metadata
                    }
                    
                    yield "data: " + json.dumps(chunk_data) + "\n\n"
                    
                    if chunk.is_final:
                        break
                
                # Send completion signal
                yield "data: " + json.dumps({
                    "type": "complete",
                    "timestamp": datetime.now().isoformat()
                }) + "\n\n"
                
            except Exception as e:
                logger.error(f"Error in chat streaming: {e}")
                yield "data: " + json.dumps({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }) + "\n\n"
        
        # Get secure CORS configuration
        from app.core.cors_security import cors_security
        allowed_origins = cors_security.get_allowed_origins()
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": ", ".join(allowed_origins),
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Cache-Control",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up chat stream: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming setup failed: {str(e)}")


@router.post("/text/stream",
    summary="Streaming Text Generation", 
    description="Generate streaming text responses for real-time text generation")
async def stream_text(
    request: StreamingTextRequest,
    advanced_model_manager: ConsolidatedOllamaService = Depends(get_advanced_model_manager)
):
    """
    Stream text generation responses in real-time
    """
    try:
        async def generate_stream():
            """Generate streaming text responses"""
            try:
                # Add SSE headers
                yield "data: " + json.dumps({
                    "type": "start",
                    "timestamp": datetime.now().isoformat(),
                    "model": request.model or advanced_model_manager.default_model,
                    "prompt_preview": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
                }) + "\n\n"
                
                # Stream the text generation
                async for chunk in advanced_model_manager.generate_streaming(
                    prompt=request.prompt,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p
                ):
                    # Sanitize chunk content before streaming
                    safe_content = chunk.chunk
                    try:
                        safe_content = xss_protection.validator.validate_input(chunk.chunk, "text")
                    except ValueError:
                        safe_content = "[Content filtered for security]"
                        logger.warning("Text streaming content blocked by XSS protection")
                    
                    chunk_data = {
                        "type": "chunk",
                        "request_id": chunk.request_id,
                        "content": safe_content,
                        "is_final": chunk.is_final,
                        "metadata": chunk.metadata
                    }
                    
                    yield "data: " + json.dumps(chunk_data) + "\n\n"
                    
                    if chunk.is_final:
                        break
                
                # Send completion signal
                yield "data: " + json.dumps({
                    "type": "complete",
                    "timestamp": datetime.now().isoformat()
                }) + "\n\n"
                
            except Exception as e:
                logger.error(f"Error in text streaming: {e}")
                yield "data: " + json.dumps({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }) + "\n\n"
        
        # Get secure CORS configuration
        from app.core.cors_security import cors_security
        allowed_origins = cors_security.get_allowed_origins()
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": ", ".join(allowed_origins),
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Cache-Control",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up text stream: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming setup failed: {str(e)}")


@router.post("/batch/process",
    summary="Batch Processing",
    description="Process multiple requests in batches for high-concurrency scenarios")
async def process_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    advanced_model_manager: ConsolidatedOllamaService = Depends(get_advanced_model_manager)
):
    """
    Process multiple requests in batches for optimal throughput
    """
    try:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process requests in parallel batches
        batch_results = []
        
        for i in range(0, len(request.requests), request.batch_size):
            batch = request.requests[i:i + request.batch_size]
            
            # Process batch in parallel
            batch_tasks = []
            for req in batch:
                if "prompt" in req:
                    task = advanced_model_manager.generate_text(
                        prompt=req["prompt"],
                        model=req.get("model", request.model),
                        **{k: v for k, v in req.items() if k not in ["prompt", "model"]}
                    )
                elif "messages" in req:
                    # Convert to text for now - could be enhanced for chat
                    last_message = req["messages"][-1]["content"] if req["messages"] else ""
                    task = advanced_model_manager.generate_text(
                        prompt=last_message,
                        model=req.get("model", request.model),
                        **{k: v for k, v in req.items() if k not in ["messages", "model"]}
                    )
                else:
                    continue
                
                batch_tasks.append(task)
            
            # Execute batch
            if batch_tasks:
                batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, response in enumerate(batch_responses):
                    result = {
                        "request_index": i + j,
                        "success": not isinstance(response, Exception),
                        "response": str(response) if not isinstance(response, Exception) else "",
                        "error": str(response) if isinstance(response, Exception) else None
                    }
                    batch_results.append(result)
        
        return {
            "batch_id": batch_id,
            "total_requests": len(request.requests),
            "batch_size": request.batch_size,
            "results": batch_results,
            "processing_time": datetime.now().isoformat(),
            "success_rate": sum(1 for r in batch_results if r["success"]) / len(batch_results) if batch_results else 0
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/cache/manage",
    summary="Cache Management", 
    description="Manage model cache for performance optimization")
async def manage_cache(
    request: CacheManagementRequest,
    advanced_model_manager: ConsolidatedOllamaService = Depends(get_advanced_model_manager)
):
    """
    Manage model cache for performance optimization
    """
    try:
        if request.action == "status":
            return await advanced_model_manager.get_performance_metrics()
        
        elif request.action == "save":
            cache_bytes = await advanced_model_manager.save_cache_artifacts()
            return {
                "action": "save",
                "cache_size_bytes": len(cache_bytes),
                "timestamp": datetime.now().isoformat(),
                "success": len(cache_bytes) > 0
            }
        
        elif request.action == "clear":
            # Clear cache for specific models or all
            if request.models:
                for model in request.models:
                    if model in advanced_model_manager.warm_cache:
                        del advanced_model_manager.warm_cache[model]
                cleared_count = len(request.models)
            else:
                cleared_count = len(advanced_model_manager.warm_cache)
                advanced_model_manager.warm_cache.clear()
            
            return {
                "action": "clear",
                "cleared_models": cleared_count,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        
        elif request.action == "warmup":
            # Warm up specific models
            if not request.models:
                raise HTTPException(status_code=400, detail="Models list required for warmup action")
            
            warmup_results = []
            for model in request.models:
                try:
                    success = await advanced_model_manager._warm_model(model)
                    warmup_results.append({
                        "model": model,
                        "success": success
                    })
                except Exception as e:
                    warmup_results.append({
                        "model": model,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "action": "warmup",
                "results": warmup_results,
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown cache action: {request.action}")
            
    except Exception as e:
        logger.error(f"Error in cache management: {e}")
        raise HTTPException(status_code=500, detail=f"Cache management failed: {str(e)}")


@router.get("/performance/metrics",
    summary="Performance Metrics",
    description="Get comprehensive performance metrics for the advanced model manager")
async def get_performance_metrics(
    advanced_model_manager: ConsolidatedOllamaService = Depends(get_advanced_model_manager)
):
    """
    Get comprehensive performance metrics
    """
    try:
        return await advanced_model_manager.get_performance_metrics()
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/health/advanced",
    summary="Advanced Health Check",
    description="Comprehensive health check for advanced model manager")
async def advanced_health_check(
    advanced_model_manager: ConsolidatedOllamaService = Depends(get_advanced_model_manager)
):
    """
    Comprehensive health check for the advanced model manager
    """
    try:
        health_status = advanced_model_manager.get_health_status()
        performance_metrics = await advanced_model_manager.get_performance_metrics()
        
        return {
            "status": "healthy" if health_status["healthy"] else "unhealthy",
            "service_name": health_status["service_name"],
            "uptime_seconds": health_status["uptime_seconds"],
            "gpu_available": advanced_model_manager.gpu_available,
            "cache_status": performance_metrics["cache_status"],
            "active_processors": performance_metrics["system_info"]["active_batch_processors"],
            "pending_requests": performance_metrics["system_info"]["pending_batch_requests"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in advanced health check: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}") 
