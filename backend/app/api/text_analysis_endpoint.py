#!/usr/bin/env python3
"""
Text Analysis API Endpoint
===========================

FastAPI endpoint for the Text Analysis Agent.
Provides REST API access to all text analysis capabilities.

This demonstrates how to integrate a real AI agent into the backend.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import asyncio
import logging

# Import the Text Analysis Agent
from ..agents.text_analysis_agent import TextAnalysisAgent, AnalysisType

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/text-analysis",
    tags=["text-analysis"],
    responses={404: {"description": "Not found"}}
)

# Initialize the agent as a singleton
text_agent: Optional[TextAnalysisAgent] = None
agent_lock = asyncio.Lock()


async def get_agent() -> TextAnalysisAgent:
    """Get or create the singleton Text Analysis Agent"""
    global text_agent
    
    async with agent_lock:
        if text_agent is None:
            logger.info("Initializing Text Analysis Agent...")
            text_agent = TextAnalysisAgent()
            success = await text_agent.initialize()
            if not success:
                raise RuntimeError("Failed to initialize Text Analysis Agent")
            logger.info("Text Analysis Agent initialized successfully")
    
    return text_agent


# Pydantic models for request/response
class TextAnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=50000)
    analysis_type: Literal[
        "sentiment", "entities", "summary", "keywords", "language", "full_analysis"
    ] = Field("full_analysis", description="Type of analysis to perform")
    options: Optional[Dict[str, Any]] = Field(
        default={},
        description="Analysis options (e.g., max_sentences for summary, num_keywords for keywords)"
    )


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    sentiment: str
    confidence: float
    reason: Optional[str] = None
    text_length: int
    processing_time: float
    model_used: str
    cached: bool = False
    timestamp: str


class EntitiesResponse(BaseModel):
    """Response model for entity extraction"""
    entities: Dict[str, List[str]]
    entity_count: int
    text_length: int
    processing_time: float
    model_used: str
    cached: bool = False
    timestamp: str


class SummaryResponse(BaseModel):
    """Response model for text summarization"""
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time: float
    model_used: str
    cached: bool = False
    timestamp: str


class KeywordsResponse(BaseModel):
    """Response model for keyword extraction"""
    keywords: List[str]
    keyword_count: int
    text_length: int
    processing_time: float
    model_used: str
    cached: bool = False
    timestamp: str


class LanguageResponse(BaseModel):
    """Response model for language detection"""
    language: str
    confidence: float
    text_sample_length: int
    processing_time: float
    model_used: str
    cached: bool = False
    timestamp: str


class FullAnalysisResponse(BaseModel):
    """Response model for comprehensive analysis"""
    analysis_type: str
    text_length: int
    processing_time: float
    sentiment: Optional[Dict[str, Any]]
    entities: Optional[Dict[str, List[str]]]
    summary: Optional[str]
    keywords: Optional[List[str]]
    language: Optional[Dict[str, Any]]
    confidence: float
    model_used: str
    cached: bool = False
    timestamp: str


class AgentStatsResponse(BaseModel):
    """Response model for agent statistics"""
    agent_id: str
    status: str
    uptime: float
    total_analyses: int
    analysis_metrics: Dict[str, int]
    cache_hit_rate: float
    average_text_length: float
    model_name: str
    error_count: int
    task_count: int


# API Endpoints

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text using the specified analysis type
    
    Available analysis types:
    - sentiment: Analyze sentiment (positive/negative/neutral)
    - entities: Extract named entities (people, organizations, locations, dates)
    - summary: Generate a concise summary
    - keywords: Extract important keywords
    - language: Detect the language
    - full_analysis: Perform all analyses
    """
    try:
        agent = await get_agent()
        
        # Route to appropriate analysis method
        if request.analysis_type == "sentiment":
            result = await agent.analyze_sentiment(request.text)
            
        elif request.analysis_type == "entities":
            result = await agent.extract_entities(request.text)
            
        elif request.analysis_type == "summary":
            max_sentences = request.options.get("max_sentences", 3)
            result = await agent.generate_summary(request.text, max_sentences)
            
        elif request.analysis_type == "keywords":
            num_keywords = request.options.get("num_keywords", 5)
            result = await agent.extract_keywords(request.text, num_keywords)
            
        elif request.analysis_type == "language":
            result = await agent.detect_language(request.text)
            
        elif request.analysis_type == "full_analysis":
            analysis_result = await agent.analyze_text_full(request.text)
            result = analysis_result.to_dict()
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis type: {request.analysis_type}"
            )
        
        return {
            "success": True,
            "analysis_type": request.analysis_type,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(text: str = Query(..., description="Text to analyze")):
    """
    Quick sentiment analysis endpoint
    
    Returns sentiment (positive/negative/neutral) with confidence score
    """
    try:
        agent = await get_agent()
        result = await agent.analyze_sentiment(text)
        return SentimentResponse(**result)
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


@router.post("/entities", response_model=EntitiesResponse)
async def extract_entities(text: str = Query(..., description="Text to analyze")):
    """
    Extract named entities from text
    
    Returns people, organizations, locations, and dates
    """
    try:
        agent = await get_agent()
        result = await agent.extract_entities(text)
        return EntitiesResponse(**result)
        
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Entity extraction failed: {str(e)}"
        )


@router.post("/summary", response_model=SummaryResponse)
async def generate_summary(
    text: str = Query(..., description="Text to summarize"),
    max_sentences: int = Query(3, description="Maximum number of sentences")
):
    """
    Generate a concise summary of the text
    
    Returns a summary with the specified number of sentences
    """
    try:
        agent = await get_agent()
        result = await agent.generate_summary(text, max_sentences)
        return SummaryResponse(**result)
        
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/keywords", response_model=KeywordsResponse)
async def extract_keywords(
    text: str = Query(..., description="Text to analyze"),
    num_keywords: int = Query(5, description="Number of keywords to extract")
):
    """
    Extract important keywords from text
    
    Returns a list of the most significant keywords/phrases
    """
    try:
        agent = await get_agent()
        result = await agent.extract_keywords(text, num_keywords)
        return KeywordsResponse(**result)
        
    except Exception as e:
        logger.error(f"Keyword extraction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Keyword extraction failed: {str(e)}"
        )


@router.post("/language", response_model=LanguageResponse)
async def detect_language(text: str = Query(..., description="Text to analyze")):
    """
    Detect the language of the text
    
    Returns the detected language with confidence score
    """
    try:
        agent = await get_agent()
        result = await agent.detect_language(text)
        return LanguageResponse(**result)
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Language detection failed: {str(e)}"
        )


@router.post("/full", response_model=FullAnalysisResponse)
async def full_analysis(text: str = Query(..., description="Text to analyze")):
    """
    Perform comprehensive text analysis
    
    Combines all analysis types into a single result
    """
    try:
        agent = await get_agent()
        analysis_result = await agent.analyze_text_full(text)
        result = analysis_result.to_dict()
        return FullAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Full analysis failed: {str(e)}"
        )


@router.get("/stats", response_model=AgentStatsResponse)
async def get_agent_stats():
    """
    Get Text Analysis Agent statistics
    
    Returns performance metrics and usage statistics
    """
    try:
        agent = await get_agent()
        stats = await agent.get_agent_stats()
        return AgentStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Check the health of the Text Analysis Agent
    
    Returns agent status and connectivity information
    """
    try:
        agent = await get_agent()
        health = await agent.health_check()
        
        return {
            "healthy": health.get("healthy", False),
            "agent_name": health.get("agent_name"),
            "status": health.get("status"),
            "model": health.get("model"),
            "ollama_healthy": health.get("ollama_healthy", False),
            "backend_healthy": health.get("backend_healthy", False),
            "uptime_seconds": health.get("uptime_seconds", 0),
            "tasks_processed": health.get("tasks_processed", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.post("/batch")
async def batch_analysis(
    texts: List[str] = Query(..., description="List of texts to analyze"),
    analysis_type: str = Query("sentiment", description="Type of analysis")
):
    """
    Perform batch text analysis
    
    Analyze multiple texts in parallel for efficiency
    """
    try:
        agent = await get_agent()
        
        # Create analysis tasks
        tasks = []
        for text in texts[:50]:  # Limit to 50 texts
            if analysis_type == "sentiment":
                tasks.append(agent.analyze_sentiment(text))
            elif analysis_type == "entities":
                tasks.append(agent.extract_entities(text))
            elif analysis_type == "keywords":
                tasks.append(agent.extract_keywords(text))
            elif analysis_type == "language":
                tasks.append(agent.detect_language(text))
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid analysis type for batch: {analysis_type}"
                )
        
        # Run analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful = []
        failed = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({
                    "index": i,
                    "error": str(result)
                })
            else:
                successful.append({
                    "index": i,
                    "result": result
                })
        
        return {
            "success": True,
            "analysis_type": analysis_type,
            "total": len(texts),
            "successful": len(successful),
            "failed": len(failed),
            "results": successful,
            "errors": failed
        }
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.websocket("/stream")
async def websocket_endpoint(websocket):
    """
    WebSocket endpoint for real-time text analysis
    
    Allows streaming analysis for interactive applications
    """
    await websocket.accept()
    
    try:
        agent = await get_agent()
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            text = data.get("text", "")
            analysis_type = data.get("analysis_type", "sentiment")
            
            if not text:
                await websocket.send_json({
                    "error": "No text provided"
                })
                continue
            
            # Perform analysis
            try:
                if analysis_type == "sentiment":
                    result = await agent.analyze_sentiment(text)
                elif analysis_type == "entities":
                    result = await agent.extract_entities(text)
                elif analysis_type == "summary":
                    result = await agent.generate_summary(text)
                elif analysis_type == "keywords":
                    result = await agent.extract_keywords(text)
                elif analysis_type == "language":
                    result = await agent.detect_language(text)
                else:
                    result = {"error": f"Unknown analysis type: {analysis_type}"}
                
                # Send result
                await websocket.send_json({
                    "success": True,
                    "analysis_type": analysis_type,
                    "result": result
                })
                
            except Exception as e:
                await websocket.send_json({
                    "error": str(e)
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Include this router in the main FastAPI app
def include_router(app):
    """Include this router in the main FastAPI application"""
    app.include_router(router)
    logger.info("Text Analysis API endpoints registered")