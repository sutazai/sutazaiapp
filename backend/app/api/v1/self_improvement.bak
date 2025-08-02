"""
Self-Improvement API endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging

from app.services.self_improvement import self_improvement_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/start")
async def start_self_improvement(background_tasks: BackgroundTasks):
    """Start the self-improvement monitoring and optimization process"""
    try:
        # Start monitoring in background
        background_tasks.add_task(self_improvement_service.start_monitoring)
        
        return {
            "status": "started",
            "message": "Self-improvement monitoring initiated",
            "batch_size": self_improvement_service.batch_size,
            "confidence_threshold": self_improvement_service.min_confidence_threshold
        }
    except Exception as e:
        logger.error(f"Error starting self-improvement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_self_improvement():
    """Stop the self-improvement process"""
    try:
        await self_improvement_service.stop_monitoring()
        
        return {
            "status": "stopped",
            "message": "Self-improvement monitoring stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping self-improvement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_self_improvement_status():
    """Get current status of self-improvement system"""
    try:
        report = await self_improvement_service.get_improvement_report()
        
        return {
            "active": self_improvement_service._start_monitoring,
            "report": report,
            "current_metrics": len(self_improvement_service.metrics_buffer),
            "improvement_history_size": len(self_improvement_service.improvement_history)
        }
    except Exception as e:
        logger.error(f"Error getting self-improvement status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_improvement_report():
    """Get detailed improvement report"""
    try:
        report = await self_improvement_service.get_improvement_report()
        return report
    except Exception as e:
        logger.error(f"Error getting improvement report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-file")
async def analyze_specific_file(file_path: str):
    """Analyze a specific file for improvements"""
    try:
        from pathlib import Path
        
        # Read file content
        file = Path(file_path)
        if not file.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(file, 'r') as f:
            content = f.read()
        
        # Analyze file
        improvements = await self_improvement_service._analyze_file_content(file, content)
        
        return {
            "file": file_path,
            "improvements_found": len(improvements),
            "improvements": [
                {
                    "line_start": imp.line_start,
                    "line_end": imp.line_end,
                    "issue_type": imp.issue_type,
                    "current_code": imp.current_code[:100] + "..." if len(imp.current_code) > 100 else imp.current_code,
                    "suggested_code": imp.suggested_code[:100] + "..." if len(imp.suggested_code) > 100 else imp.suggested_code,
                    "explanation": imp.explanation
                }
                for imp in improvements
            ]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/add")
async def add_performance_metric(
    metric_type: str,
    value: float,
    context: Dict[str, Any] = None
):
    """Manually add a performance metric"""
    try:
        from datetime import datetime
        from app.services.self_improvement import PerformanceMetric
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            context=context or {}
        )
        
        self_improvement_service.metrics_buffer.append(metric)
        
        # Trigger analysis if buffer is full
        if len(self_improvement_service.metrics_buffer) >= 100:
            await self_improvement_service._analyze_metrics()
        
        return {
            "status": "added",
            "buffer_size": len(self_improvement_service.metrics_buffer)
        }
    except Exception as e:
        logger.error(f"Error adding metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions/pending")
async def get_pending_suggestions():
    """Get pending improvement suggestions"""
    try:
        # Query vector DB for pending improvements
        suggestions = await self_improvement_service.vector_db.search(
            collection_name="improvements",
            query="pending",
            k=20
        )
        
        return {
            "count": len(suggestions),
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Error getting pending suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggestions/{suggestion_id}/apply")
async def apply_suggestion(suggestion_id: str, background_tasks: BackgroundTasks):
    """Apply a specific improvement suggestion"""
    try:
        # Get suggestion from vector DB
        suggestion_doc = await self_improvement_service.vector_db.get_document(
            collection_name="improvements",
            document_id=suggestion_id
        )
        
        if not suggestion_doc:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        # Parse and process suggestion
        import json
        from app.services.self_improvement import ImprovementSuggestion
        
        suggestion_data = json.loads(suggestion_doc['content'])
        suggestion = ImprovementSuggestion(**suggestion_data)
        
        # Process in background
        background_tasks.add_task(
            self_improvement_service._process_improvement,
            suggestion
        )
        
        return {
            "status": "processing",
            "suggestion": suggestion_data
        }
    except Exception as e:
        logger.error(f"Error applying suggestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_self_improvement_config():
    """Get self-improvement configuration"""
    return {
        "batch_size": self_improvement_service.batch_size,
        "min_confidence_threshold": self_improvement_service.min_confidence_threshold,
        "max_concurrent_improvements": self_improvement_service.max_concurrent_improvements,
        "code_patterns": list(self_improvement_service.code_patterns.keys()),
        "issue_patterns": list(self_improvement_service.issue_patterns.keys())
    }


@router.put("/config")
async def update_self_improvement_config(
    batch_size: int = None,
    min_confidence_threshold: float = None,
    max_concurrent_improvements: int = None
):
    """Update self-improvement configuration"""
    try:
        if batch_size is not None:
            self_improvement_service.batch_size = batch_size
        
        if min_confidence_threshold is not None:
            self_improvement_service.min_confidence_threshold = min_confidence_threshold
        
        if max_concurrent_improvements is not None:
            self_improvement_service.max_concurrent_improvements = max_concurrent_improvements
        
        return await get_self_improvement_config()
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))