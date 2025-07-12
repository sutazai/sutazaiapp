from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import tempfile
import os
from pathlib import Path

from tools import code_analyzer
from api.auth import get_current_user, require_admin
from api.database import db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/code/snippet")
async def analyze_code_snippet(
    code_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Analyze a code snippet for issues, metrics, and suggestions."""
    try:
        code = code_data.get("code", "")
        language = code_data.get("language", "python")
        analysis_types = code_data.get("analysis_types", ["security", "quality", "style"])
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Perform comprehensive analysis
        analysis_result = await code_analyzer.analyze_code(
            code=code,
            language=language,
            analysis_types=analysis_types
        )
        
        await db_manager.log_system_event(
            "info", "analysis", "Code snippet analyzed",
            {"user": current_user.get("username"), "language": language, "code_length": len(code)}
        )
        
        return {
            "analysis": analysis_result,
            "language": language,
            "code_length": len(code),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing code snippet: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/code/file")
async def analyze_code_file(
    file: UploadFile = File(...),
    analysis_types: List[str] = ["security", "quality", "style"],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Analyze an uploaded code file."""
    try:
        # Validate file type
        allowed_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported for code analysis"
            )
        
        # Read file content
        content = await file.read()
        code = content.decode('utf-8')
        
        # Determine language from extension
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust'
        }
        language = language_map.get(file_extension, 'unknown')
        
        # Perform analysis
        analysis_result = await code_analyzer.analyze_code(
            code=code,
            language=language,
            analysis_types=analysis_types,
            filename=file.filename
        )
        
        await db_manager.log_system_event(
            "info", "analysis", "Code file analyzed",
            {"user": current_user.get("username"), "filename": file.filename, "language": language}
        )
        
        return {
            "filename": file.filename,
            "language": language,
            "analysis": analysis_result,
            "file_size": len(code),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing code file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security/scan")
async def security_scan(
    scan_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Perform security-focused code analysis."""
    try:
        code = scan_data.get("code", "")
        language = scan_data.get("language", "python")
        scan_level = scan_data.get("level", "standard")  # basic, standard, comprehensive
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Perform security analysis
        security_result = await code_analyzer.security_scan(
            code=code,
            language=language,
            level=scan_level
        )
        
        await db_manager.log_system_event(
            "info", "analysis", "Security scan performed",
            {"user": current_user.get("username"), "language": language, "level": scan_level}
        )
        
        return {
            "security_analysis": security_result,
            "language": language,
            "scan_level": scan_level,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing security scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quality/metrics")
async def quality_metrics(
    metrics_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Calculate code quality metrics."""
    try:
        code = metrics_data.get("code", "")
        language = metrics_data.get("language", "python")
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Calculate metrics
        quality_result = await code_analyzer.calculate_quality_metrics(
            code=code,
            language=language
        )
        
        await db_manager.log_system_event(
            "info", "analysis", "Quality metrics calculated",
            {"user": current_user.get("username"), "language": language}
        )
        
        return {
            "quality_metrics": quality_result,
            "language": language,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/style/check")
async def style_check(
    style_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Check code style and formatting."""
    try:
        code = style_data.get("code", "")
        language = style_data.get("language", "python")
        style_guide = style_data.get("style_guide", "default")  # pep8, google, etc.
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Perform style analysis
        style_result = await code_analyzer.check_style(
            code=code,
            language=language,
            style_guide=style_guide
        )
        
        await db_manager.log_system_event(
            "info", "analysis", "Style check performed",
            {"user": current_user.get("username"), "language": language, "style_guide": style_guide}
        )
        
        return {
            "style_analysis": style_result,
            "language": language,
            "style_guide": style_guide,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing style check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai/insights")
async def ai_insights(
    insights_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get AI-powered code insights and suggestions."""
    try:
        code = insights_data.get("code", "")
        language = insights_data.get("language", "python")
        focus_areas = insights_data.get("focus_areas", ["optimization", "readability", "bugs"])
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Get AI insights
        ai_result = await code_analyzer.get_ai_insights(
            code=code,
            language=language,
            focus_areas=focus_areas
        )
        
        await db_manager.log_system_event(
            "info", "analysis", "AI insights generated",
            {"user": current_user.get("username"), "language": language, "focus_areas": focus_areas}
        )
        
        return {
            "ai_insights": ai_result,
            "language": language,
            "focus_areas": focus_areas,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating AI insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-languages")
async def get_supported_languages(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get list of supported programming languages for analysis."""
    try:
        supported_languages = await code_analyzer.get_supported_languages()
        
        return {
            "supported_languages": supported_languages,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis-types")
async def get_analysis_types(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get available analysis types and their descriptions."""
    try:
        analysis_types = {
            "security": {
                "name": "Security Analysis",
                "description": "Scans for security vulnerabilities and potential exploits",
                "tools": ["semgrep", "bandit", "custom_rules"]
            },
            "quality": {
                "name": "Quality Metrics",
                "description": "Calculates code complexity, maintainability, and quality scores",
                "metrics": ["cyclomatic_complexity", "maintainability_index", "technical_debt"]
            },
            "style": {
                "name": "Style Analysis",
                "description": "Checks code formatting and style guide compliance",
                "guides": ["pep8", "google", "airbnb", "standard"]
            },
            "ai_insights": {
                "name": "AI Insights",
                "description": "AI-powered suggestions for optimization and improvements",
                "focus_areas": ["optimization", "readability", "bugs", "patterns"]
            }
        }
        
        return {
            "analysis_types": analysis_types,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analysis types: {e}")
        raise HTTPException(status_code=500, detail=str(e))
