from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from uuid import uuid4

from tools import code_generator
from api.auth import get_current_user, require_admin
from api.database import db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/code")
async def generate_code(
    generation_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate code based on specifications."""
    try:
        prompt = generation_request.get("prompt", "")
        language = generation_request.get("language", "python")
        style = generation_request.get("style", "default")
        complexity = generation_request.get("complexity", "medium")
        include_tests = generation_request.get("include_tests", False)
        include_docs = generation_request.get("include_docs", True)
        
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Code generation prompt is required")
        
        # Generate code
        generation_id = str(uuid4())
        result = await code_generator.generate_code(
            prompt=prompt,
            language=language,
            style=style,
            complexity=complexity,
            include_tests=include_tests,
            include_docs=include_docs,
            generation_id=generation_id
        )
        
        await db_manager.log_system_event(
            "info", "generation", "Code generated",
            {
                "user": current_user.get("username"),
                "generation_id": generation_id,
                "language": language,
                "prompt_length": len(prompt)
            }
        )
        
        return {
            "generation_id": generation_id,
            "result": result,
            "language": language,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refactor")
async def refactor_code(
    refactor_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Refactor existing code for improvements."""
    try:
        code = refactor_request.get("code", "")
        language = refactor_request.get("language", "python")
        goals = refactor_request.get("goals", ["readability", "performance"])
        preserve_functionality = refactor_request.get("preserve_functionality", True)
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code to refactor is required")
        
        # Refactor code
        refactor_id = str(uuid4())
        result = await code_generator.refactor_code(
            code=code,
            language=language,
            goals=goals,
            preserve_functionality=preserve_functionality,
            refactor_id=refactor_id
        )
        
        await db_manager.log_system_event(
            "info", "generation", "Code refactored",
            {
                "user": current_user.get("username"),
                "refactor_id": refactor_id,
                "language": language,
                "goals": goals
            }
        )
        
        return {
            "refactor_id": refactor_id,
            "result": result,
            "language": language,
            "goals": goals,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refactoring code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/template")
async def generate_from_template(
    template_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Generate code from a template."""
    try:
        template_name = template_request.get("template_name", "")
        parameters = template_request.get("parameters", {})
        language = template_request.get("language", "python")
        customizations = template_request.get("customizations", {})
        
        if not template_name:
            raise HTTPException(status_code=400, detail="Template name is required")
        
        # Generate from template
        generation_id = str(uuid4())
        result = await code_generator.generate_from_template(
            template_name=template_name,
            parameters=parameters,
            language=language,
            customizations=customizations,
            generation_id=generation_id
        )
        
        await db_manager.log_system_event(
            "info", "generation", "Code generated from template",
            {
                "user": current_user.get("username"),
                "generation_id": generation_id,
                "template_name": template_name,
                "language": language
            }
        )
        
        return {
            "generation_id": generation_id,
            "result": result,
            "template_name": template_name,
            "language": language,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating from template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_code(
    optimization_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Optimize code for performance or other criteria."""
    try:
        code = optimization_request.get("code", "")
        language = optimization_request.get("language", "python")
        optimization_type = optimization_request.get("type", "performance")  # performance, memory, readability
        target_metrics = optimization_request.get("target_metrics", {})
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code to optimize is required")
        
        # Optimize code
        optimization_id = str(uuid4())
        result = await code_generator.optimize_code(
            code=code,
            language=language,
            optimization_type=optimization_type,
            target_metrics=target_metrics,
            optimization_id=optimization_id
        )
        
        await db_manager.log_system_event(
            "info", "generation", "Code optimized",
            {
                "user": current_user.get("username"),
                "optimization_id": optimization_id,
                "language": language,
                "type": optimization_type
            }
        )
        
        return {
            "optimization_id": optimization_id,
            "result": result,
            "language": language,
            "optimization_type": optimization_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fix")
async def fix_code(
    fix_request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Fix code issues and bugs."""
    try:
        code = fix_request.get("code", "")
        language = fix_request.get("language", "python")
        error_message = fix_request.get("error_message", "")
        issue_description = fix_request.get("issue_description", "")
        fix_approach = fix_request.get("approach", "conservative")  # conservative, aggressive
        
        if not code.strip():
            raise HTTPException(status_code=400, detail="Code to fix is required")
        
        # Fix code
        fix_id = str(uuid4())
        result = await code_generator.fix_code(
            code=code,
            language=language,
            error_message=error_message,
            issue_description=issue_description,
            fix_approach=fix_approach,
            fix_id=fix_id
        )
        
        await db_manager.log_system_event(
            "info", "generation", "Code fixed",
            {
                "user": current_user.get("username"),
                "fix_id": fix_id,
                "language": language,
                "approach": fix_approach
            }
        )
        
        return {
            "fix_id": fix_id,
            "result": result,
            "language": language,
            "fix_approach": fix_approach,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fixing code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def list_templates(
    language: Optional[str] = None,
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """List available code generation templates."""
    try:
        templates = await code_generator.list_templates(
            language=language,
            category=category
        )
        
        return {
            "templates": templates,
            "filters": {
                "language": language,
                "category": category
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/{template_name}")
async def get_template(
    template_name: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get details about a specific template."""
    try:
        template = await code_generator.get_template(template_name)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
        
        return {
            "template": template,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template {template_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/styles")
async def get_code_styles(
    language: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get available code styles for generation."""
    try:
        styles = await code_generator.get_code_styles(language=language)
        
        return {
            "styles": styles,
            "language": language,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting code styles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/languages")
async def get_supported_languages(
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get supported programming languages for code generation."""
    try:
        languages = await code_generator.get_supported_languages()
        
        return {
            "languages": languages,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{generation_id}")
async def get_generation_history(
    generation_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get history of a specific code generation."""
    try:
        history = await code_generator.get_generation_history(generation_id)
        
        if not history:
            raise HTTPException(status_code=404, detail=f"Generation {generation_id} not found")
        
        return {
            "generation_id": generation_id,
            "history": history,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting generation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
