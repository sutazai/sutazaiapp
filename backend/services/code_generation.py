"""
SutazAI Code Generation Service
Handles code generation requests using local LLMs
"""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

from backend.core.config import get_settings
from backend.core.database import get_db
from backend.models.base_models import CodeGeneration
from backend.services.code_generation.code_generator import CodeGenerator

# Initialize router
router = APIRouter()
settings = get_settings()
logger = logging.getLogger("code_generation")

# Initialize the code generator (lazy loading)
_code_generator = None


def get_code_generator():
    """Get or initialize the code generator."""
    global _code_generator
    if _code_generator is None:
        try:
            _code_generator = CodeGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize code generator: {str(e)}")
            raise RuntimeError(f"Failed to initialize code generator: {str(e)}")
    return _code_generator


# Define data models
class CodeGenerationRequest(BaseModel):
    """Model for code generation requests"""

    spec_text: str = Field(
        ..., description="Specification text describing the code to generate"
    )
    language: str = Field(
        "python", description="Programming language for code generation"
    )

    class Config:
        schema_extra = {
            "example": {
                "spec_text": "Create a function that calculates the Fibonacci sequence up to n terms",
                "language": "python",
            }
        }


class CodeGenerationResponse(BaseModel):
    """Model for code generation responses"""

    id: int = Field(..., description="ID of the generated code in the database")
    language: str = Field(..., description="Programming language of the generated code")
    generated_code: str = Field(..., description="Generated code")
    issues: List[str] = Field(
        [], description="List of potential issues in the generated code"
    )
    generation_time_ms: int = Field(
        ..., description="Time taken to generate the code in milliseconds"
    )

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "language": "python",
                "generated_code": "def fibonacci(n):\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result",
                "issues": [],
                "generation_time_ms": 1250,
            }
        }


class CodeImprovementRequest(BaseModel):
    code: str = Field(..., description="Original code to improve")
    issues: List[str] = Field([], description="List of issues to fix")
    language: str = Field("python", description="Programming language of the code")

    class Config:
        schema_extra = {
            "example": {
                "code": "def fibonacci(n):\n    result = []\n    a, b = 0, 1\n    for i in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result",
                "issues": ["Function lacks docstring", "Variable 'i' is not used"],
                "language": "python",
            }
        }


# API Routes
@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    code_generator: CodeGenerator = Depends(get_code_generator),
):
    """
    Generate code based on a specification.
    """
    try:
        # Create a database record
        code_generation = CodeGeneration(
            prompt=request.spec_text,
            language=request.language,
            status="processing",
            model=settings.CODE_GENERATION_MODEL,
        )

        db.add(code_generation)
        db.commit()
        db.refresh(code_generation)

        # Generate code
        result = code_generator.generate_code(
            spec_text=request.spec_text, language=request.language
        )

        # Update database record
        code_generation.generated_code = result["generated_code"]
        code_generation.status = "completed"
        code_generation.metadata = {
            "issues": result["issues"],
            "generation_time_ms": result["generation_time_ms"],
        }

        db.commit()

        return {
            "id": code_generation.id,
            "language": result["language"],
            "generated_code": result["generated_code"],
            "issues": result["issues"],
            "generation_time_ms": result["generation_time_ms"],
        }

    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")

        # Update database record if it was created
        if "code_generation" in locals() and code_generation.id:
            code_generation.status = "failed"
            code_generation.error = str(e)
            db.commit()

        raise HTTPException(
            status_code=500, detail=f"Failed to generate code: {str(e)}"
        )


@router.post("/improve")
async def improve_code(
    request: CodeImprovementRequest,
    code_generator: CodeGenerator = Depends(get_code_generator),
):
    """
    Improve code based on identified issues.
    """
    try:
        result = code_generator.improve_code(
            code=request.code, issues=request.issues, language=request.language
        )

        return result

    except Exception as e:
        logger.error(f"Error improving code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to improve code: {str(e)}")


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_code_generation_history(
    limit: Optional[int] = 10, skip: Optional[int] = 0, db: Session = Depends(get_db)
):
    """
    Get the history of code generation requests.
    """
    code_generations = (
        db.query(CodeGeneration)
        .order_by(CodeGeneration.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        {
            "id": cg.id,
            "prompt": cg.prompt,
            "language": cg.language,
            "status": cg.status,
            "model": cg.model,
            "created_at": cg.created_at,
            "code_snippet": cg.generated_code[:100] + "..."
            if cg.generated_code and len(cg.generated_code) > 100
            else cg.generated_code,
        }
        for cg in code_generations
    ]


@router.get("/{code_generation_id}")
async def get_code_generation(code_generation_id: int, db: Session = Depends(get_db)):
    """
    Get a specific code generation by ID.
    """
    code_generation = (
        db.query(CodeGeneration).filter(CodeGeneration.id == code_generation_id).first()
    )

    if not code_generation:
        raise HTTPException(
            status_code=404,
            detail=f"Code generation with ID {code_generation_id} not found",
        )

    return {
        "id": code_generation.id,
        "prompt": code_generation.prompt,
        "language": code_generation.language,
        "generated_code": code_generation.generated_code,
        "status": code_generation.status,
        "model": code_generation.model,
        "created_at": code_generation.created_at,
        "updated_at": code_generation.updated_at,
        "metadata": code_generation.metadata,
        "error": code_generation.error,
    }


@router.get("/languages", status_code=status.HTTP_200_OK)
async def get_supported_languages():
    """
    Get list of supported programming languages for code generation
    """
    # This would be dynamically determined based on model capabilities
    # For now, we'll hardcode a list of commonly supported languages
    languages = [
        {
            "id": "python",
            "name": "Python",
            "version": "3.x",
            "supported_features": ["code_generation", "code_improvement"],
        },
        {
            "id": "javascript",
            "name": "JavaScript",
            "version": "ES6+",
            "supported_features": ["code_generation"],
        },
        {
            "id": "typescript",
            "name": "TypeScript",
            "version": "4.x",
            "supported_features": ["code_generation"],
        },
        {
            "id": "java",
            "name": "Java",
            "version": "11+",
            "supported_features": ["code_generation"],
        },
        {
            "id": "csharp",
            "name": "C#",
            "version": "9.0+",
            "supported_features": ["code_generation"],
        },
        {
            "id": "cpp",
            "name": "C++",
            "version": "17+",
            "supported_features": ["code_generation"],
        },
        {
            "id": "go",
            "name": "Go",
            "version": "1.16+",
            "supported_features": ["code_generation"],
        },
        {
            "id": "rust",
            "name": "Rust",
            "version": "1.50+",
            "supported_features": ["code_generation"],
        },
        {
            "id": "php",
            "name": "PHP",
            "version": "8.x",
            "supported_features": ["code_generation"],
        },
        {
            "id": "ruby",
            "name": "Ruby",
            "version": "3.x",
            "supported_features": ["code_generation"],
        },
    ]

    return {"languages": languages, "count": len(languages)}
