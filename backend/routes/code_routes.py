#!/usr/bin/env python3.11
"""Code Routes Module

This module provides routes for code generation functionality.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from backend.services.code_generation import CodeGenerationService
from typing import Optional

# Initialize FastAPI router
router = APIRouter(prefix="/code", tags=["Code Generation"])

# Initialize code generation service
code_gen_service = CodeGenerationService()


class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""
    specification: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Code generation specification",
    )
    model_name: str | None = Field(
        default="deepseek-coder",
        description="Name of the code generation model to use",
    )
    language: str | None = Field(
        default="python",
        description="Target programming language",
    )
    otp: str = Field(
        ...,
        min_length=6,
        max_length=6,
        description="One-time password for authentication",
    )


class CodeGenerationResponse(BaseModel):
    """Response model for code generation."""
    success: bool = Field(description="Whether code generation was successful")
    message: str = Field(description="Status message or error description")
    code: str | None = Field(None, description="Generated code")
    language: str = Field(description="Programming language of generated code")
    security_warnings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Security warnings from code analysis",
    )


@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest, background_tasks: BackgroundTasks
) -> CodeGenerationResponse:
    """Generate code based on a specification.

    Args:
        request: Code generation request
        background_tasks: FastAPI background tasks

    Returns:
        Code generation response with status and generated code
    """
    # Validate OTP
    if not validate_otp(request.otp):
        raise HTTPException(status_code=401, detail="Invalid OTP")

    # Check if model is available
    try:
        # Ensure language is not None before passing to generate_code
        language = request.language or "python"

        # Generate code
        result = code_gen_service.generate_code(
            specification=request.specification,
            model_name=request.model_name or "deepseek-coder",
            language=language,
        )

        return CodeGenerationResponse(
            success=True,
            message="Code generated successfully",
            code=result.get("code", ""),
            language=language,
            security_warnings=result.get("security_warnings", []),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")


def setup_routes(app) -> None:
    """Include router in the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    app.include_router(router)
