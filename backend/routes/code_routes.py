from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from backend.services.auth import validate_otp  # From Phase 3 OTP validation
from backend.services.code_generation import CodeGenerationService

router = APIRouter(prefix="/code", tags=["Code Generation"])

# Initialize code generation service
code_gen_service = CodeGenerationService()


class CodeGenerationRequest(BaseModel):
    """
    Request model for code generation
    """

    specification: str = Field(..., min_length=10, max_length=2000)
    model_name: Optional[str] = Field(default="deepseek-coder")
    language: Optional[str] = Field(default="python")
    otp: str = Field(..., min_length=6, max_length=6)


class CodeGenerationResponse(BaseModel):
    """
    Response model for code generation
    """

    success: bool
    generated_code: Optional[str] = None
    security_warnings: list = []
    error: Optional[str] = None


@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate code from specification with security scanning

    Args:
        request (CodeGenerationRequest): Code generation parameters

    Returns:
        CodeGenerationResponse with generated code or error
    """
    # Validate OTP first
    if not validate_otp(request.otp):
        raise HTTPException(status_code=403, detail="Invalid OTP")

    try:
        # Validate model availability
        if request.model_name not in code_gen_service.available_models:
            raise ValueError(f"Model {request.model_name} not available")

        # Generate code
        result = code_gen_service.generate_code(
            specification=request.specification, model_name=request.model_name, language=request.language,
        )

        # Check for generation errors
        if result.get("error"):
            return CodeGenerationResponse(success=False, error=result["error"])

        return CodeGenerationResponse(
            success=True, generated_code=result["code"], security_warnings=result["security_warnings"],
        )

    except Exception as e:
        return CodeGenerationResponse(success=False, error=str(e))


def setup_routes(app):
    """
    Setup code generation routes

    Args:
        app (FastAPI): FastAPI application instance
    """
    app.include_router(router)
