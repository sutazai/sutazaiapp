from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from backend.services.auth import validate_otp  # From Phase 3 OTP validation
from backend.services.code_generation import CodeGenerationService

# Initialize FastAPI router
router = APIRouter(prefix="/code", tags=["Code Generation"])

# Initialize code generation service
code_gen_service = CodeGenerationService()


class CodeGenerationRequest(BaseModel):
    """Request model for code generation."""

    specification: str = Field(
    ..., min_length=10, max_length=2000, description="Code generation specification",
    )
    model_name: Optional[str] = Field(
    default="deepseek-coder", description="Name of the code generation model to use",
    )
    language: Optional[str] = Field(
    default="python", description="Target programming language",
    )
    otp: str = Field(
    ...,
    min_length=6,
    max_length=6,
    description="One-time password for authentication",
    )


    class CodeGenerationResponse(BaseModel):
        """Response model for code generation."""

        success: bool = Field(description="Whether the code generation was successful")
        generated_code: Optional[str] = Field(
        default=None, description="Generated code if successful",
        )
        security_warnings: List[str] = Field(
        default_factory=list,
        description="List of security warnings for the generated code",
        )
        error: Optional[str] = Field(
        default=None, description="Error message if code generation failed",
        )


        @router.post("/generate", response_model=CodeGenerationResponse)
        async def generate_code(
        request: CodeGenerationRequest, background_tasks: BackgroundTasks,
        ) -> CodeGenerationResponse:
        """
        Generate code from specification with security scanning.

        Args:
        request: Code generation parameters
        background_tasks: Background tasks runner

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
                    specification=request.specification,
                    model_name=request.model_name,
                    language=request.language,
                    )

                    # Check for generation errors
                    if result.get("error"):
                        return CodeGenerationResponse(success=False, error=result["error"])

                    return CodeGenerationResponse(
                success=True,
                generated_code=result["code"],
                security_warnings=result.get("security_warnings", []),
                )

                except Exception as e:
                    return CodeGenerationResponse(success=False, error=str(e))


                def setup_routes(app) -> None:
                    """
                    Setup code generation routes.

                    Args:
                    app: FastAPI application instance
                    """
                    app.include_router(router)
