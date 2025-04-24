from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
import logging
from typing import List, Optional
# Import the service and getter
from ..services.code_service import CodeService, get_code_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Adjusted Request Models to match Service expectations

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Detailed natural language prompt describing the desired software.")
    project_name: str = Field(..., description="Unique name for the project (used as directory name).")

class EditRequest(BaseModel):
    files: List[str] = Field(..., description="List of file paths (relative to workspace/repo) to edit.")
    instruction: str = Field(..., description="Natural language instruction for the edit.")
    repo_path: Optional[str] = Field(None, description="Optional path to the git repository root relative to workspace.")

# Define Response Models (matching service/integration outputs)
class CodeGenerationResponse(BaseModel):
    status: str
    message: str
    output_path: Optional[str] = None
    log: Optional[str] = None

class CodeEditResponse(BaseModel):
    status: str
    message: str
    output: Optional[str] = None # Aider output/diff
    log: Optional[str] = None


@router.post("/generate", 
              response_model=CodeGenerationResponse, 
              summary="Generate code based on a prompt using GPT-Engineer")
async def generate_code(
    request: GenerateRequest = Body(...),
    service: CodeService = Depends(get_code_service) # Inject service
):
    """
    Receives a prompt and generates code, potentially an entire project structure,
    by calling the CodeService.
    """
    logger.info(f"Received code generation request for project: {request.project_name}")
    try:
        result = await service.generate_codebase(request.prompt, request.project_name)
        if result.get("status") == "error":
            # Use 500 for internal errors, maybe 400 if input was bad (though service might handle that)
            raise HTTPException(status_code=500, detail=result.get("message", "Code generation failed."))
        return CodeGenerationResponse(**result) # Return validated response
    except HTTPException as http_exc:
         raise http_exc # Re-raise validation errors
    except Exception as e:
        logger.error(f"Error during code generation endpoint processing for '{request.project_name}': {e}", exc_info=True)
        # Catch potential unexpected errors even before/after service call
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/edit", 
              response_model=CodeEditResponse, 
              summary="Edit code based on a prompt using Aider")
async def edit_code(
    request: EditRequest = Body(...),
    service: CodeService = Depends(get_code_service) # Inject service
):
    """
    Receives an instruction and edits specified files within the workspace
    by calling the CodeService.
    """
    logger.info(f"Received code editing request for files {request.files}")
    try:
        result = await service.edit_code_files(
            files=request.files, 
            instruction=request.instruction, 
            repo_path=request.repo_path
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Code editing failed."))
        return CodeEditResponse(**result) # Return validated response
    except HTTPException as http_exc:
         raise http_exc # Re-raise validation errors
    except Exception as e:
        logger.error(f"Error during code edit endpoint processing for files '{request.files}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") 