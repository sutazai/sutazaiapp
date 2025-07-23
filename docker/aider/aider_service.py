"""
Aider Service for SutazAI
Provides AI-powered code assistance and editing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import subprocess
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI Aider Service",
    description="AI-powered code assistance service",
    version="1.0.0"
)

# Request/Response models
class AiderRequest(BaseModel):
    message: str
    files: Optional[List[str]] = []
    model: Optional[str] = "deepseek-r1:8b"
    workspace: Optional[str] = "/app/workspace"

class AiderResponse(BaseModel):
    status: str
    result: Any
    modified_files: Optional[List[str]] = None

# Aider Manager
class AiderManager:
    def __init__(self):
        self.workspace = "/app/workspace"
        self.config = {
            "model": "ollama/llama3.2:1b",
            "api_base": "http://ollama:11434",
            "api_key": "local"
        }
        
        # Ensure workspace exists
        os.makedirs(self.workspace, exist_ok=True)
        
    def execute_aider(self, request: AiderRequest) -> Dict[str, Any]:
        """Execute Aider command"""
        try:
            # Change to workspace directory
            original_cwd = os.getcwd()
            os.chdir(request.workspace)
            
            # Set environment variables
            env = os.environ.copy()
            env.update({
                "OPENAI_API_BASE": self.config["api_base"],
                "OPENAI_API_KEY": self.config["api_key"],
                "OLLAMA_API_BASE": self.config["api_base"],
                "OLLAMA_API_KEY": self.config["api_key"]
            })
            
            # Build Aider command
            cmd = [
                "aider",
                "--model", f"ollama/{request.model or 'llama3.2:1b'}",
                "--openai-api-base", self.config["api_base"] + "/v1",
                "--openai-api-key", self.config["api_key"],
                "--message", request.message,
                "--no-git",
                "--yes"  # Auto-confirm changes
            ]
            
            # Add files if specified
            if request.files:
                # Validate files exist or create them
                for file_path in request.files:
                    full_path = os.path.join(request.workspace, file_path)
                    if not os.path.exists(full_path):
                        # Create file with basic content
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, 'w') as f:
                            f.write(f"# {file_path}\n# Created by Aider\n")
                    cmd.append(file_path)
            
            # Execute Aider
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Get modified files
            modified_files = []
            if request.files:
                modified_files = request.files
            
            # Return to original directory
            os.chdir(original_cwd)
            
            return {
                "status": "success" if result.returncode == 0 else "partial",
                "result": {
                    "output": result.stdout,
                    "errors": result.stderr,
                    "return_code": result.returncode
                },
                "modified_files": modified_files
            }
            
        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            return {
                "status": "timeout",
                "result": "Aider execution timed out after 5 minutes",
                "modified_files": []
            }
        except Exception as e:
            os.chdir(original_cwd)
            logger.error(f"Aider execution failed: {e}")
            return {
                "status": "failed",
                "result": str(e),
                "modified_files": []
            }

# Initialize Aider manager
aider_manager = AiderManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Aider",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "aider"}

@app.post("/code", response_model=AiderResponse)
async def code_assistance(request: AiderRequest):
    """Get AI-powered code assistance"""
    try:
        result = aider_manager.execute_aider(request)
        return AiderResponse(**result)
    except Exception as e:
        logger.error(f"Failed to execute Aider: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspace")
async def list_workspace():
    """List files in workspace"""
    try:
        files = []
        for root, dirs, filenames in os.walk("/app/workspace"):
            for filename in filenames:
                relative_path = os.path.relpath(os.path.join(root, filename), "/app/workspace")
                files.append(relative_path)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspace/{file_path:path}")
async def get_file_content(file_path: str):
    """Get content of a workspace file"""
    full_path = os.path.join("/app/workspace", file_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return {"content": f.read(), "path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-file")
async def create_file(file_path: str, content: str = ""):
    """Create a new file in workspace"""
    full_path = os.path.join("/app/workspace", file_path)
    
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"message": f"File {file_path} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)