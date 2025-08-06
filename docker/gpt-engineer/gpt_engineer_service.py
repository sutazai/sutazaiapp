"""
GPT-Engineer Service for SutazAI
Provides AI-powered software engineering capabilities
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
import subprocess
import tempfile
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI GPT-Engineer Service",
    description="AI-powered software engineering service",
    version="1.0.0"
)

# Request/Response models
class GPTEngineerRequest(BaseModel):
    prompt: str
    project_name: Optional[str] = "sutazai_project"
    workspace: Optional[str] = "/app/workspace"
    model: Optional[str] = "gpt-oss-r1:8b"

class GPTEngineerResponse(BaseModel):
    status: str
    result: Any
    project_files: Optional[List[str]] = None
    project_path: Optional[str] = None

# GPT-Engineer Manager
class GPTEngineerManager:
    def __init__(self):
        self.workspace = "/app/workspace"
        self.config = {
            "model": "gpt-oss-r1:8b",
            "api_base": "http://ollama:11434/v1",
            "api_key": "local"
        }
        
        # Ensure workspace exists
        os.makedirs(self.workspace, exist_ok=True)
        
    def create_project_prompt(self, prompt: str, project_path: str) -> str:
        """Create prompt file for GPT-Engineer"""
        prompt_path = os.path.join(project_path, "prompt")
        
        with open(prompt_path, 'w') as f:
            f.write(prompt)
            
        return prompt_path
    
    def execute_gpt_engineer(self, request: GPTEngineerRequest) -> Dict[str, Any]:
        """Execute GPT-Engineer"""
        try:
            # Create project directory
            project_path = os.path.join(request.workspace, request.project_name)
            os.makedirs(project_path, exist_ok=True)
            
            # Create prompt file
            prompt_path = self.create_project_prompt(request.prompt, project_path)
            
            # Set environment variables for local Ollama
            env = os.environ.copy()
            env.update({
                "OPENAI_API_BASE": "http://ollama:11434/v1",
                "OLLAMA_HOST": "http://ollama:11434,  # Ollama doesn't need real API key
                "OPENAI_API_MODEL": request.model or "gpt-oss-r1:8b"
            })
            
            # Execute GPT-Engineer with local model
            cmd = [
                "gpt-engineer",
                project_path,
                "--model", request.model or "gpt-oss-r1:8b",
                "--temperature", "0.7"
            ]
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for code generation
                cwd=project_path
            )
            
            # Get generated files
            project_files = []
            if os.path.exists(project_path):
                for root, dirs, files in os.walk(project_path):
                    for file in files:
                        if file != "prompt":  # Exclude the prompt file
                            relative_path = os.path.relpath(os.path.join(root, file), project_path)
                            project_files.append(relative_path)
            
            return {
                "status": "success" if result.returncode == 0 else "partial",
                "result": {
                    "output": result.stdout,
                    "errors": result.stderr,
                    "return_code": result.returncode
                },
                "project_files": project_files,
                "project_path": project_path
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "result": "GPT-Engineer execution timed out after 10 minutes",
                "project_files": [],
                "project_path": project_path if 'project_path' in locals() else None
            }
        except Exception as e:
            logger.error(f"GPT-Engineer execution failed: {e}")
            return {
                "status": "failed",
                "result": str(e),
                "project_files": [],
                "project_path": project_path if 'project_path' in locals() else None
            }

# Initialize GPT-Engineer manager
gpt_engineer_manager = GPTEngineerManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "GPT-Engineer",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "gpt_engineer"}

@app.post("/generate", response_model=GPTEngineerResponse)
async def generate_code(request: GPTEngineerRequest):
    """Generate code based on prompt"""
    try:
        result = gpt_engineer_manager.execute_gpt_engineer(request)
        return GPTEngineerResponse(**result)
    except Exception as e:
        logger.error(f"Failed to execute GPT-Engineer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects")
async def list_projects():
    """List all projects in workspace"""
    try:
        projects = []
        workspace_path = "/app/workspace"
        
        if os.path.exists(workspace_path):
            for item in os.listdir(workspace_path):
                item_path = os.path.join(workspace_path, item)
                if os.path.isdir(item_path):
                    # Check if it's a GPT-Engineer project (has a prompt file)
                    prompt_file = os.path.join(item_path, "prompt")
                    if os.path.exists(prompt_file):
                        projects.append({
                            "name": item,
                            "path": item_path,
                            "has_prompt": True
                        })
                    else:
                        projects.append({
                            "name": item,
                            "path": item_path,
                            "has_prompt": False
                        })
        
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/project/{project_name}")
async def get_project_info(project_name: str):
    """Get information about a specific project"""
    project_path = os.path.join("/app/workspace", project_name)
    
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        files = []
        for root, dirs, filenames in os.walk(project_path):
            for filename in filenames:
                relative_path = os.path.relpath(os.path.join(root, filename), project_path)
                files.append(relative_path)
        
        # Get prompt if exists
        prompt_content = ""
        prompt_file = os.path.join(project_path, "prompt")
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompt_content = f.read()
        
        return {
            "name": project_name,
            "path": project_path,
            "files": files,
            "prompt": prompt_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/project/{project_name}/{file_path:path}")
async def get_project_file(project_name: str, file_path: str):
    """Get content of a project file"""
    full_path = os.path.join("/app/workspace", project_name, file_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return {"content": f.read(), "path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)