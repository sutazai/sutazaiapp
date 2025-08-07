"""
AutoGPT Service for SutazAI
Provides autonomous AI agent capabilities
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
import subprocess
import json
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI AutoGPT Service",
    description="Autonomous AI agent service",
    version="1.0.0"
)

# Request/Response models
class AutoGPTRequest(BaseModel):
    task: str
    max_iterations: Optional[int] = 10
    workspace: Optional[str] = "/app/workspace"

class AutoGPTResponse(BaseModel):
    status: str
    result: Any
    workspace_files: Optional[list] = None

# AutoGPT Manager
class AutoGPTManager:
    def __init__(self):
        self.workspace = "/app/workspace"
        self.config = {
            "ai_model": "tinyllama.2:1b",
            "api_base": "http://ollama:10104/v1",
            "api_key": "local"
        }
        
    def create_ai_settings(self, task: str) -> str:
        """Create AI settings file for the task"""
        settings = {
            "ai_name": "SutazAI-AutoGPT",
            "ai_role": "Autonomous AI Assistant",
            "ai_goals": [
                task,
                "Complete the task efficiently and safely",
                "Provide clear documentation of actions taken"
            ],
            "api_budget": 10.0,
            "model": self.config["ai_model"]
        }
        
        settings_path = "/app/ai_settings.yaml"
        with open(settings_path, 'w') as f:
            import yaml
            yaml.dump(settings, f)
            
        return settings_path
    
    def execute_task(self, request: AutoGPTRequest) -> Dict[str, Any]:
        """Execute AutoGPT task"""
        try:
            # Create AI settings
            settings_path = self.create_ai_settings(request.task)
            
            # Set environment variables for local Ollama
            env = os.environ.copy()
            env.update({
                "OPENAI_API_BASE": "http://ollama:10104/v1",
                "OLLAMA_HOST": "http://ollama:10104,  # Ollama doesn't need real API key
                "AUTOGPT_WORKSPACE": request.workspace,
                "LLM_MODEL": "tinyllama.2:1b"
            })
            
            # Change to autogpt classic directory
            os.chdir('/app/autogpt_repo/classic/original_autogpt')
            
            # Execute AutoGPT command using the correct module path
            cmd = [
                "python", "-m", "autogpt",
                "--continuous-mode",
                f"--continuous-limit={request.max_iterations}"
            ]
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Change back to app directory
            os.chdir('/app')
            
            # Get workspace files
            workspace_files = []
            if os.path.exists(request.workspace):
                workspace_files = os.listdir(request.workspace)
            
            return {
                "status": "success" if result.returncode == 0 else "partial",
                "result": {
                    "output": result.stdout,
                    "errors": result.stderr,
                    "return_code": result.returncode
                },
                "workspace_files": workspace_files
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "result": "Task execution timed out after 5 minutes",
                "workspace_files": []
            }
        except Exception as e:
            logger.error(f"AutoGPT execution failed: {e}")
            return {
                "status": "failed",
                "result": str(e),
                "workspace_files": []
            }

# Initialize AutoGPT manager
autogpt_manager = AutoGPTManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AutoGPT",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "autogpt"}

@app.post("/execute", response_model=AutoGPTResponse)
async def execute_task(request: AutoGPTRequest):
    """Execute an AutoGPT task"""
    try:
        result = autogpt_manager.execute_task(request)
        return AutoGPTResponse(**result)
    except Exception as e:
        logger.error(f"Failed to execute AutoGPT task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspace/{file_path:path}")
async def get_workspace_file(file_path: str):
    """Get a file from the workspace"""
    full_path = os.path.join("/app/workspace", file_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if os.path.isfile(full_path):
        with open(full_path, 'r') as f:
            return {"content": f.read(), "type": "file"}
    else:
        return {"content": os.listdir(full_path), "type": "directory"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)