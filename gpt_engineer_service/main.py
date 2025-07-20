# GPT-Engineer Service Application
# --------------------------------

import asyncio
import time
import logging
import tempfile
import shutil
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPT-Engineer Service", version="1.0.0")

class CodeRequest(BaseModel):
    prompt: str
    project_name: str
    language: Optional[str] = "python"
    framework: Optional[str] = None

class GPTEngineerService:
    def __init__(self):
        self.workspace_dir = "/tmp/gpt_engineer_workspace"
        os.makedirs(self.workspace_dir, exist_ok=True)
        logger.info("GPT-Engineer service initialized")
    
    async def generate_code(self, request: CodeRequest) -> Dict[str, Any]:
        """Generate code using GPT-Engineer"""
        try:
            # Create project directory
            project_dir = os.path.join(self.workspace_dir, request.project_name)
            os.makedirs(project_dir, exist_ok=True)
            
            # Create prompt file
            prompt_file = os.path.join(project_dir, "prompt.txt")
            with open(prompt_file, "w") as f:
                f.write(request.prompt)
            
            # Run GPT-Engineer in non-interactive mode
            cmd = [
                "gpt-engineer",
                project_dir,
                "--steps", "gen_entrypoint",
                "--no-execution"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_dir
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Read generated files
                generated_files = self.read_generated_files(project_dir)
                
                return {
                    "status": "success",
                    "project_name": request.project_name,
                    "files": generated_files,
                    "output": stdout.decode()
                }
            else:
                return {
                    "status": "error",
                    "error": stderr.decode(),
                    "project_name": request.project_name
                }
                
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "project_name": request.project_name
            }
    
    def read_generated_files(self, project_dir: str) -> Dict[str, str]:
        """Read all generated files from project directory"""
        files = {}
        
        for root, dirs, filenames in os.walk(project_dir):
            for filename in filenames:
                if filename.endswith(('.py', '.js', '.html', '.css', '.md', '.txt', '.json')):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, project_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            files[relative_path] = f.read()
                    except Exception as e:
                        files[relative_path] = f"Error reading file: {e}"
        
        return files
    
    async def validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """Validate generated code"""
        try:
            if language == "python":
                # Basic Python syntax check
                compile(code, '<string>', 'exec')
                return {"valid": True, "message": "Code is syntactically valid"}
            else:
                return {"valid": True, "message": "Validation not implemented for this language"}
                
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno
            }

# Initialize service
gpt_engineer_service = GPTEngineerService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "gpt-engineer"}

@app.post("/generate")
async def generate_code(request: CodeRequest):
    """Generate code using GPT-Engineer"""
    result = await gpt_engineer_service.generate_code(request)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/validate")
async def validate_code(code: str, language: str = "python"):
    """Validate code syntax"""
    result = await gpt_engineer_service.validate_code(code, language)
    return result

@app.get("/projects")
async def list_projects():
    """List all generated projects"""
    try:
        projects = []
        workspace_dir = gpt_engineer_service.workspace_dir
        
        if os.path.exists(workspace_dir):
            for item in os.listdir(workspace_dir):
                item_path = os.path.join(workspace_dir, item)
                if os.path.isdir(item_path):
                    projects.append({
                        "name": item,
                        "created": os.path.getctime(item_path),
                        "modified": os.path.getmtime(item_path)
                    })
        
        return {"projects": projects}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting GPT-Engineer service...")
    uvicorn.run(app, host="0.0.0.0", port=8080)