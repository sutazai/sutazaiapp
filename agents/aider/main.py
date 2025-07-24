from fastapi import FastAPI
import uvicorn
import subprocess
import os

app = FastAPI(title="Aider Code Assistant")

@app.post("/code")
async def code_assistance(request: dict):
    prompt = request.get("prompt", "")
    files = request.get("files", [])
    
    # Use aider command line interface
    cmd = ["aider", "--model", "ollama/deepseek-r1:8b"] + files + ["--message", prompt]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return {
            "result": result.stdout,
            "error": result.stderr,
            "status": "completed" if result.returncode == 0 else "error"
        }
    except subprocess.TimeoutExpired:
        return {"error": "Request timed out", "status": "timeout"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "aider"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
