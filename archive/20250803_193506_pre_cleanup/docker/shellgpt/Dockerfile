FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    shell-gpt \
    fastapi \
    uvicorn

# Create a simple API wrapper
RUN cat > /app/shellgpt_api.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os

app = FastAPI(title="ShellGPT API")

class ShellRequest(BaseModel):
    command: str
    use_gpt: bool = True

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "shellgpt"}

@app.post("/execute")
async def execute_command(request: ShellRequest):
    try:
        if request.use_gpt:
            # Use shell-gpt to process command
            result = subprocess.run(
                ["sgpt", request.command],
                capture_output=True,
                text=True,
                env={**os.environ, "OPENAI_API_KEY": "sk-local"}
            )
        else:
            # Direct shell execution
            result = subprocess.run(
                request.command,
                shell=True,
                capture_output=True,
                text=True
            )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

EXPOSE 8080

CMD ["python", "/app/shellgpt_api.py"]