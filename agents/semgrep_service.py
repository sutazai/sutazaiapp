from fastapi import FastAPI, HTTPException
import subprocess
import tempfile
import json

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "semgrep"}

@app.post("/scan")
async def scan_code(request: dict):
    code = request.get("code", "")
    language = request.get("language", "python")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
        f.write(code)
        f.flush()
        
        try:
            result = subprocess.run(
                ["semgrep", "--json", "--config=auto", f.name],
                capture_output=True,
                text=True
            )
            
            return {
                "findings": json.loads(result.stdout) if result.stdout else {},
                "errors": result.stderr
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
