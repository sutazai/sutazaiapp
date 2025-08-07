from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Jarvis Hardware Resource Optimizer", version="0.1.0")

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

