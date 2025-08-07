from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Jarvis Knowledge Management", version="0.1.0")

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

