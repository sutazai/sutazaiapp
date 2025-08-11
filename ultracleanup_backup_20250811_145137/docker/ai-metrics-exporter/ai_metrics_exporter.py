from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Metrics Exporter", version="1.0.0")

class Health(BaseModel):
    status: str

@app.get("/health", response_model=Health)
def health():
    return Health(status="healthy")

@app.get("/metrics")
def metrics():
    # Minimal placeholder metrics; replace with real Prometheus output if needed
    return {"uptime_seconds": 0, "requests_total": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9200)
