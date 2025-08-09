import os
from fastapi import FastAPI

app = FastAPI(title="Jarvis Automation Agent", version="1.0.0")

@app.get("/")
def root():
    return {"service": "jarvis-automation-agent", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port)
