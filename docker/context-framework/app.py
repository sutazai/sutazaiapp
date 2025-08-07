from fastapi import FastAPI

app = FastAPI(title="Context Framework", version="1.0.0")

@app.get("/")
def root():
    return {"service": "context-framework", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8589)
