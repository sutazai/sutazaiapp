from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

app = FastAPI(title="SutazAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SutazAI Backend v9.0.0", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/api/system/status")
async def system_status():
    return {
        "status": "online",
        "uptime": 100.0,
        "active_agents": 3,
        "loaded_models": 5,
        "requests_count": 42
    }

@app.get("/api/agents/")
async def get_agents():
    return [
        {"id": "1", "name": "DeepSeek-Coder", "type": "coding", "status": "active"},
        {"id": "2", "name": "Qwen3", "type": "general", "status": "active"},
        {"id": "3", "name": "AutoGPT", "type": "automation", "status": "idle"}
    ]

@app.post("/api/agents/")
async def create_agent(data: dict):
    return {"id": "new", "name": data.get("name", "New Agent"), "status": "active"}

@app.post("/api/agents/{agent_id}/chat")
async def chat(agent_id: str, data: dict):
    message = data.get("message", "")
    return {
        "response": f"Hello! This is a response from agent {agent_id} to your message: {message}",
        "agent_id": agent_id
    }

@app.get("/api/models/")
async def get_models():
    return [
        {"id": "1", "name": "deepseek-r1:8b", "status": "loaded"},
        {"id": "2", "name": "qwen3:8b", "status": "loaded"}
    ]

@app.post("/api/code/generate")
async def generate_code(data: dict):
    prompt = data.get("prompt", "")
    language = data.get("language", "python")
    
    code = f"""# Generated {language} code for: {prompt}
def example_function():
    print("Hello from SutazAI!")
    return "Generated code example"

if __name__ == "__main__":
    result = example_function()
    print(result)
"""
    
    return {"code": code, "language": language}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
