import os
import asyncio
import httpx
import json
from datetime import datetime

class CodeGenerationImprover:
    def __init__(self):
        self.name = "code-generation-improver"
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
        self.model = os.getenv("MODEL_NAME", "tinyllama")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
        
    async def analyze_code(self, code: str) -> dict:
        """Analyze code and suggest improvements"""
        prompt = f"""Analyze this code and suggest improvements:

```
{code}
```

Provide:
1. Code quality issues
2. Performance optimizations
3. Security concerns
4. Refactored version

Be specific and actionable."""

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": self.max_tokens,
                        "temperature": 0.7,
                        "stream": False
                    }
                )
                result = response.json()
                return {
                    "status": "success",
                    "analysis": result.get("response", ""),
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    async def register(self):
        """Register with backend"""
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    f"{self.backend_url}/api/v1/agents/register",
                    json={
                        "name": self.name,
                        "capabilities": [
                            "code_analysis",
                            "refactoring",
                            "optimization",
                            "security_review"
                        ],
                        "status": "active"
                    }
                )
                print(f"[{self.name}] Registered with backend")
            except Exception as e:
                print(f"[{self.name}] Registration failed: {e}")
    
    async def run(self):
        """Main agent loop"""
        print(f"[{self.name}] Starting...")
        await self.register()
        
        while True:
            try:
                # Poll for tasks
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.backend_url}/api/v1/agents/tasks/{self.name}"
                    )
                    if response.status_code == 200:
                        task = response.json()
                        if task:
                            print(f"[{self.name}] Processing task: {task.get('id')}")
                            # Process the task
                            if task.get("type") == "analyze_code":
                                result = await self.analyze_code(task.get("code", ""))
                                # Report result
                                await client.post(
                                    f"{self.backend_url}/api/v1/agents/tasks/{task['id']}/result",
                                    json=result
                                )
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
            
            await asyncio.sleep(5)  # Poll every 5 seconds

if __name__ == "__main__":
    agent = CodeGenerationImprover()
    asyncio.run(agent.run())