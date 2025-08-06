#!/usr/bin/env python3
"""
health-monitor Agent Implementation
"""
import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional

class HealthMonitorAgent:
    def __init__(self):
        self.name = "health-monitor"
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = os.getenv("DEFAULT_MODEL", "gpt-oss-r1:8b")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request"""
        # Implement agent-specific logic here
        prompt = request.get("prompt", "")
        
        # Call Ollama for LLM processing
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_base}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "agent": self.name,
                        "response": result.get("response", ""),
                        "status": "success"
                    }
                else:
                    return {
                        "agent": self.name,
                        "error": f"Ollama request failed: {response.status}",
                        "status": "error"
                    }

def run():
    """Run the agent"""
    print(f"Starting health-monitor agent...")
    agent = Health-monitorAgent()
    
    # In production, this would start a web server
    # For now, just indicate the agent is ready
    print(f"health-monitor agent ready on port :8100")
    
    # Keep the agent running
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print(f"Shutting down health-monitor agent...")

if __name__ == "__main__":
    run()
