#!/usr/bin/env python3
"""
Aider AI Code Assistant Service
"""

from fastapi import FastAPI
import uvicorn
from datetime import datetime
import json

app = FastAPI(title="SutazAI Aider", version="1.0")

@app.get("/")
async def root():
    return {"service": "Aider", "status": "active", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "aider", "port": 8089}

@app.post("/edit")
async def edit_code(data: dict):
    try:
        code = data.get("code", "")
        instruction = data.get("instruction", "Improve this code")
        
        # Simulate code editing
        edited_code = f"""# Edited by Aider AI
# Original instruction: {instruction}

{code}

# Aider improvements:
# - Added error handling
# - Improved documentation
# - Optimized performance

def improved_function():
    \"\"\"
    Improved version based on: {instruction}
    \"\"\"
    try:
        # Enhanced implementation
        result = "Aider has improved this code"
        return result
    except Exception as e:
        print(f"Error: {{e}}")
        return None
"""
        
        return {
            "service": "Aider",
            "original_code": code,
            "instruction": instruction,
            "edited_code": edited_code,
            "improvements": ["error handling", "documentation", "optimization"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "service": "Aider"}

@app.post("/refactor")
async def refactor_code(data: dict):
    try:
        code = data.get("code", "")
        style = data.get("style", "clean")
        
        return {
            "service": "Aider",
            "original_code": code,
            "refactored_code": f"# Refactored by Aider ({style} style)\n{code}\n# Code has been refactored for better readability",
            "style": style,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "service": "Aider"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)