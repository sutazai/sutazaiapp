#!/usr/bin/env python3
"""
Mock Ollama Server for Testing and Development
Provides a simple HTTP server that mimics Ollama API responses
"""

import json
import asyncio
from datetime import datetime
from aiohttp import web
import aiohttp_cors

# Mock responses for different models
MOCK_MODELS = [
    {"name": "llama3-chatqa:latest", "size": 4700000000},
    {"name": "llama3:latest", "size": 4700000000},
    {"name": "codellama:latest", "size": 3800000000}
]

MOCK_RESPONSES = {
    "hello": "Hello! I'm a mock AI assistant running in development mode. How can I help you today?",
    "test": "This is a test response from the mock Ollama server. The system is working correctly!",
    "code": "I can help you with coding tasks! Here's a simple Python example:\n\n```python\ndef hello_world():\n    print('Hello, World!')\n    return 'Success'\n```",
    "system": "System Status: Mock Ollama server is running. This is a development/testing environment.",
    "default": "I'm a mock AI assistant. In production, this would be powered by a real language model. What would you like to know?"
}

async def handle_tags(request):
    """Handle /api/tags endpoint"""
    return web.json_response({"models": MOCK_MODELS})

async def handle_generate(request):
    """Handle /api/generate endpoint"""
    try:
        data = await request.json()
        prompt = data.get("prompt", "").lower()
        model = data.get("model", "llama3-chatqa")
        
        # Generate response based on prompt keywords
        response_text = MOCK_RESPONSES["default"]
        for keyword, response in MOCK_RESPONSES.items():
            if keyword in prompt and keyword != "default":
                response_text = response
                break
        
        # Add context about being a mock server
        if "mock" not in response_text.lower():
            response_text += "\n\n[Note: This is a mock response for development/testing]"
        
        mock_response = {
            "model": model,
            "created_at": datetime.now().isoformat(),
            "response": response_text,
            "done": True,
            "context": [],
            "total_duration": 1500000000,
            "load_duration": 500000000,
            "prompt_eval_count": len(prompt.split()),
            "prompt_eval_duration": 300000000,
            "eval_count": len(response_text.split()),
            "eval_duration": 700000000
        }
        
        return web.json_response(mock_response)
        
    except Exception as e:
        return web.json_response(
            {"error": f"Mock server error: {str(e)}"}, 
            status=500
        )

async def handle_health(request):
    """Handle health check"""
    return web.json_response({
        "status": "ok",
        "message": "Mock Ollama server running",
        "models_available": len(MOCK_MODELS)
    })

async def create_app():
    """Create and configure the web application"""
    app = web.Application()
    
    # Add CORS support
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add routes
    app.router.add_get('/api/tags', handle_tags)
    app.router.add_post('/api/generate', handle_generate)
    app.router.add_get('/health', handle_health)
    app.router.add_get('/', handle_health)
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def main():
    """Main server function"""
    app = await create_app()
    
    print("🤖 Starting Mock Ollama Server...")
    print("📍 Server: http://127.0.0.1:11434")
    print("🔍 Health: http://127.0.0.1:11434/health")
    print("📋 Models: http://127.0.0.1:11434/api/tags")
    print("💬 Generate: POST http://127.0.0.1:11434/api/generate")
    print("✅ Mock server ready for testing!")
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '127.0.0.1', 11434)
    await site.start()
    
    # Keep the server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down mock server...")
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Failed to start mock server: {e}")
        print("Install dependencies: pip install aiohttp aiohttp-cors")