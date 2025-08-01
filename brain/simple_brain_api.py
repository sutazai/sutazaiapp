#!/usr/bin/env python3
"""
Simple Brain API - Direct Ollama Integration
Provides AI brain functionality without complex dependencies
"""

import json
import requests
import logging
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class SimpleBrainAPI:
    def __init__(self):
        self.ollama_base = "http://localhost:11434"
        self.default_model = "llama3.2:3b"
        self.available_models = self.get_available_models()
        
    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_base}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
        return []
    
    def generate_response(self, prompt, model=None, max_tokens=1000, temperature=0.7):
        """Generate response using Ollama"""
        if not model:
            model = self.default_model
            
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.ollama_base}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "model": model,
                    "done": data.get("done", False)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "model": model
                }
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model
            }
    
    def chat_completion(self, messages, model=None, max_tokens=1000, temperature=0.7):
        """OpenAI-compatible chat completion"""
        if not model:
            model = self.default_model
            
        # Convert messages to single prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant: "
        
        result = self.generate_response(prompt, model, max_tokens, temperature)
        
        if result["success"]:
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["response"]
                    },
                    "finish_reason": "stop"
                }]
            }
        else:
            return {
                "error": {
                    "message": result["error"],
                    "type": "api_error"
                }
            }

# Global brain instance
brain = SimpleBrainAPI()

@app.route('/health')
def health():
    """Health check endpoint"""
    ollama_healthy = len(brain.available_models) > 0
    return jsonify({
        'status': 'healthy' if ollama_healthy else 'degraded',
        'ollama_connected': ollama_healthy,
        'available_models': brain.available_models,
        'timestamp': time.time()
    })

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models (OpenAI compatible)"""
    models = []
    for model_name in brain.available_models:
        models.append({
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "sutazai"
        })
    
    return jsonify({
        "object": "list",
        "data": models
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        model = data.get('model', brain.default_model)
        max_tokens = data.get('max_tokens', 1000)
        temperature = data.get('temperature', 0.7)
        
        result = brain.chat_completion(messages, model, max_tokens, temperature)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "invalid_request_error"
            }
        }), 400

@app.route('/generate', methods=['POST'])
def generate():
    """Simple generation endpoint"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        model = data.get('model', brain.default_model)
        max_tokens = data.get('max_tokens', 1000)
        temperature = data.get('temperature', 0.7)
        
        result = brain.generate_response(prompt, model, max_tokens, temperature)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/models')
def models_simple():
    """Simple models endpoint"""
    return jsonify({
        "models": brain.available_models,
        "default": brain.default_model
    })

if __name__ == '__main__':
    logger.info(f"Starting Simple Brain API with models: {brain.available_models}")
    app.run(host='0.0.0.0', port=8889, debug=False)