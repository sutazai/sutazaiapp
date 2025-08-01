#!/usr/bin/env python3
"""
Ollama Integration Specialist Agent
"""

import os
import time
import logging
import requests
import docker
from flask import Flask, jsonify
from threading import Thread
import schedule
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class OllamaIntegrationSpecialist:
    def __init__(self):
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://sutazai-ollama:11434')
        self.docker_client = docker.from_env()
        
    def check_ollama_health(self):
        """Check if Ollama is healthy"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
        try:
            return response.status_code == 200
        except Exception as e:
            return False
    
    def list_ollama_models(self):
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                return response.json().get('models', [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
        return []
    
    def pull_essential_models(self):
        """Pull essential models if not present"""
        essential_models = [
            'llama3.2:3b',
            'qwen2.5:3b',
            'deepseek-coder:1.3b'
        ]
        
        current_models = [model['name'] for model in self.list_ollama_models()]
        
        for model in essential_models:
            if not any(model in current_model for current_model in current_models):
                logger.info(f"Pulling essential model: {model}")
                try:
                    # Set a timeout for model pulling to prevent hanging
                    import signal
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Model pull timeout")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)  # 5 minute timeout
                    
                    ollama.pull(model)
                    signal.alarm(0)  # Cancel timeout
                    logger.info(f"Successfully pulled {model}")
                except TimeoutError:
                    logger.error(f"Timeout pulling {model} after 5 minutes")
                except Exception as e:
                    logger.error(f"Failed to pull {model}: {e}")
                finally:
                    signal.alarm(0)  # Ensure timeout is cancelled
    
    def monitor_integration(self):
        if not self.check_ollama_health():
            logger.warning("Ollama is not healthy")
            
            
        models = self.list_ollama_models()
        logger.info(f"Available Ollama models: {len(models)}")

# Global instance
specialist = OllamaIntegrationSpecialist()

@app.route('/health')
def health():
    """Health check endpoint"""
    ollama_healthy = specialist.check_ollama_health()
    
    return jsonify({
        'ollama_healthy': ollama_healthy,
        'timestamp': time.time()
    })

@app.route('/models')
def models():
    """Get available models"""
    return jsonify({
        'models': specialist.list_ollama_models()
    })

@app.route('/pull-essential')
def pull_essential():
    """Pull essential models"""
    try:
        specialist.pull_essential_models()
        return jsonify({'status': 'success', 'message': 'Essential models pull initiated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def run_scheduler():
    """Run scheduled tasks"""
    schedule.every(5).minutes.do(specialist.monitor_integration)
    schedule.every(1).hours.do(specialist.pull_essential_models)
    
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == '__main__':
    # Start scheduler in background
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Pull essential models on startup in background
    model_pulling_thread = Thread(target=specialist.pull_essential_models, daemon=True)
    model_pulling_thread.start()
    
    app.run(host='0.0.0.0', port=8520, debug=False)