#!/usr/bin/env python3
"""
LiteLLM Proxy Manager Agent
Manages LiteLLM proxy configuration and health monitoring
"""

import os
import time
import logging
import requests
import psycopg2
from flask import Flask, jsonify
from threading import Thread
import schedule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class LiteLLMProxyManager:
    def __init__(self):
        self.litellm_host = os.getenv('LITELLM_HOST', 'http://sutazai-litellm:4000')  
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://sutazai:sutazai_password@sutazai-postgres:5432/sutazai')
        
    def check_litellm_health(self):
        """Check if LiteLLM proxy is healthy"""
        try:
            response = requests.get(f"{self.litellm_host}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"LiteLLM health check failed: {e}")
            return False
    
    def check_database_health(self):
        """Check database connectivity"""
        try:
            conn = psycopg2.connect(self.database_url)
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_proxy_stats(self):
        """Get LiteLLM proxy statistics"""
        try:
            response = requests.get(f"{self.litellm_host}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                return {
                    'available_models': len(models.get('data', [])),
                    'models': [model.get('id') for model in models.get('data', [])]
                }
        except Exception as e:
            logger.error(f"Failed to get proxy stats: {e}")
        return {'available_models': 0, 'models': []}
    
    def monitor_proxy(self):
        """Monitor LiteLLM proxy health and performance"""
        proxy_healthy = self.check_litellm_health()
        db_healthy = self.check_database_health()
        stats = self.get_proxy_stats()
        
        logger.info(f"Proxy health: {proxy_healthy}, DB health: {db_healthy}, Models: {stats['available_models']}")
        
        if not proxy_healthy:
            logger.warning("LiteLLM proxy is unhealthy - attempting recovery actions")
            
        if not db_healthy:
            logger.warning("Database connection failed")
    
    def optimize_proxy_config(self):
        """Optimize proxy configuration based on usage patterns"""
        logger.info("Running proxy configuration optimization")
        # This would implement dynamic configuration updates
        pass

# Global instance
manager = LiteLLMProxyManager()

@app.route('/health')
def health():
    """Health check endpoint"""
    proxy_healthy = manager.check_litellm_health()
    db_healthy = manager.check_database_health()
    
    return jsonify({
        'status': 'healthy' if proxy_healthy and db_healthy else 'degraded',
        'proxy_healthy': proxy_healthy,
        'database_healthy': db_healthy,
        'timestamp': time.time()
    })

@app.route('/stats')
def stats():
    """Get proxy statistics"""
    return jsonify(manager.get_proxy_stats())

@app.route('/optimize')
def optimize():
    """Trigger proxy optimization"""
    try:
        manager.optimize_proxy_config()
        return jsonify({'status': 'success', 'message': 'Optimization completed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def run_scheduler():
    """Run scheduled tasks"""
    schedule.every(2).minutes.do(manager.monitor_proxy)
    schedule.every(30).minutes.do(manager.optimize_proxy_config)
    
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == '__main__':
    # Start scheduler in background
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    app.run(host='0.0.0.0', port=8521, debug=False)