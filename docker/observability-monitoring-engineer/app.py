#\!/usr/bin/env python3
"""
observability-monitoring-engineer Service
Basic Flask service implementation with health endpoints
"""

import os
from flask import Flask, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "observability-monitoring-engineer",
        "version": "1.0.0"
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint with more details"""
    return jsonify({
        "status": "operational",
        "service": "observability-monitoring-engineer",
        "uptime": "running",
        "version": "1.0.0"
    }), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "observability-monitoring-engineer Service",
        "status": "running",
        "endpoints": ["/health", "/status"]
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting observability-monitoring-engineer service on {host}:{port}")
    app.run(host=host, port=port, debug=False)
