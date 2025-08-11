#\!/usr/bin/env python3
"""
hygiene-backend Service
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
        "service": "hygiene-backend",
        "version": "1.0.0"
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint with more details"""
    return jsonify({
        "status": "operational",
        "service": "hygiene-backend",
        "uptime": "running",
        "version": "1.0.0"
    }), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "hygiene-backend Service",
        "status": "running",
        "endpoints": ["/health", "/status"]
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting hygiene-backend service on {host}:{port}")
    app.run(host=host, port=port, debug=False)
