#!/usr/bin/env python3
"""Health Monitor Service for SutazAI"""

import os
import time
import docker
import logging
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
client = docker.from_env()

def get_container_health():
    """Get health status of all containers"""
    containers = []
    try:
        for container in client.containers.list(all=True):
            if container.name.startswith('sutazai-'):
                containers.append({
                    'name': container.name,
                    'status': container.status,
                    'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'none')
                })
    except Exception as e:
        logger.error(f"Error getting container health: {e}")
    return containers

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/containers')
def containers():
    """Get container status"""
    return jsonify(get_container_health())

@app.route('/metrics')
def metrics():
    """Get system metrics"""
    containers = get_container_health()
    healthy = sum(1 for c in containers if c['health'] == 'healthy')
    unhealthy = sum(1 for c in containers if c['health'] == 'unhealthy')
    
    return jsonify({
        'total_containers': len(containers),
        'healthy': healthy,
        'unhealthy': unhealthy,
        'containers': containers
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090, debug=False)