from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "JAX"})

@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    operation = data.get('operation', 'matrix_multiply')
    size = data.get('size', 100)
    
    result = {
        "status": "completed",
        "operation": operation,
        "size": size,
        "result": f"JAX computation {operation} completed with size {size}",
        "agent": "JAX"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
