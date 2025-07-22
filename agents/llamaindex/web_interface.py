from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "LlamaIndex"})

@app.route('/index', methods=['POST'])
def index_data():
    data = request.get_json()
    documents = data.get('documents', [])
    index_type = data.get('index_type', 'vector')
    
    result = {
        "status": "completed",
        "documents": len(documents),
        "index_type": index_type,
        "result": f"Indexed {len(documents)} documents using {index_type}",
        "agent": "LlamaIndex"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
