from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "Documind"})

@app.route('/process', methods=['POST'])
def process_document():
    data = request.get_json()
    document = data.get('document', '')
    action = data.get('action', 'analyze')
    
    result = {
        "status": "completed",
        "document": document,
        "action": action,
        "result": f"Document {action} completed for {document}",
        "agent": "Documind"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
