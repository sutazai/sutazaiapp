from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "Aider"})

@app.route('/edit', methods=['POST'])
def edit_code():
    data = request.get_json()
    file_path = data.get('file_path', '')
    instructions = data.get('instructions', '')
    
    result = {
        "status": "completed",
        "file_path": file_path,
        "instructions": instructions,
        "result": f"Code editing completed for: {file_path}",
        "agent": "Aider"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
