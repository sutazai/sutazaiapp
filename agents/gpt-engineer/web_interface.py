from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "GPT-Engineer"})

@app.route('/generate', methods=['POST'])
def generate_code():
    data = request.get_json()
    prompt = data.get('prompt', '')
    language = data.get('language', 'python')
    
    result = {
        "status": "completed",
        "prompt": prompt,
        "language": language,
        "result": f"Code generation completed for: {prompt}",
        "agent": "GPT-Engineer"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
