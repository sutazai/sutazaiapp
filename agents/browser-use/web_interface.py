from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "Browser-Use"})

@app.route('/browse', methods=['POST'])
def browse():
    data = request.get_json()
    url = data.get('url', '')
    action = data.get('action', 'visit')
    
    result = {
        "status": "completed",
        "url": url,
        "action": action,
        "result": f"Browser action '{action}' completed on {url}",
        "agent": "Browser-Use"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
