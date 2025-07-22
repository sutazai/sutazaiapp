from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "ShellGPT"})

@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.get_json()
    command = data.get('command', '')
    shell = data.get('shell', 'bash')
    
    result = {
        "status": "completed",
        "command": command,
        "shell": shell,
        "result": f"Shell command executed: {command}",
        "agent": "ShellGPT"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
