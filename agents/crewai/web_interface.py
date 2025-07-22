from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "CrewAI"})

@app.route('/execute', methods=['POST'])
def execute_crew():
    data = request.get_json()
    task = data.get('task', '')
    crew_size = data.get('crew_size', 3)
    
    result = {
        "status": "completed",
        "task": task,
        "crew_size": crew_size,
        "result": f"CrewAI task completed with {crew_size} agents",
        "agent": "CrewAI"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
