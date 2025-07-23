from flask import Flask, request, jsonify, Response
import logging
import os
import time
import psutil

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Track agent metrics
start_time = time.time()
edit_count = 0
last_edit_time = 0

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "Aider"})

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    global edit_count, last_edit_time
    
    uptime = time.time() - start_time
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    metrics_data = f"""# HELP aider_uptime_seconds Agent uptime in seconds
# TYPE aider_uptime_seconds counter
aider_uptime_seconds {uptime:.2f}

# HELP aider_edits_total Total number of code edits processed
# TYPE aider_edits_total counter
aider_edits_total {edit_count}

# HELP aider_cpu_usage_percent CPU usage percentage
# TYPE aider_cpu_usage_percent gauge
aider_cpu_usage_percent {cpu_percent}

# HELP aider_memory_usage_percent Memory usage percentage
# TYPE aider_memory_usage_percent gauge
aider_memory_usage_percent {memory.percent}

# HELP aider_last_edit_timestamp_seconds Timestamp of last edit
# TYPE aider_last_edit_timestamp_seconds gauge
aider_last_edit_timestamp_seconds {last_edit_time}

# HELP aider_info Agent information
# TYPE aider_info gauge
aider_info{{version="1.0.0",agent="aider"}} 1
"""
    
    return Response(metrics_data, mimetype='text/plain')

@app.route('/edit', methods=['POST'])
def edit_code():
    global edit_count, last_edit_time
    
    data = request.get_json()
    file_path = data.get('file_path', '')
    instructions = data.get('instructions', '')
    
    # Update metrics
    edit_count += 1
    last_edit_time = time.time()
    
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
