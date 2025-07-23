from flask import Flask, request, jsonify, Response
import logging
import os
import time
import psutil

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Track agent metrics
start_time = time.time()
task_count = 0
last_task_time = 0

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "CrewAI"})

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    global task_count, last_task_time
    
    uptime = time.time() - start_time
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    metrics_data = f"""# HELP crewai_uptime_seconds Agent uptime in seconds
# TYPE crewai_uptime_seconds counter
crewai_uptime_seconds {uptime:.2f}

# HELP crewai_tasks_total Total number of tasks processed
# TYPE crewai_tasks_total counter
crewai_tasks_total {task_count}

# HELP crewai_cpu_usage_percent CPU usage percentage
# TYPE crewai_cpu_usage_percent gauge
crewai_cpu_usage_percent {cpu_percent}

# HELP crewai_memory_usage_percent Memory usage percentage
# TYPE crewai_memory_usage_percent gauge
crewai_memory_usage_percent {memory.percent}

# HELP crewai_last_task_timestamp_seconds Timestamp of last task execution
# TYPE crewai_last_task_timestamp_seconds gauge
crewai_last_task_timestamp_seconds {last_task_time}

# HELP crewai_info Agent information
# TYPE crewai_info gauge
crewai_info{{version="1.0.0",agent="crewai"}} 1
"""
    
    return Response(metrics_data, mimetype='text/plain')

@app.route('/execute', methods=['POST'])
def execute_crew():
    global task_count, last_task_time
    
    data = request.get_json()
    task = data.get('task', '')
    crew_size = data.get('crew_size', 3)
    
    # Update metrics
    task_count += 1
    last_task_time = time.time()
    
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
