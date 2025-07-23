from flask import Flask, request, jsonify, Response
import logging
import os
import time
import psutil

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Track agent metrics
start_time = time.time()
generation_count = 0
last_generation_time = 0

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "GPT-Engineer"})

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    global generation_count, last_generation_time
    
    uptime = time.time() - start_time
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    metrics_data = f"""# HELP gpt_engineer_uptime_seconds Agent uptime in seconds
# TYPE gpt_engineer_uptime_seconds counter
gpt_engineer_uptime_seconds {uptime:.2f}

# HELP gpt_engineer_generations_total Total number of code generations
# TYPE gpt_engineer_generations_total counter
gpt_engineer_generations_total {generation_count}

# HELP gpt_engineer_cpu_usage_percent CPU usage percentage
# TYPE gpt_engineer_cpu_usage_percent gauge
gpt_engineer_cpu_usage_percent {cpu_percent}

# HELP gpt_engineer_memory_usage_percent Memory usage percentage
# TYPE gpt_engineer_memory_usage_percent gauge
gpt_engineer_memory_usage_percent {memory.percent}

# HELP gpt_engineer_last_generation_timestamp_seconds Timestamp of last generation
# TYPE gpt_engineer_last_generation_timestamp_seconds gauge
gpt_engineer_last_generation_timestamp_seconds {last_generation_time}

# HELP gpt_engineer_info Agent information
# TYPE gpt_engineer_info gauge
gpt_engineer_info{{version="1.0.0",agent="gpt-engineer"}} 1
"""
    
    return Response(metrics_data, mimetype='text/plain')

@app.route('/generate', methods=['POST'])
def generate_code():
    global generation_count, last_generation_time
    
    data = request.get_json()
    prompt = data.get('prompt', '')
    language = data.get('language', 'python')
    
    # Update metrics
    generation_count += 1
    last_generation_time = time.time()
    
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
