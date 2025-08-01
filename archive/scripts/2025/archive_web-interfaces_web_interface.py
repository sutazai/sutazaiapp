from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "FinRobot"})

@app.route('/analyze', methods=['POST'])
def analyze_financial():
    data = request.get_json()
    symbol = data.get('symbol', '')
    analysis_type = data.get('type', 'overview')
    
    result = {
        "status": "completed",
        "symbol": symbol,
        "analysis_type": analysis_type,
        "result": f"Financial analysis of {symbol} completed",
        "agent": "FinRobot"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
