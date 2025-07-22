from flask import Flask, request, jsonify
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "RealtimeSTT"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    audio_file = data.get('audio_file', '')
    language = data.get('language', 'en')
    
    result = {
        "status": "completed",
        "audio_file": audio_file,
        "language": language,
        "result": f"Audio transcription completed for {audio_file}",
        "agent": "RealtimeSTT"
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
