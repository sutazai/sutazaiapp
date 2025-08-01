#!/usr/bin/env python3
"""
Simple HTTP Backend for SutazAI using only Python standard library
"""

import http.server
import socketserver
import json
import urllib.parse
import time
from datetime import datetime

class SutazAIHandler(http.server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        """Handle GET requests"""
        path = urllib.parse.urlparse(self.path).path
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        
        # Set CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # Route handlers
        if path == '/':
            response = {"message": "SutazAI Backend v9.0.0", "status": "online"}
        elif path == '/health':
            response = {"status": "healthy", "timestamp": time.time()}
        elif path == '/api/system/status':
            response = {
                "status": "online",
                "uptime": 100.0,
                "active_agents": 3,
                "loaded_models": 5,
                "requests_count": 42,
                "timestamp": time.time()
            }
        elif path == '/api/agents/':
            response = [
                {"id": "1", "name": "DeepSeek-R1", "type": "coding", "status": "active", "created_at": "2024-01-01"},
                {"id": "2", "name": "Qwen3", "type": "general", "status": "active", "created_at": "2024-01-01"},
                {"id": "3", "name": "AutoGPT", "type": "automation", "status": "idle", "created_at": "2024-01-01"}
            ]
        elif path == '/api/models/':
            response = [
                {"id": "1", "name": "tinyllama", "status": "loaded"},
                {"id": "2", "name": "qwen3:8b", "status": "loaded"},
                {"id": "3", "name": "llama2:13b", "status": "loaded"}
            ]
        elif path == '/ai/services/status':
            response = {
                "services": {
                    "tinyllama": {"status": "healthy", "details": {"model": "tinyllama", "memory_usage": "2.1GB"}},
                    "qwen3": {"status": "healthy", "details": {"model": "qwen3:8b", "memory_usage": "1.8GB"}},
                    "ollama": {"status": "healthy", "details": {"port": 11434, "models": 5}},
                    "vector_db": {"status": "healthy", "details": {"collections": 3, "vectors": 10000}},
                    "redis": {"status": "healthy", "details": {"connections": 5, "memory": "150MB"}}
                }
            }
        else:
            response = {"error": "Not found", "path": path}
        
        # Send response
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_POST(self):
        """Handle POST requests"""
        path = urllib.parse.urlparse(self.path).path
        content_length = int(self.headers.get('Content-Length', 0))
        
        # Read request body
        post_data = {}
        if content_length > 0:
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                post_data = json.loads(body)
            except:
                pass
        
        # Set CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # Route handlers
        if path == '/api/agents/':
            # Create agent
            name = post_data.get('name', 'New Agent')
            agent_type = post_data.get('type', 'general')
            response = {
                "id": str(int(time.time())),
                "name": name,
                "type": agent_type,
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
        elif path.startswith('/api/agents/') and path.endswith('/chat'):
            # Chat with agent
            agent_id = path.split('/')[-2]
            message = post_data.get('message', '')
            response = {
                "agent_id": agent_id,
                "response": f"Hello! I'm agent {agent_id}. You said: '{message}'. This is a simulated response from SutazAI v9.",
                "timestamp": time.time()
            }
        elif path == '/api/code/generate':
            # Generate code
            prompt = post_data.get('prompt', '')
            language = post_data.get('language', 'python')
            
            code = f'''# Generated {language} code for: {prompt}
def sutazai_generated_function():
    """
    SutazAI v9 Generated Code
    Prompt: {prompt}
    Language: {language}
    """
    print("Hello from SutazAI v9!")
    print("This code was generated by the autonomous system")
    
    # Example implementation
    result = "SutazAI v9 - Code generation successful"
    return result

if __name__ == "__main__":
    output = sutazai_generated_function()
    print(output)
'''
            
            response = {
                "code": code,
                "language": language,
                "prompt": prompt,
                "timestamp": time.time()
            }
        else:
            response = {"error": "Not found", "path": path}
        
        # Send response
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Log requests"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")

class SutazAIServer:
    """SutazAI HTTP Server"""
    
    def __init__(self, host='0.0.0.0', port=8000):
        self.host = host
        self.port = port
        self.httpd = None
    
    def start(self):
        """Start the server"""
        try:
            self.httpd = socketserver.TCPServer((self.host, self.port), SutazAIHandler)
            print(f"🚀 SutazAI Backend v9 starting on {self.host}:{self.port}")
            print(f"🌐 API Available at: http://{self.host}:{self.port}")
            print(f"📚 Health Check: http://{self.host}:{self.port}/health")
            print(f"🤖 System Status: http://{self.host}:{self.port}/api/system/status")
            print("=" * 60)
            print("🛑 Press Ctrl+C to stop the server")
            print("=" * 60)
            
            self.httpd.serve_forever()
            
        except KeyboardInterrupt:
            print("\\n🛑 Server stopping...")
            self.stop()
        except Exception as e:
            print(f"❌ Server error: {e}")
    
    def stop(self):
        """Stop the server"""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            print("✅ Server stopped")

if __name__ == "__main__":
    server = SutazAIServer()
    server.start()