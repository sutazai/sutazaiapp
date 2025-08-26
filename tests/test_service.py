#!/usr/bin/env python3
"""
Simple test service for mesh testing
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
import threading

class TestServiceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'status': 'healthy', 'service': 'test-service', 'port': self.server.server_port}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'message': 'Hello from test service', 'path': self.path, 'port': self.server.server_port}
            self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        response = {
            'echo': json.loads(body) if body else {},
            'path': self.path,
            'port': self.server.server_port,
            'headers': dict(self.headers)
        }
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        pass  # Suppress logs

def start_server(port):
    server = HTTPServer(('localhost', port), TestServiceHandler)
    print(f"Test service started on port {port}")
    server.serve_forever()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(port)