#!/usr/bin/env python3
"""Simple WebSocket test using only standard library"""

import json
import base64
import hashlib
import socket
import struct
import threading
import time

def create_websocket_key():
    """Generate a WebSocket key"""
    return base64.b64encode(b"test_key_12345678").decode('utf-8')

def create_websocket_accept(key):
    """Create the WebSocket accept header value"""
    magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    accept = base64.b64encode(
        hashlib.sha1((key + magic).encode()).digest()
    ).decode('utf-8')
    return accept

def send_websocket_frame(sock, payload):
    """Send a WebSocket text frame"""
    data = json.dumps(payload).encode('utf-8')
    
    # Create WebSocket frame
    frame = bytearray()
    
    # FIN (1) + RSV (000) + Opcode (0001 for text)
    frame.append(0x81)
    
    # Mask bit (1) + payload length
    length = len(data)
    if length < 126:
        frame.append(0x80 | length)
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack('>H', length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack('>Q', length))
    
    # Masking key (4 bytes)
    mask = bytearray([0x12, 0x34, 0x56, 0x78])
    frame.extend(mask)
    
    # Masked payload
    masked_data = bytearray()
    for i, byte in enumerate(data):
        masked_data.append(byte ^ mask[i % 4])
    frame.extend(masked_data)
    
    sock.send(frame)

def receive_websocket_frame(sock):
    """Receive and decode a WebSocket frame"""
    try:
        # Read first two bytes
        header = sock.recv(2)
        if not header or len(header) < 2:
            return None
        
        # Parse header
        fin = header[0] & 0x80
        opcode = header[0] & 0x0F
        masked = header[1] & 0x80
        payload_len = header[1] & 0x7F
        
        # Get actual payload length
        if payload_len == 126:
            payload_len = struct.unpack('>H', sock.recv(2))[0]
        elif payload_len == 127:
            payload_len = struct.unpack('>Q', sock.recv(8))[0]
        
        # Read mask if present
        if masked:
            mask = sock.recv(4)
        
        # Read payload
        payload = sock.recv(payload_len)
        
        # Unmask if needed
        if masked:
            unmasked = bytearray()
            for i, byte in enumerate(payload):
                unmasked.append(byte ^ mask[i % 4])
            payload = bytes(unmasked)
        
        # Handle different opcodes
        if opcode == 0x8:  # Close frame
            return None
        elif opcode == 0x1:  # Text frame
            return json.loads(payload.decode('utf-8'))
        
        return None
    except Exception as e:
        print(f"Error receiving frame: {e}")
        return None

def test_websocket():
    """Test the WebSocket endpoint"""
    HOST = 'localhost'
    PORT = 10200
    PATH = '/api/v1/chat/ws'
    
    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    
    try:
        # Connect
        sock.connect((HOST, PORT))
        
        # Send WebSocket handshake
        key = create_websocket_key()
        handshake = f"""GET {PATH} HTTP/1.1\r
Host: {HOST}:{PORT}\r
Upgrade: websocket\r
Connection: Upgrade\r
Sec-WebSocket-Key: {key}\r
Sec-WebSocket-Version: 13\r
\r
"""
        sock.send(handshake.encode())
        
        # Read handshake response
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(1024)
            if not chunk:
                break
            response += chunk
        
        print("Handshake response:")
        print(response.decode().split('\r\n\r\n')[0])
        
        # Check if upgrade was successful
        if b"101 Switching Protocols" not in response:
            print("WebSocket handshake failed!")
            return
        
        print("\n✅ WebSocket connected!\n")
        
        # Wait for connection message
        print("Waiting for connection message...")
        msg = receive_websocket_frame(sock)
        if msg:
            print(f"Connection: {json.dumps(msg, indent=2)}")
            session_id = msg.get('session_id')
        
        # Test 1: Send a chat message
        print("\n--- Test 1: Send chat message ---")
        chat_msg = {
            "type": "chat",
            "message": "What is 2+2?",
            "model": "tinyllama:latest",
            "stream": False
        }
        print(f"Sending: {chat_msg}")
        send_websocket_frame(sock, chat_msg)
        
        # Receive responses
        print("Waiting for responses...")
        response_count = 0
        while response_count < 5:  # Wait for up to 5 messages
            msg = receive_websocket_frame(sock)
            if msg:
                msg_type = msg.get('type')
                print(f"Received: {msg_type}")
                
                if msg_type == 'response':
                    print(f"Response: {msg.get('content', '')[:100]}")
                    break
                elif msg_type == 'error':
                    print(f"Error: {msg.get('message')}")
                    break
                
                response_count += 1
            else:
                break
        
        # Test 2: Ping/Pong
        print("\n--- Test 2: Ping/Pong ---")
        ping_msg = {"type": "ping"}
        print(f"Sending: {ping_msg}")
        send_websocket_frame(sock, ping_msg)
        
        msg = receive_websocket_frame(sock)
        if msg and msg.get('type') == 'pong':
            print(f"Pong received at: {msg.get('timestamp')}")
        
        print("\n✅ WebSocket tests completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    print("Simple WebSocket Test")
    print("=====================")
    test_websocket()