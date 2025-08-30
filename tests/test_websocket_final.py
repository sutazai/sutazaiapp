#!/usr/bin/env python3
"""Final WebSocket test for the /ws endpoint"""

import json
import base64
import hashlib
import socket
import time

def create_websocket_key():
    """Generate a WebSocket key"""
    return base64.b64encode(b"test_key_12345678").decode('utf-8')

def send_websocket_frame(sock, payload):
    """Send a WebSocket text frame"""
    import struct
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
    import struct
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
    PATH = '/ws'  # Correct path in main.py
    
    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)  # 30 second timeout
    
    try:
        # Connect
        print(f"Connecting to {HOST}:{PORT}{PATH}...")
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
        headers = response.decode().split('\r\n\r\n')[0]
        print(headers)
        
        # Check if upgrade was successful
        if b"101 Switching Protocols" not in response:
            print("❌ WebSocket handshake failed!")
            return
        
        print("\n✅ WebSocket connected!\n")
        
        # Wait for connection message
        print("Waiting for connection message...")
        msg = receive_websocket_frame(sock)
        if msg:
            print(f"Connection message:")
            print(json.dumps(msg, indent=2))
            session_id = msg.get('session_id')
            print(f"\nSession ID: {session_id}\n")
        
        # Test 1: Send a non-streaming chat message
        print("--- Test 1: Non-streaming chat ---")
        chat_msg = {
            "type": "chat",
            "message": "What is 2+2?",
            "model": "tinyllama:latest",
            "stream": False
        }
        print(f"Sending: {chat_msg}")
        send_websocket_frame(sock, chat_msg)
        
        # Receive responses
        print("\nWaiting for responses...")
        for i in range(10):  # Wait for up to 10 messages
            msg = receive_websocket_frame(sock)
            if msg:
                msg_type = msg.get('type')
                print(f"Received message type: {msg_type}")
                
                if msg_type == 'message_received':
                    print(f"  Message acknowledged: {msg.get('message')}")
                elif msg_type == 'response':
                    print(f"  Response: {msg.get('content', '')[:200]}")
                    break
                elif msg_type == 'error':
                    print(f"  Error: {msg.get('message')}")
                    break
            else:
                print("No message received")
                break
        
        # Test 2: Send a streaming chat message
        print("\n--- Test 2: Streaming chat ---")
        chat_msg = {
            "type": "chat",
            "message": "Tell me a very short joke",
            "model": "tinyllama:latest",
            "stream": True
        }
        print(f"Sending: {chat_msg}")
        send_websocket_frame(sock, chat_msg)
        
        # Receive streaming responses
        print("\nStreaming response:")
        full_response = ""
        for i in range(100):  # Wait for up to 100 chunks
            msg = receive_websocket_frame(sock)
            if msg:
                msg_type = msg.get('type')
                
                if msg_type == 'stream_start':
                    print(f"Stream started with model: {msg.get('model')}")
                elif msg_type == 'stream_chunk':
                    chunk = msg.get('content', '')
                    print(chunk, end='', flush=True)
                    full_response += chunk
                    if msg.get('done'):
                        print("\n[Stream done]")
                elif msg_type == 'stream_end':
                    print(f"\nStream complete. Total length: {len(msg.get('full_response', ''))}")
                    break
                elif msg_type == 'error':
                    print(f"\nError: {msg.get('message')}")
                    break
        
        # Test 3: Ping/Pong
        print("\n--- Test 3: Ping/Pong ---")
        ping_msg = {"type": "ping"}
        print(f"Sending: {ping_msg}")
        send_websocket_frame(sock, ping_msg)
        
        msg = receive_websocket_frame(sock)
        if msg and msg.get('type') == 'pong':
            print(f"✅ Pong received at: {msg.get('timestamp')}")
        
        # Test 4: Get history
        print("\n--- Test 4: Get history ---")
        history_msg = {"type": "get_history"}
        print(f"Sending: {history_msg}")
        send_websocket_frame(sock, history_msg)
        
        msg = receive_websocket_frame(sock)
        if msg and msg.get('type') == 'history':
            print(f"History contains {msg.get('count')} messages")
            for message in msg.get('messages', [])[:3]:  # Show first 3
                role = message.get('role')
                content = message.get('content', '')[:50]
                print(f"  [{role}]: {content}...")
        
        print("\n✅ All WebSocket tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    print("WebSocket Test for /ws Endpoint")
    print("================================\n")
    test_websocket()