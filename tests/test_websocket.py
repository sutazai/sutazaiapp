#!/usr/bin/env python3
"""Test WebSocket chat endpoint"""

import asyncio
import websockets
import json
import sys

async def test_websocket():
    uri = "ws://localhost:10200/api/v1/chat/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Wait for connection message
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Connection response: {json.dumps(data, indent=2)}")
            session_id = data.get("session_id")
            
            # Test 1: Send a chat message with streaming
            print("\n--- Test 1: Streaming chat ---")
            message = {
                "type": "chat",
                "message": "What is the capital of France?",
                "model": "tinyllama:latest",
                "stream": True
            }
            await websocket.send(json.dumps(message))
            print(f"Sent: {message}")
            
            # Receive streaming responses
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                if data.get("type") == "message_received":
                    print(f"Message acknowledged")
                elif data.get("type") == "stream_start":
                    print(f"Stream started with model: {data.get('model')}")
                elif data.get("type") == "stream_chunk":
                    print(data.get("content"), end="", flush=True)
                elif data.get("type") == "stream_end":
                    print(f"\n\nStream complete. Full response length: {len(data.get('full_response', ''))}")
                    break
                elif data.get("type") == "error":
                    print(f"Error: {data.get('message')}")
                    break
            
            # Test 2: Send a non-streaming message
            print("\n--- Test 2: Non-streaming chat ---")
            message = {
                "type": "chat",
                "message": "Count to 5",
                "model": "tinyllama:latest",
                "stream": False
            }
            await websocket.send(json.dumps(message))
            print(f"Sent: {message}")
            
            # Receive responses
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                if data.get("type") == "response":
                    print(f"Response: {data.get('content')}")
                    break
                elif data.get("type") == "error":
                    print(f"Error: {data.get('message')}")
                    break
            
            # Test 3: Get chat history
            print("\n--- Test 3: Get chat history ---")
            message = {"type": "get_history"}
            await websocket.send(json.dumps(message))
            
            response = await websocket.recv()
            data = json.loads(response)
            if data.get("type") == "history":
                print(f"History: {data.get('count')} messages")
                for msg in data.get("messages", []):
                    print(f"  [{msg.get('role')}]: {msg.get('content')[:50]}...")
            
            # Test 4: Ping/Pong
            print("\n--- Test 4: Ping/Pong ---")
            message = {"type": "ping"}
            await websocket.send(json.dumps(message))
            
            response = await websocket.recv()
            data = json.loads(response)
            if data.get("type") == "pong":
                print(f"Pong received at: {data.get('timestamp')}")
            
            print("\n✅ All WebSocket tests passed!")
            
    except websockets.exceptions.ConnectionRefused:
        print(f"❌ Could not connect to {uri}")
        print("Make sure the backend is running on port 10200")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Testing WebSocket Chat Endpoint")
    print("================================")
    asyncio.run(test_websocket())