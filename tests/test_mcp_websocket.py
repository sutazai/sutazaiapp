#!/usr/bin/env python3
"""
WebSocket Test Suite for MCP Bridge
Tests real-time communication, rate limiting, and message routing
"""

import asyncio
import json
import time
from typing import List
import websockets
from websockets.exceptions import ConnectionClosed


async def test_basic_connection():
    """Test basic WebSocket connection"""
    print("Test 1: Basic WebSocket Connection")
    try:
        uri = "ws://localhost:11100/ws/test-client-1"
        async with websockets.connect(uri) as websocket:
            # Receive welcome message
            message = await websocket.recv()
            data = json.loads(message)
            
            assert data["type"] == "connected", "Should receive connected message"
            assert data["client_id"] == "test-client-1", "Should receive correct client_id"
            assert "rate_limits" in data, "Should include rate limits info"
            
            print(f"✅ Connected successfully: {data}")
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


async def test_ping_pong():
    """Test ping/pong keep-alive"""
    print("\nTest 2: Ping/Pong Keep-Alive")
    try:
        uri = "ws://localhost:11100/ws/test-client-ping"
        async with websockets.connect(uri) as websocket:
            # Skip welcome message
            await websocket.recv()
            
            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Receive pong
            message = await websocket.recv()
            data = json.loads(message)
            
            assert data["type"] == "pong", "Should receive pong response"
            print(f"✅ Ping/Pong working: {data}")
            return True
    except Exception as e:
        print(f"❌ Ping/Pong failed: {e}")
        return False


async def test_broadcast():
    """Test broadcast messages to multiple clients"""
    print("\nTest 3: Broadcast Messages")
    try:
        # Connect two clients
        uri1 = "ws://localhost:11100/ws/broadcaster"
        uri2 = "ws://localhost:11100/ws/receiver"
        
        async with websockets.connect(uri1) as ws1, websockets.connect(uri2) as ws2:
            # Skip welcome messages
            await ws1.recv()
            await ws2.recv()
            
            # Broadcaster sends broadcast message
            await ws1.send(json.dumps({
                "type": "broadcast",
                "payload": {"message": "Hello everyone!"}
            }))
            
            # Receiver should get the broadcast
            message = await asyncio.wait_for(ws2.recv(), timeout=2)
            data = json.loads(message)
            
            assert data["from"] == "broadcaster", "Should identify sender"
            assert data["data"]["message"] == "Hello everyone!", "Should contain payload"
            print(f"✅ Broadcast working: {data}")
            return True
    except Exception as e:
        print(f"❌ Broadcast failed: {e}")
        return False


async def test_direct_message():
    """Test direct message routing"""
    print("\nTest 4: Direct Message Routing")
    try:
        uri1 = "ws://localhost:11100/ws/sender"
        uri2 = "ws://localhost:11100/ws/recipient"
        
        async with websockets.connect(uri1) as ws1, websockets.connect(uri2) as ws2:
            # Skip welcome messages
            await ws1.recv()
            await ws2.recv()
            
            # Send direct message
            await ws1.send(json.dumps({
                "type": "direct",
                "target": "recipient",
                "payload": {"message": "Private message"}
            }))
            
            # Sender receives ACK
            ack_message = await asyncio.wait_for(ws1.recv(), timeout=2)
            ack_data = json.loads(ack_message)
            assert ack_data["type"] == "ack", "Sender should receive ACK"
            
            # Recipient receives message
            message = await asyncio.wait_for(ws2.recv(), timeout=2)
            data = json.loads(message)
            
            assert data["from"] == "sender", "Should identify sender"
            assert data["data"]["message"] == "Private message", "Should contain payload"
            print(f"✅ Direct message working: {data}")
            print(f"✅ ACK received: {ack_data}")
            return True
    except Exception as e:
        print(f"❌ Direct message failed: {e}")
        return False


async def test_rate_limiting():
    """Test rate limiting (1000 messages/minute)"""
    print("\nTest 5: Rate Limiting (1000 msg/min)")
    try:
        uri = "ws://localhost:11100/ws/rate-test"
        async with websockets.connect(uri) as websocket:
            # Skip welcome message
            await websocket.recv()
            
            # Send 1005 messages rapidly
            print("Sending 1005 messages rapidly...")
            rate_limit_hit = False
            
            for i in range(1005):
                await websocket.send(json.dumps({
                    "type": "ping"
                }))
                
                # Try to receive response
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    data = json.loads(message)
                    
                    if data.get("type") == "error" and data.get("error") == "rate_limit_exceeded":
                        rate_limit_hit = True
                        print(f"✅ Rate limit triggered at message {i+1}: {data}")
                        break
                except asyncio.TimeoutError:
                    continue
            
            assert rate_limit_hit, "Should hit rate limit after 1000 messages"
            return True
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False


async def test_invalid_target():
    """Test error handling for non-existent target"""
    print("\nTest 6: Invalid Target Error Handling")
    try:
        uri = "ws://localhost:11100/ws/error-test"
        async with websockets.connect(uri) as websocket:
            # Skip welcome message
            await websocket.recv()
            
            # Send to non-existent target
            await websocket.send(json.dumps({
                "type": "direct",
                "target": "non-existent-client",
                "payload": {"message": "Test"}
            }))
            
            # Should receive error
            message = await asyncio.wait_for(websocket.recv(), timeout=2)
            data = json.loads(message)
            
            assert data["type"] == "error", "Should receive error"
            assert data["error"] == "target_not_found", "Should indicate target not found"
            print(f"✅ Error handling working: {data}")
            return True
    except Exception as e:
        print(f"❌ Invalid target test failed: {e}")
        return False


async def test_unknown_message_type():
    """Test error handling for unknown message types"""
    print("\nTest 7: Unknown Message Type")
    try:
        uri = "ws://localhost:11100/ws/unknown-type-test"
        async with websockets.connect(uri) as websocket:
            # Skip welcome message
            await websocket.recv()
            
            # Send unknown message type
            await websocket.send(json.dumps({
                "type": "unknown-type",
                "payload": {"data": "test"}
            }))
            
            # Should receive error
            message = await asyncio.wait_for(websocket.recv(), timeout=2)
            data = json.loads(message)
            
            assert data["type"] == "error", "Should receive error"
            assert data["error"] == "unknown_message_type", "Should indicate unknown type"
            print(f"✅ Unknown type handling working: {data}")
            return True
    except Exception as e:
        print(f"❌ Unknown type test failed: {e}")
        return False


async def main():
    """Run all WebSocket tests"""
    print("=" * 60)
    print("WebSocket Test Suite for MCP Bridge")
    print("=" * 60)
    
    tests = [
        test_basic_connection,
        test_ping_pong,
        test_broadcast,
        test_direct_message,
        test_rate_limiting,
        test_invalid_target,
        test_unknown_message_type,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
            await asyncio.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"❌ {total - passed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
