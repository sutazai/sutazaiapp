#!/usr/bin/env python3
"""
Test script to verify audit functionality works without stack overflow
"""
import asyncio
import aiohttp
import time
import json
from datetime import datetime

API_BASE = "http://localhost:8080/api/hygiene"

async def test_single_audit():
    """Test a single audit call"""
    async with aiohttp.ClientSession() as session:
        try:
            start_time = time.time()
            async with session.post(f"{API_BASE}/audit") as response:
                elapsed = time.time() - start_time
                data = await response.json()
                return {
                    "success": response.status == 200 and data.get("success", False),
                    "violations": data.get("violations_found", 0),
                    "message": data.get("message", ""),
                    "elapsed_ms": round(elapsed * 1000, 2)
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "is_stack_overflow": "stack" in str(e).lower()
            }

async def test_rapid_audits(count=20, delay_ms=50):
    """Test rapid audit calls to check for stack overflow"""
    print(f"\n🔍 Testing {count} rapid audit calls with {delay_ms}ms delay...")
    
    results = []
    stack_overflow_detected = False
    
    for i in range(count):
        result = await test_single_audit()
        results.append(result)
        
        if result.get("is_stack_overflow"):
            stack_overflow_detected = True
            print(f"❌ STACK OVERFLOW detected on call {i + 1}!")
            break
        
        if result["success"]:
            print(f"✅ Call {i + 1}: Success - {result['violations']} violations found in {result['elapsed_ms']}ms")
        else:
            print(f"❌ Call {i + 1}: Failed - {result.get('error', result.get('message', 'Unknown error'))}")
        
        if i < count - 1:  # Don't delay after last call
            await asyncio.sleep(delay_ms / 1000)
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    avg_time = sum(r.get("elapsed_ms", 0) for r in results if r.get("success")) / max(successful, 1)
    
    print(f"\n📊 Summary:")
    print(f"   Total calls: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Stack overflow: {'YES ❌' if stack_overflow_detected else 'NO ✅'}")
    print(f"   Avg response time: {avg_time:.2f}ms")
    
    return not stack_overflow_detected

async def test_concurrent_audits(count=10):
    """Test concurrent audit calls"""
    print(f"\n🔥 Testing {count} concurrent audit calls...")
    
    tasks = [test_single_audit() for _ in range(count)]
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start_time
    
    successful = 0
    stack_overflow_detected = False
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Call {i + 1}: Exception - {result}")
            if "stack" in str(result).lower():
                stack_overflow_detected = True
        elif isinstance(result, dict):
            if result.get("success"):
                successful += 1
                print(f"✅ Call {i + 1}: Success - {result['violations']} violations")
            elif result.get("is_stack_overflow"):
                stack_overflow_detected = True
                print(f"❌ Call {i + 1}: STACK OVERFLOW!")
            else:
                print(f"❌ Call {i + 1}: Failed - {result.get('error', 'Unknown error')}")
    
    print(f"\n📊 Concurrent test summary:")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Successful: {successful}/{count}")
    print(f"   Stack overflow: {'YES ❌' if stack_overflow_detected else 'NO ✅'}")
    
    return not stack_overflow_detected

async def test_dashboard_connection():
    """Test if dashboard can connect to backend"""
    print("\n🌐 Testing dashboard connectivity...")
    
    async with aiohttp.ClientSession() as session:
        # Test status endpoint
        try:
            async with session.get(f"{API_BASE}/status") as response:
                data = await response.json()
                print(f"✅ Status endpoint: OK - {data.get('totalViolations', 0)} total violations")
        except Exception as e:
            print(f"❌ Status endpoint: Failed - {e}")
            return False
        
        # Test metrics endpoint
        try:
            async with session.get("http://localhost:8080/api/system/metrics") as response:
                data = await response.json()
                print(f"✅ Metrics endpoint: OK - CPU: {data.get('cpu_usage', 0)}%")
        except Exception as e:
            print(f"❌ Metrics endpoint: Failed - {e}")
            return False
        
        # Test WebSocket endpoint
        try:
            ws_url = "ws://localhost:8080/ws"
            async with session.ws_connect(ws_url) as ws:
                # Send a test message
                await ws.send_json({"type": "ping"})
                
                # Wait for response with timeout
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        print(f"✅ WebSocket endpoint: OK - Received: {data.get('type', 'unknown')}")
                    await ws.close()
                except asyncio.TimeoutError:
                    print("⚠️  WebSocket: Connected but no response (might be normal)")
        except Exception as e:
            print(f"⚠️  WebSocket endpoint: {e} (non-critical)")
    
    return True

async def main():
    print("🧪 Hygiene Monitor - Stack Overflow Test Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend API: {API_BASE}")
    
    all_passed = True
    
    # Test 1: Dashboard connectivity
    if not await test_dashboard_connection():
        print("\n⚠️  Dashboard connectivity issues detected")
        all_passed = False
    
    # Test 2: Single audit
    print("\n📝 Testing single audit call...")
    result = await test_single_audit()
    if result["success"]:
        print(f"✅ Single audit: Success - {result['violations']} violations in {result['elapsed_ms']}ms")
    else:
        print(f"❌ Single audit: Failed - {result.get('error', 'Unknown error')}")
        all_passed = False
    
    # Test 3: Rapid sequential audits
    if not await test_rapid_audits(20, 50):
        all_passed = False
    
    # Test 4: Concurrent audits
    if not await test_concurrent_audits(10):
        all_passed = False
    
    # Final verdict
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL TESTS PASSED! No stack overflow detected.")
        print("🎉 The audit functionality is working correctly!")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)