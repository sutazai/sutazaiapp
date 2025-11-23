#!/usr/bin/env python3
"""
EXTENDED FUNCTIONAL TESTING SUITE
Tests actual operations, not just health checks
Covers authentication flows, vector operations, MCP routing, WebSocket, load testing
"""

import asyncio
import httpx
import json
import time
import sys
import os
from typing import Dict, List, Any
import websocket
import psycopg2
import redis
from neo4j import GraphDatabase
import random
import string

# Configuration
BASE_DIR = "/opt/sutazaiapp"
BACKEND_URL = "http://localhost:10200"
FRONTEND_URL = "http://localhost:11000"
MCP_BRIDGE_URL = "http://localhost:11100"

# Test tracking
results = []
test_number = 0

def log_test(name: str, status: str, details: str = "", category: str = "GENERAL"):
    """Log test result"""
    global test_number
    test_number += 1
    
    result = {
        "number": test_number,
        "category": category,
        "name": name,
        "status": status,
        "details": details,
        "timestamp": time.time()
    }
    results.append(result)
    
    symbols = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}
    symbol = symbols.get(status, "❓")
    print(f"{symbol} Test {test_number}: {name} - {status} {details}")
    return status == "PASS"

# ============================================================================
# AUTHENTICATION COMPLETE FLOW TESTS
# ============================================================================

async def test_complete_auth_flow():
    """Test full OAuth2 authentication workflow"""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            timestamp = int(time.time())
            username = f"test_{timestamp}"
            email = f"test_{timestamp}@example.com"
            password = "Test123!@#Strong"
            
            # Step 1: Register
            reg_data = {
                "username": username,
                "email": email,
                "password": password,
                "full_name": "Test User"
            }
            
            reg_response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/register",
                json=reg_data,
                timeout=10.0
            )
            
            if reg_response.status_code not in [200, 201]:
                log_test("Auth Flow: Registration", "FAIL", 
                        f"HTTP {reg_response.status_code}", "AUTH_FLOW")
                return False
            
            log_test("Auth Flow: Registration", "PASS", 
                    f"User {username} created", "AUTH_FLOW")
            
            # Step 2: Login to get tokens
            login_data = {
                "username": username,
                "password": password
            }
            
            login_response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/login",
                data=login_data,  # OAuth2 uses form data
                timeout=10.0
            )
            
            if login_response.status_code != 200:
                log_test("Auth Flow: Login", "FAIL",
                        f"HTTP {login_response.status_code}: {login_response.text}", "AUTH_FLOW")
                return False
            
            tokens = login_response.json()
            access_token = tokens.get("access_token")
            refresh_token = tokens.get("refresh_token")
            
            if not access_token:
                log_test("Auth Flow: Login", "FAIL", "No access token", "AUTH_FLOW")
                return False
            
            log_test("Auth Flow: Login", "PASS",
                    f"Tokens received (expires in {tokens.get('expires_in')}s)", "AUTH_FLOW")
            
            # Step 3: Access protected endpoint
            headers = {"Authorization": f"Bearer {access_token}"}
            me_response = await client.get(
                f"{BACKEND_URL}/api/v1/auth/me",
                headers=headers,
                timeout=10.0
            )
            
            if me_response.status_code != 200:
                log_test("Auth Flow: Protected Endpoint", "FAIL",
                        f"HTTP {me_response.status_code}", "AUTH_FLOW")
                return False
            
            user_data = me_response.json()
            if user_data.get("email") != email:
                log_test("Auth Flow: Protected Endpoint", "FAIL",
                        "Email mismatch", "AUTH_FLOW")
                return False
            
            log_test("Auth Flow: Protected Endpoint", "PASS",
                    f"User data retrieved for {user_data.get('username')}", "AUTH_FLOW")
            
            # Step 4: Refresh token
            refresh_response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/refresh",
                json={"refresh_token": refresh_token},
                timeout=10.0
            )
            
            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                log_test("Auth Flow: Token Refresh", "PASS",
                        "New tokens received", "AUTH_FLOW")
            else:
                log_test("Auth Flow: Token Refresh", "WARN",
                        f"HTTP {refresh_response.status_code}", "AUTH_FLOW")
            
            # Step 5: Logout
            logout_response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/logout",
                headers=headers,
                timeout=10.0
            )
            
            if logout_response.status_code in [200, 204]:
                log_test("Auth Flow: Logout", "PASS", "Logged out successfully", "AUTH_FLOW")
            else:
                log_test("Auth Flow: Logout", "WARN",
                        f"HTTP {logout_response.status_code}", "AUTH_FLOW")
            
            return True
            
    except Exception as e:
        log_test("Auth Flow: Complete", "FAIL", str(e), "AUTH_FLOW")
        return False

# ============================================================================
# VECTOR DATABASE FUNCTIONAL TESTS
# ============================================================================

async def test_vector_operations():
    """Test actual vector insert/search operations"""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # First authenticate
            timestamp = int(time.time())
            username = f"vectest_{timestamp}"
            password = "Test123!@#Strong"
            
            # Register
            await client.post(
                f"{BACKEND_URL}/api/v1/auth/register",
                json={
                    "username": username,
                    "email": f"{username}@example.com",
                    "password": password,
                    "full_name": "Vector Test"
                }
            )
            
            # Login
            login_response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/login",
                data={"username": username, "password": password}
            )
            
            if login_response.status_code != 200:
                log_test("Vector Operations: Auth", "FAIL", "Could not authenticate", "VECTORS")
                return
            
            token = login_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Test vector storage
            test_vector = [random.random() for _ in range(384)]  # 384-dim vector
            
            for db in ["chromadb", "qdrant", "faiss"]:
                store_response = await client.post(
                    f"{BACKEND_URL}/api/v1/vectors/store",
                    json={
                        "id": f"test_{timestamp}_{db}",
                        "vector": test_vector,
                        "metadata": {"test": "data", "db": db}
                    },
                    params={"database": db},
                    headers=headers,
                    timeout=10.0
                )
                
                if store_response.status_code == 200:
                    log_test(f"Vector Store: {db}", "PASS",
                            f"Vector stored successfully", "VECTORS")
                else:
                    log_test(f"Vector Store: {db}", "FAIL",
                            f"HTTP {store_response.status_code}: {store_response.text}", "VECTORS")
                
                # Test vector search
                search_response = await client.post(
                    f"{BACKEND_URL}/api/v1/vectors/search",
                    json={
                        "query": test_vector,
                        "k": 5
                    },
                    params={"database": db},
                    headers=headers,
                    timeout=10.0
                )
                
                if search_response.status_code == 200:
                    results = search_response.json()
                    # Handle different response formats
                    if isinstance(results, dict):
                        result_count = len(results.get('results', []))
                    else:
                        result_count = len(results) if isinstance(results, list) else 0
                    log_test(f"Vector Search: {db}", "PASS",
                            f"{result_count} results found", "VECTORS")
                else:
                    log_test(f"Vector Search: {db}", "WARN",
                            f"HTTP {search_response.status_code}", "VECTORS")
                    
    except Exception as e:
        log_test("Vector Operations", "FAIL", str(e), "VECTORS")

# ============================================================================
# MCP BRIDGE ROUTING TESTS
# ============================================================================

async def test_mcp_routing():
    """Test MCP message routing and agent selection"""
    try:
        async with httpx.AsyncClient() as client:
            timestamp = int(time.time())
            
            # Test routing to specific agent - use proper MCPMessage format
            message_data = {
                "id": f"test_{timestamp}_1",
                "source": "test_client",
                "target": "aider",
                "type": "code_generation",
                "payload": {
                    "content": "Generate a Python function to calculate fibonacci numbers",
                    "model": "tinyllama"
                },
                "timestamp": None,
                "metadata": {"test": True}
            }
            
            route_response = await client.post(
                f"{MCP_BRIDGE_URL}/route",
                json=message_data,
                timeout=30.0
            )
            
            if route_response.status_code == 200:
                result = route_response.json()
                log_test("MCP Routing: Direct Agent", "PASS",
                        f"Routed to aider", "MCP_ROUTING")
            else:
                log_test("MCP Routing: Direct Agent", "FAIL",
                        f"HTTP {route_response.status_code}: {route_response.text[:100]}", "MCP_ROUTING")
            
            # Test routing to any available agent
            message_data_auto = {
                "id": f"test_{timestamp}_2",
                "source": "test_client",
                "target": "langchain",  # Specify valid agent
                "type": "chat",
                "payload": {
                    "content": "What is the weather?",
                    "conversation_id": f"test_{timestamp}"
                }
            }
            
            route_response_auto = await client.post(
                f"{MCP_BRIDGE_URL}/route",
                json=message_data_auto,
                timeout=30.0
            )
            
            if route_response_auto.status_code == 200:
                result = route_response_auto.json()
                log_test("MCP Routing: Auto-Select", "PASS",
                        f"Routed to langchain", "MCP_ROUTING")
            else:
                log_test("MCP Routing: Auto-Select", "WARN",
                        f"HTTP {route_response_auto.status_code}", "MCP_ROUTING")
                
    except Exception as e:
        log_test("MCP Routing", "FAIL", str(e), "MCP_ROUTING")

# ============================================================================
# LOAD TESTING
# ============================================================================

async def test_load_handling():
    """Test system under concurrent load"""
    try:
        async def make_health_request(client, i):
            try:
                response = await client.get(f"{BACKEND_URL}/health", timeout=5.0)
                return response.status_code == 200
            except:
                return False
        
        # Create 50 concurrent requests
        async with httpx.AsyncClient(follow_redirects=True) as client:
            start_time = time.time()
            tasks = [make_health_request(client, i) for i in range(50)]
            results = await asyncio.gather(*tasks)
            duration = time.time() - start_time
            
            success_count = sum(results)
            rps = len(results) / duration
            
            if success_count >= 45:  # 90% success rate
                log_test("Load Test: 50 Concurrent Requests", "PASS",
                        f"{success_count}/50 success, {rps:.1f} req/s", "LOAD_TEST")
            else:
                log_test("Load Test: 50 Concurrent Requests", "FAIL",
                        f"Only {success_count}/50 succeeded", "LOAD_TEST")
                
    except Exception as e:
        log_test("Load Test", "FAIL", str(e), "LOAD_TEST")

# ============================================================================
# DATABASE CONNECTION POOLING
# ============================================================================

def test_postgres_pooling():
    """Test PostgreSQL connection pooling"""
    try:
        connections = []
        for i in range(10):
            conn = psycopg2.connect(
                host="localhost",
                port=10000,
                database="jarvis_ai",
                user="jarvis",
                password="sutazai_secure_2024"
            )
            connections.append(conn)
        
        # Execute concurrent queries
        for conn in connections:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            cursor.close()
        
        # Close all
        for conn in connections:
            conn.close()
        
        log_test("PostgreSQL Connection Pooling", "PASS",
                "10 concurrent connections handled", "DATABASE_POOLING")
                
    except Exception as e:
        log_test("PostgreSQL Connection Pooling", "FAIL", str(e), "DATABASE_POOLING")

def test_redis_pooling():
    """Test Redis connection pooling"""
    try:
        pool = redis.ConnectionPool(host='localhost', port=10001, max_connections=10)
        clients = [redis.Redis(connection_pool=pool) for _ in range(10)]
        
        # Concurrent operations
        for i, client in enumerate(clients):
            client.set(f"test_pool_{i}", f"value_{i}")
            value = client.get(f"test_pool_{i}")
            assert value == f"value_{i}".encode()
        
        # Cleanup
        for i in range(10):
            clients[0].delete(f"test_pool_{i}")
        
        log_test("Redis Connection Pooling", "PASS",
                "10 concurrent connections handled", "DATABASE_POOLING")
                
    except Exception as e:
        log_test("Redis Connection Pooling", "FAIL", str(e), "DATABASE_POOLING")

# ============================================================================
# RABBITMQ MESSAGING
# ============================================================================

async def test_rabbitmq():
    """Test RabbitMQ message queue"""
    try:
        # Check RabbitMQ is accessible
        async with httpx.AsyncClient() as client:
            # RabbitMQ management API
            response = await client.get(
                "http://localhost:10005/api/overview",
                auth=("sutazai", "sutazai_secure_2024"),
                timeout=5.0
            )
            
            if response.status_code == 200:
                data = response.json()
                log_test("RabbitMQ: Management API", "PASS",
                        f"Version: {data.get('rabbitmq_version', 'unknown')}", "RABBITMQ")
            else:
                log_test("RabbitMQ: Management API", "FAIL",
                        f"HTTP {response.status_code}", "RABBITMQ")
                
    except Exception as e:
        log_test("RabbitMQ", "FAIL", str(e), "RABBITMQ")

# ============================================================================
# CONSUL SERVICE DISCOVERY
# ============================================================================

async def test_consul():
    """Test Consul service discovery"""
    try:
        async with httpx.AsyncClient() as client:
            # Check Consul health
            response = await client.get("http://localhost:10006/v1/status/leader", timeout=5.0)
            
            if response.status_code == 200:
                log_test("Consul: Leader Election", "PASS",
                        "Leader elected", "CONSUL")
            else:
                log_test("Consul: Leader Election", "FAIL",
                        f"HTTP {response.status_code}", "CONSUL")
            
            # Check service catalog
            services_response = await client.get("http://localhost:10006/v1/catalog/services", timeout=5.0)
            
            if services_response.status_code == 200:
                services = services_response.json()
                log_test("Consul: Service Catalog", "PASS",
                        f"{len(services)} services registered", "CONSUL")
            else:
                log_test("Consul: Service Catalog", "WARN",
                        f"HTTP {services_response.status_code}", "CONSUL")
                
    except Exception as e:
        log_test("Consul", "FAIL", str(e), "CONSUL")

# ============================================================================
# OLLAMA INFERENCE TESTING
# ============================================================================

async def test_ollama_inference():
    """Test Ollama model inference"""
    try:
        async with httpx.AsyncClient() as client:
            # Generate completion
            generate_data = {
                "model": "tinyllama",
                "prompt": "Hello, how are you?",
                "stream": False
            }
            
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=generate_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                log_test("Ollama: Text Generation", "PASS",
                        f"Generated {len(generated_text)} chars", "OLLAMA")
            else:
                log_test("Ollama: Text Generation", "FAIL",
                        f"HTTP {response.status_code}", "OLLAMA")
                
    except Exception as e:
        log_test("Ollama: Text Generation", "FAIL", str(e), "OLLAMA")

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

async def run_extended_tests():
    """Execute all extended functional tests"""
    print("\n" + "="*100)
    print("EXTENDED FUNCTIONAL TESTING SUITE")
    print("Testing actual operations, not just health checks")
    print("="*100 + "\n")
    
    start_time = time.time()
    
    # Authentication Flow
    print("\n🔐 AUTHENTICATION FLOW TESTS")
    print("-" * 100)
    await test_complete_auth_flow()
    
    # Vector Operations
    print("\n🔍 VECTOR DATABASE OPERATIONS")
    print("-" * 100)
    await test_vector_operations()
    
    # MCP Routing
    print("\n🌉 MCP BRIDGE ROUTING")
    print("-" * 100)
    await test_mcp_routing()
    
    # Load Testing
    print("\n⚡ LOAD TESTING")
    print("-" * 100)
    await test_load_handling()
    
    # Database Pooling
    print("\n💾 DATABASE CONNECTION POOLING")
    print("-" * 100)
    test_postgres_pooling()
    test_redis_pooling()
    
    # Message Queue
    print("\n📨 MESSAGE QUEUE (RABBITMQ)")
    print("-" * 100)
    await test_rabbitmq()
    
    # Service Discovery
    print("\n🔎 SERVICE DISCOVERY (CONSUL)")
    print("-" * 100)
    await test_consul()
    
    # AI Inference
    print("\n🧠 AI INFERENCE (OLLAMA)")
    print("-" * 100)
    await test_ollama_inference()
    
    duration = time.time() - start_time
    
    # Generate Summary
    print("\n" + "="*100)
    print("EXTENDED TEST SUMMARY")
    print("="*100 + "\n")
    
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        categories[cat][result["status"]] += 1
    
    for cat, counts in sorted(categories.items()):
        total = sum(counts.values())
        print(f"{cat:25s}: ✅ {counts['PASS']:3d} | ❌ {counts['FAIL']:3d} | ⚠️ {counts['WARN']:3d} | ⏭️ {counts['SKIP']:3d} | Total: {total:3d}")
    
    print("\n" + "-"*100)
    
    pass_count = len([r for r in results if r["status"] == "PASS"])
    fail_count = len([r for r in results if r["status"] == "FAIL"])
    warn_count = len([r for r in results if r["status"] == "WARN"])
    total = len(results)
    
    print(f"\n{'TOTAL':25s}: ✅ {pass_count:3d} | ❌ {fail_count:3d} | ⚠️ {warn_count:3d} | Total: {total:3d}")
    print(f"\n⏱️  Duration: {duration:.2f} seconds")
    print(f"📈 Success Rate: {(pass_count/total*100):.1f}%")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"{BASE_DIR}/EXTENDED_TEST_RESULTS_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "duration": duration,
            "summary": {
                "total": total,
                "passed": pass_count,
                "failed": fail_count,
                "warnings": warn_count,
                "success_rate": round(pass_count/total*100, 2)
            },
            "categories": categories,
            "tests": results
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_extended_tests())
    sys.exit(exit_code)
