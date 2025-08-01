#!/usr/bin/env python3
"""
Test script for native Ollama API integration
Shows how to use Ollama directly without any compatibility layers
"""

import httpx
import json
import asyncio

async def test_ollama_native():
    """Test all native Ollama API endpoints"""
    base_url = "http://localhost:11434"
    
    print("🧪 Testing Native Ollama API")
    print("=" * 50)
    
    # Test 1: Check if Ollama is running
    print("\n1️⃣ Testing Ollama availability...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                print("✅ Ollama is running!")
                print(f"📦 Available models: {[m['name'] for m in models.get('models', [])]}")
            else:
                print("❌ Ollama not responding")
                return
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return
    
    # Test 2: Generate text
    print("\n2️⃣ Testing text generation...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": "tinyllama:latest",
                    "prompt": "What is the meaning of life? Answer in one sentence.",
                    "stream": False
                }
            )
            if response.status_code == 200:
                result = response.json()
                print("✅ Generation successful!")
                print(f"📝 Response: {result.get('response', '')[:200]}...")
            else:
                print(f"❌ Generation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error generating text: {e}")
    
    # Test 3: Chat completion
    print("\n3️⃣ Testing chat API...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": "tinyllama:latest",
                    "messages": [
                        {"role": "user", "content": "Hello! What are you?"},
                        {"role": "assistant", "content": "I am TinyLlama, a small but capable language model."},
                        {"role": "user", "content": "What can you help me with?"}
                    ],
                    "stream": False
                }
            )
            if response.status_code == 200:
                result = response.json()
                print("✅ Chat successful!")
                message = result.get('message', {})
                print(f"🤖 Assistant: {message.get('content', '')[:200]}...")
            else:
                print(f"❌ Chat failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error in chat: {e}")
    
    # Test 4: Embeddings
    print("\n4️⃣ Testing embeddings...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/api/embeddings",
                json={
                    "model": "tinyllama:latest",
                    "prompt": "Hello world"
                }
            )
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                print("✅ Embeddings successful!")
                print(f"📊 Embedding dimension: {len(embedding)}")
                print(f"📈 First 5 values: {embedding[:5]}")
            else:
                print(f"❌ Embeddings failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting embeddings: {e}")
    
    # Test 5: Model information
    print("\n5️⃣ Testing model info...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/show",
                json={"name": "tinyllama:latest"}
            )
            if response.status_code == 200:
                info = response.json()
                print("✅ Model info retrieved!")
                print(f"📋 Model details: {info.get('details', {}).get('parameter_size', 'Unknown')}")
            else:
                print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
    
    print("\n" + "=" * 50)
    print("✨ Native Ollama API test complete!")
    print("\n💡 Key Points:")
    print("  - No OpenAI compatibility layer needed")
    print("  - Direct communication with Ollama")
    print("  - 100% local, 100% private")
    print("  - All agent code uses these native APIs")

if __name__ == "__main__":
    asyncio.run(test_ollama_native())