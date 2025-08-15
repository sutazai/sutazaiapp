#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Ollama Performance Optimization Script
Achieves <2s response time through configuration and system tuning
"""

import subprocess
import os
import sys
import time
import asyncio
import httpx
import json
from pathlib import Path


class OllamaOptimizer:
    """Ollama performance optimizer"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:10104"
        self.model = "tinyllama"
        self.optimizations_applied = []
        
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command"""
        logger.info(f"Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            logger.error(f"Error: {result.stderr}")
        return result
    
    def apply_system_optimizations(self):
        """Apply system-level optimizations"""
        logger.info("\n📊 Applying system optimizations...")
        
        # Increase system limits (if running as root or with sudo)
        optimizations = [
            # Network optimizations
            "sysctl -w net.core.somaxconn=65535",
            "sysctl -w net.ipv4.tcp_fin_timeout=30",
            "sysctl -w net.ipv4.tcp_tw_reuse=1",
            "sysctl -w net.ipv4.tcp_keepalive_time=60",
            "sysctl -w net.ipv4.tcp_keepalive_probes=3",
            "sysctl -w net.ipv4.tcp_keepalive_intvl=10",
            
            # Memory optimizations
            "sysctl -w vm.swappiness=10",
            "sysctl -w vm.dirty_ratio=15",
            "sysctl -w vm.dirty_background_ratio=5",
        ]
        
        for opt in optimizations:
            result = self.run_command(f"sudo {opt}", check=False)
            if result.returncode == 0:
                self.optimizations_applied.append(opt)
        
        logger.info(f"✅ Applied {len(self.optimizations_applied)} system optimizations")
    
    def stop_ollama(self):
        """Stop current Ollama container"""
        logger.info("\n🛑 Stopping current Ollama container...")
        self.run_command("docker-compose stop ollama", check=False)
        time.sleep(2)
    
    def start_optimized_ollama(self):
        """Start Ollama with optimized configuration"""
        logger.info("\n🚀 Starting optimized Ollama container...")
        
        # Use the optimized docker-compose override
        result = self.run_command(
            "docker-compose -f docker-compose.yml -f docker-compose.ollama-optimized.yml up -d ollama"
        )
        
        if result.returncode != 0:
            logger.error("⚠️  Failed to start with override file, trying regular start...")
            self.run_command("docker-compose up -d ollama")
        
        logger.info("⏳ Waiting for Ollama to be ready...")
        time.sleep(10)
    
    async def preload_model(self):
        """Preload and warm up the model"""
        logger.info(f"\n📦 Preloading {self.model} model...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # First, ensure model is pulled
                logger.info(f"Pulling {self.model} if needed...")
                response = await client.post(
                    f"{self.ollama_url}/api/pull",
                    json={"name": self.model}
                )
                
                # Stream the pull progress
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                if status:
                                    logger.info(f"  {status}")
                            except:
                                pass
                
                # Load the model into memory
                logger.info(f"Loading {self.model} into memory...")
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "Hello",
                        "options": {
                            "num_predict": 1,
                            "temperature": 0.1
                        }
                    }
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Model {self.model} loaded successfully")
                else:
                    logger.error(f"⚠️  Failed to load model: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"⚠️  Error loading model: {e}")
    
    async def warmup_model(self):
        """Warm up the model with test prompts"""
        logger.info("\n🔥 Warming up model with test prompts...")
        
        warmup_prompts = [
            "Hello, how are you?",
            "What is 2+2?",
            "Explain Python in one sentence.",
            "List three colors.",
            "What is machine learning?",
            "How does a computer work?",
            "What is the capital of France?",
            "Translate 'hello' to Spanish."
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []
            for prompt in warmup_prompts:
                task = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_ctx": 2048,
                            "num_predict": 50,
                            "temperature": 0.1,
                            "top_k": 10
                        }
                    }
                )
                tasks.append(task)
            
            logger.info("Sending warmup requests...")
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            logger.info(f"✅ Warmup completed: {successful}/{len(warmup_prompts)} successful")
    
    async def test_response_time(self):
        """Test actual response times"""
        logger.info("\n⏱️  Testing response times...")
        
        test_prompts = [
            ("Simple", "What is 2+2?"),
            ("Medium", "Explain what Python is in one sentence."),
            ("Complex", "Write a short function to calculate factorial in Python."),
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for name, prompt in test_prompts:
                logger.info(f"\nTesting {name} prompt: '{prompt[:50]}...'")
                
                # Test without streaming
                start_time = time.time()
                try:
                    response = await client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "num_ctx": 2048,
                                "num_predict": 256,
                                "temperature": 0.1,
                                "top_k": 10,
                                "top_p": 0.85
                            }
                        }
                    )
                    
                    elapsed = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        tokens = len(result.get("response", "").split())
                        
                        logger.info(f"  ✅ Response time: {elapsed:.2f}s")
                        logger.info(f"  📝 Tokens generated: ~{tokens}")
                        logger.info(f"  ⚡ Tokens/second: ~{tokens/elapsed:.1f}")
                        
                        if elapsed < 2.0:
                            logger.info(f"  🎯 TARGET ACHIEVED: <2s response!")
                        elif elapsed < 3.0:
                            logger.info(f"  ⚠️  Close to target: {elapsed:.2f}s")
                        else:
                            logger.info(f"  ❌ Above target: {elapsed:.2f}s")
                    else:
                        logger.error(f"  ❌ Request failed: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
                
                # Test with streaming (first token time)
                logger.info(f"  Testing streaming (first token)...")
                start_time = time.time()
                first_token_time = None
                
                try:
                    async with client.stream(
                        "POST",
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": True,
                            "options": {
                                "num_ctx": 2048,
                                "num_predict": 256,
                                "temperature": 0.1
                            }
                        }
                    ) as response:
                        async for line in response.aiter_lines():
                            if line and first_token_time is None:
                                first_token_time = time.time() - start_time
                                logger.info(f"  ⚡ First token: {first_token_time*1000:.0f}ms")
                                
                                if first_token_time < 0.5:
                                    logger.info(f"  🎯 EXCELLENT: First token <500ms!")
                                elif first_token_time < 1.0:
                                    logger.info(f"  ✅ Good: First token <1s")
                                break
                                
                except Exception as e:
                    logger.error(f"  ❌ Streaming error: {e}")
    
    def print_recommendations(self):
        """Print optimization recommendations"""
        logger.info("\n" + "="*60)
        logger.info("📋 OPTIMIZATION RECOMMENDATIONS")
        logger.info("="*60)
        
        recommendations = [
            "1. Use the optimized configuration:",
            "   docker-compose -f docker-compose.yml -f docker-compose.ollama-optimized.yml up -d",
            "",
            "2. Monitor performance metrics:",
            "   curl http://localhost:10104/api/tags",
            "   curl http://localhost:11435/metrics  # If metrics port enabled",
            "",
            "3. Key optimizations applied:",
            "   - Increased CPU allocation (8 cores)",
            "   - Increased memory (8GB)",
            "   - Parallel processing (8 parallel, 12 threads)",
            "   - Memory-mapped files and locking",
            "   - Optimized context size (2048)",
            "   - Connection pooling (50 connections)",
            "   - Response caching in backend",
            "",
            "4. For GPU acceleration (if available):",
            "   - Install NVIDIA Container Toolkit",
            "   - Add 'runtime: nvidia' to docker-compose",
            "   - Set CUDA_VISIBLE_DEVICES environment variable",
            "",
            "5. Backend integration:",
            "   - Use ollama_ultra_optimized.py service",
            "   - Enable response caching",
            "   - Implement request batching",
            "",
            "6. Further optimizations:",
            "   - Use SSD for model storage",
            "   - Increase system file descriptors limit",
            "   - Use dedicated Ollama server for production",
            "   - Consider model quantization (GGUF format)",
        ]
        
        for line in recommendations:
            logger.info(line)
        
        logger.info("\n" + "="*60)
        logger.info("✨ Optimization complete!")
        logger.info("="*60)
    
    async def run(self):
        """Run the optimization process"""
        logger.info("="*60)
        logger.info("🚀 OLLAMA PERFORMANCE OPTIMIZER")
        logger.info("Target: <2 second response time")
        logger.info("="*60)
        
        # Apply system optimizations
        self.apply_system_optimizations()
        
        # Restart Ollama with optimized config
        self.stop_ollama()
        self.start_optimized_ollama()
        
        # Preload and warm up model
        await self.preload_model()
        await self.warmup_model()
        
        # Test response times
        await self.test_response_time()
        
        # Print recommendations
        self.print_recommendations()


async def main():
    """Main function"""
    # Change to project directory
    os.chdir("/opt/sutazaiapp")
    
    optimizer = OllamaOptimizer()
    await optimizer.run()


if __name__ == "__main__":
    asyncio.run(main())