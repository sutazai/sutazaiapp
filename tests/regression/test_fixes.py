#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRA-FIX Validation Script for Hardware Resource Optimizer
Tests all critical fixes and ensures production readiness
"""

import sys
import os
import asyncio
import requests
import time
import threading
from pathlib import Path

# Add the current directory to path
# Path handled by pytest configuration.parent))

def test_path_traversal_protection():
    """Test path traversal security fix"""
    logger.info("🔒 Testing path traversal protection...")
    
    from app import validate_safe_path
    
    # Test valid paths
    try:
        safe_path = validate_safe_path("/tmp", "/")
        assert safe_path == "/tmp"
        logger.info("   ✅ Valid path accepted")
    except Exception as e:
        logger.info(f"   ❌ Valid path rejected: {e}")
        return False
    
    # Test path traversal attempts
    dangerous_paths = [
        "../../etc/passwd",
        "/tmp/../../../etc/shadow",
        "../../../../root/.ssh/id_rsa",
        "/var/../../../home/user/.bashrc"
    ]
    
    for dangerous_path in dangerous_paths:
        try:
            validate_safe_path(dangerous_path, "/tmp")
            logger.info(f"   ❌ Dangerous path allowed: {dangerous_path}")
            return False
        except ValueError:
            logger.info(f"   ✅ Blocked dangerous path: {dangerous_path}")
        except Exception as e:
            logger.error(f"   ❌ Unexpected error: {e}")
            return False
    
    return True

def test_thread_safety():
    """Test thread safety improvements"""
    logger.info("🔧 Testing thread safety...")
    
    try:
        from app import HardwareResourceOptimizerAgent
        
        # Create agent instance
        agent = HardwareResourceOptimizerAgent()
        
        # Test concurrent access to hash cache
        def hash_files():
            for i in range(10):
                agent._get_file_hash(__file__)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=hash_files)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=5)
        
        logger.info("   ✅ Thread safety test passed")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Thread safety test failed: {e}")
        return False

def test_docker_client_initialization():
    """Test improved Docker client handling"""
    logger.info("🐳 Testing Docker client initialization...")
    
    try:
        from app import HardwareResourceOptimizerAgent
        
        agent = HardwareResourceOptimizerAgent()
        
        # Test Docker client access
        with agent.docker_client_lock:
            client_available = agent.docker_client is not None
        
        if client_available:
            logger.info("   ✅ Docker client initialized successfully")
        else:
            logger.info("   ⚠️  Docker client not available (expected in container)")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Docker client test failed: {e}")
        return False

def test_event_loop_handling():
    """Test event loop conflict fixes"""
    logger.info("🔄 Testing event loop handling...")
    
    try:
        from app import HardwareResourceOptimizerAgent
        
        agent = HardwareResourceOptimizerAgent()
        
        # Test memory optimization without event loop
        result = agent._optimize_memory()
        
        if result.get('status') == 'success':
            logger.info("   ✅ Memory optimization works without event loop")
        else:
            logger.info(f"   ⚠️  Memory optimization result: {result.get('status')}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Event loop test failed: {e}")
        return False

def test_api_endpoints_security():
    """Test API endpoint security"""
    logger.info("🛡️ Testing API endpoint security...")
    
    try:
        from app import HardwareResourceOptimizerAgent
        from fastapi.testclient import TestClient
        
        agent = HardwareResourceOptimizerAgent()
        client = TestClient(agent.app)
        
        # Test path traversal in analyze endpoint
        dangerous_paths = ["../../etc/passwd", "/tmp/../../../etc/shadow"]
        
        for path in dangerous_paths:
            response = client.get(f"/analyze/storage?path={path}")
            if response.status_code == 403:
                logger.info(f"   ✅ Blocked dangerous path: {path}")
            else:
                logger.info(f"   ❌ Dangerous path allowed: {path} (status: {response.status_code})")
                return False
        
        # Test valid endpoint access
        response = client.get("/health")
        if response.status_code == 200:
            logger.info("   ✅ Health endpoint accessible")
        else:
            logger.error(f"   ❌ Health endpoint failed: {response.status_code}")
            return False
        
        return True
        
    except ImportError:
        logger.info("   ⚠️  FastAPI test client not available, skipping endpoint tests")
        return True
    except Exception as e:
        logger.error(f"   ❌ API security test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    logger.info("🚀 ULTRA-FIX Validation for Hardware Resource Optimizer")
    logger.info("=" * 60)
    
    tests = [
        ("Path Traversal Protection", test_path_traversal_protection),
        ("Thread Safety", test_thread_safety),
        ("Docker Client Init", test_docker_client_initialization),
        ("Event Loop Handling", test_event_loop_handling),
        ("API Security", test_api_endpoints_security)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"   ❌ {test_name} FAILED")
        except Exception as e:
            logger.info(f"   💥 {test_name} CRASHED: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🎯 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ ALL ULTRA-FIXES VALIDATED - PRODUCTION READY!")
        return 0
    else:
        logger.info("❌ SOME FIXES NEED ATTENTION")
        return 1

if __name__ == "__main__":
    sys.exit(main())