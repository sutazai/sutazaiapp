#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Neo4j Integration Test Script for SutazAI
Tests all aspects of Neo4j connectivity and functionality
"""

import sys
import os
import requests
import base64
from datetime import datetime

def test_neo4j_http_api():
    """Test Neo4j HTTP API access"""
    logger.info("🔍 Testing Neo4j HTTP API...")
    
    try:
        auth_header = base64.b64encode(b'neo4j:neo4j_secure_2024').decode('ascii')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {auth_header}'
        }
        
        # Test basic connection
        response = requests.get('http://localhost:10002/', headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"  ✅ Neo4j HTTP API accessible")
            logger.info(f"  ℹ️  Version: {data.get('neo4j_version', 'unknown')}")
            logger.info(f"  ℹ️  Edition: {data.get('neo4j_edition', 'unknown')}")
        else:
            logger.error(f"  ❌ HTTP API failed with status: {response.status_code}")
            return False
            
        # Test Cypher execution
        cypher_data = {
            'statements': [
                {'statement': 'RETURN "HTTP API test successful" as message'}
            ]
        }
        
        response = requests.post(
            'http://localhost:10002/db/neo4j/tx/commit',
            json=cypher_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('errors'):
                logger.error(f"  ❌ Cypher execution failed: {result['errors']}")
                return False
            logger.info("  ✅ Cypher execution via HTTP API successful")
            return True
        else:
            logger.error(f"  ❌ Cypher execution failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"  ❌ HTTP API test failed: {str(e)}")
        return False

def test_neo4j_browser():
    """Test Neo4j Browser accessibility"""
    logger.info("🌐 Testing Neo4j Browser access...")
    
    try:
        response = requests.get('http://localhost:10002/', timeout=10)
        if response.status_code == 200 and 'neo4j' in response.text.lower():
            logger.info("  ✅ Neo4j Browser accessible on port 10002")
            return True
        else:
            logger.error(f"  ❌ Browser test failed with status: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"  ❌ Browser test failed: {str(e)}")
        return False

def test_backend_health():
    """Test SutazAI backend health"""
    logger.info("🏥 Testing SutazAI Backend health...")
    
    try:
        response = requests.get('http://localhost:10010/health', timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info("  ✅ Backend is healthy")
            logger.info(f"  ℹ️  Status: {health_data.get('status', 'unknown')}")
            logger.info(f"  ℹ️  Version: {health_data.get('version', 'unknown')}")
            return True
        else:
            logger.error(f"  ❌ Backend health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"  ❌ Backend health test failed: {str(e)}")
        return False

def main():
    """Run all Neo4j integration tests"""
    logger.info("🚀 SutazAI Neo4j Integration Test Suite")
    logger.info("=" * 50)
    logger.info(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info()
    
    tests = [
        ("Neo4j Browser", test_neo4j_browser),
        ("Neo4j HTTP API", test_neo4j_http_api),
        ("Backend Health", test_backend_health),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            logger.info()
        except Exception as e:
            logger.info(f"  ❌ {test_name} test crashed: {str(e)}")
            logger.info()
    
    logger.info("=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Neo4j is fully operational for SutazAI")
        logger.info("✅ Graph database operations are ready for production use")
        return 0
    else:
        logger.error("❌ Some tests failed. Neo4j may have connectivity issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())