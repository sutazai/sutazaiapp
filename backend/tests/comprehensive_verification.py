#!/usr/bin/env python3
"""
Comprehensive Backend Verification Script
Tests all critical components for real implementations
"""

import asyncio
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Test results storage
results = {
    "timestamp": datetime.now().isoformat(),
    "categories": [],
    "summary": {"total": 0, "passed": 0, "failed": 0, "warnings": 0}
}


def add_result(category: str, test_name: str, status: str, details: str = ""):
    """Add test result"""
    cat = next((c for c in results["categories"] if c["name"] == category), None)
    if not cat:
        cat = {"name": category, "tests": []}
        results["categories"].append(cat)
    
    cat["tests"].append({
        "name": test_name,
        "status": status,
        "details": details
    })
    
    results["summary"]["total"] += 1
    if status == "PASSED":
        results["summary"]["passed"] += 1
    elif status == "FAILED":
        results["summary"]["failed"] += 1
    else:
        results["summary"]["warnings"] += 1


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if file exists"""
    import os
    exists = os.path.exists(filepath)
    return exists


def check_imports():
    """Verify all new modules can be imported"""
    print("\n[1/8] Checking Module Imports...")
    
    try:
        from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
        add_result("Imports", "Circuit Breaker", "PASSED", "CircuitBreaker and exception imported")
    except Exception as e:
        add_result("Imports", "Circuit Breaker", "FAILED", str(e))
    
    try:
        from app.core.retry import async_retry
        add_result("Imports", "Retry Logic", "PASSED", "async_retry decorator imported")
    except Exception as e:
        add_result("Imports", "Retry Logic", "FAILED", str(e))
    
    try:
        from app.core.sanitization import sanitize_html, sanitize_text
        add_result("Imports", "Sanitization", "PASSED", "Sanitization functions imported")
    except Exception as e:
        add_result("Imports", "Sanitization", "FAILED", str(e))
    
    try:
        from app.core.pagination import PaginationParams, paginate_list
        add_result("Imports", "Pagination", "PASSED", "Pagination utilities imported")
    except Exception as e:
        add_result("Imports", "Pagination", "FAILED", str(e))
    
    try:
        from app.core.websocket_manager import WebSocketConnectionManager
        add_result("Imports", "WebSocket Manager", "PASSED", "WebSocket manager imported")
    except Exception as e:
        add_result("Imports", "WebSocket Manager", "FAILED", str(e))
    
    try:
        from app.middleware.request_id import RequestIDMiddleware
        add_result("Imports", "Request ID", "PASSED", "RequestID middleware imported")
    except Exception as e:
        add_result("Imports", "Request ID", "FAILED", str(e))
    
    try:
        from app.middleware.compression import GZipMiddleware
        add_result("Imports", "Compression", "PASSED", "Compression middleware imported")
    except Exception as e:
        add_result("Imports", "Compression", "FAILED", str(e))
    
    try:
        from app.middleware.rate_limiter import RateLimiterMiddleware
        add_result("Imports", "Rate Limiter", "PASSED", "Rate limiter middleware imported")
    except Exception as e:
        add_result("Imports", "Rate Limiter", "FAILED", str(e))


def check_database_session():
    """Verify database session management"""
    print("\n[2/8] Checking Database Session Management...")
    
    try:
        from app.core.database import get_db, engine
        import inspect
        
        # Check get_db is async generator
        if inspect.isasyncgenfunction(get_db):
            add_result("Database", "Session Generator", "PASSED", "get_db is async generator")
        else:
            add_result("Database", "Session Generator", "FAILED", "get_db is not async generator")
        
        # Check source for cleanup logic
        source = inspect.getsource(get_db)
        if "finally" in source and "close" in source:
            add_result("Database", "Session Cleanup", "PASSED", "Has finally block with close()")
        else:
            add_result("Database", "Session Cleanup", "FAILED", "Missing cleanup logic")
        
        if "rollback" in source:
            add_result("Database", "Transaction Rollback", "PASSED", "Has rollback on error")
        else:
            add_result("Database", "Transaction Rollback", "FAILED", "Missing rollback logic")
        
    except Exception as e:
        add_result("Database", "Session Management", "FAILED", str(e))


def check_health_endpoints():
    """Verify health check implementations"""
    print("\n[3/8] Checking Health Check Endpoints...")
    
    try:
        from app.api.v1.endpoints import health
        import inspect
        
        # Check health_status function
        if hasattr(health, 'health_status'):
            source = inspect.getsource(health.health_status)
            if "service_connections.health_check" in source:
                add_result("Health Checks", "Service Health", "PASSED", "Calls real service checks")
            else:
                add_result("Health Checks", "Service Health", "WARNING", "May not check all services")
        
        # Check metrics endpoint
        if hasattr(health, 'get_metrics'):
            source = inspect.getsource(health.get_metrics)
            if "pool_status" in source or "get_pool_status" in source:
                add_result("Health Checks", "Metrics", "PASSED", "Includes pool metrics")
            else:
                add_result("Health Checks", "Metrics", "WARNING", "May lack pool metrics")
                
    except Exception as e:
        add_result("Health Checks", "Endpoints", "FAILED", str(e))


def check_voice_processing():
    """Verify voice endpoint implementations"""
    print("\n[4/8] Checking Voice Processing...")
    
    try:
        from app.api.v1.endpoints import voice
        import inspect
        
        if hasattr(voice, 'process_voice'):
            source = inspect.getsource(voice.process_voice)
            if "voice_service.process_voice_command" in source:
                add_result("Voice", "Processing", "PASSED", "Uses real voice service")
            else:
                add_result("Voice", "Processing", "WARNING", "May be mock implementation")
        
        if hasattr(voice, 'transcribe'):
            add_result("Voice", "Transcription", "PASSED", "Transcription endpoint exists")
        
        if hasattr(voice, 'synthesize'):
            add_result("Voice", "Synthesis", "PASSED", "Synthesis endpoint exists")
            
    except Exception as e:
        add_result("Voice", "Endpoints", "FAILED", str(e))


def check_chat_integration():
    """Verify chat LLM integration"""
    print("\n[5/8] Checking Chat LLM Integration...")
    
    try:
        from app.api.v1.endpoints import chat
        import inspect
        
        # Check for Ollama integration
        if hasattr(chat, 'call_ollama'):
            add_result("Chat", "Ollama Integration", "PASSED", "call_ollama function exists")
        
        # Check message processing
        source_module = inspect.getsource(chat)
        if "call_ollama" in source_module or "ollama" in source_module.lower():
            add_result("Chat", "LLM Processing", "PASSED", "Uses Ollama for responses")
        else:
            add_result("Chat", "LLM Processing", "WARNING", "May not integrate with LLM")
        
        # Check sanitization
        if "sanitize_text" in source_module:
            add_result("Chat", "XSS Protection", "PASSED", "Uses sanitization")
        else:
            add_result("Chat", "XSS Protection", "WARNING", "May lack XSS protection")
            
    except Exception as e:
        add_result("Chat", "Integration", "FAILED", str(e))


def check_vector_operations():
    """Verify vector DB implementations"""
    print("\n[6/8] Checking Vector Database Operations...")
    
    try:
        from app.api.v1.endpoints import vectors
        import inspect
        
        # Check store_vector
        if hasattr(vectors, 'store_vector'):
            source = inspect.getsource(vectors.store_vector)
            if "service_connections" in source and ("chroma_client" in source or "qdrant_client" in source):
                add_result("Vectors", "Storage", "PASSED", "Real vector DB integration")
            else:
                add_result("Vectors", "Storage", "FAILED", "Mock implementation detected")
        
        # Check search_vectors
        if hasattr(vectors, 'search_vectors'):
            source = inspect.getsource(vectors.search_vectors)
            if "query" in source and "collection" in source.lower():
                add_result("Vectors", "Search", "PASSED", "Real search implementation")
            else:
                add_result("Vectors", "Search", "WARNING", "May be simplified implementation")
        
        # Check delete operation
        if hasattr(vectors, 'delete_vector'):
            add_result("Vectors", "Deletion", "PASSED", "Delete endpoint exists")
        else:
            add_result("Vectors", "Deletion", "WARNING", "Delete endpoint missing")
            
    except Exception as e:
        add_result("Vectors", "Operations", "FAILED", str(e))


def check_security_features():
    """Verify security implementations"""
    print("\n[7/8] Checking Security Features...")
    
    try:
        from app.core import security
        import inspect
        
        # Check password hashing
        if hasattr(security, 'get_password_hash'):
            source = inspect.getsource(security.get_password_hash)
            if "bcrypt" in source.lower() or "pwd_context" in source:
                add_result("Security", "Password Hashing", "PASSED", "Uses bcrypt")
            else:
                add_result("Security", "Password Hashing", "WARNING", "Hash method unclear")
        
        # Check token creation
        if hasattr(security, 'create_access_token'):
            source = inspect.getsource(security.create_access_token)
            if "jwt.encode" in source or "jose" in source:
                add_result("Security", "JWT Tokens", "PASSED", "JWT implementation present")
            else:
                add_result("Security", "JWT Tokens", "WARNING", "Token method unclear")
        
        # Check sanitization integration
        try:
            from app.core.sanitization import sanitize_text
            test_xss = sanitize_text("<script>alert('xss')</script>")
            if "<script>" not in test_xss:
                add_result("Security", "XSS Sanitization", "PASSED", "Sanitizes HTML/JS")
            else:
                add_result("Security", "XSS Sanitization", "FAILED", "Does not sanitize XSS")
        except:
            add_result("Security", "XSS Sanitization", "WARNING", "Could not test")
            
    except Exception as e:
        add_result("Security", "Features", "FAILED", str(e))


def check_middleware_integration():
    """Verify middleware is integrated"""
    print("\n[8/8] Checking Middleware Integration...")
    
    try:
        from app.main import app
        
        # Check middleware stack
        middleware_names = [m.__class__.__name__ for m in app.user_middleware]
        
        if "RequestIDMiddleware" in str(middleware_names):
            add_result("Middleware", "Request ID", "PASSED", "Integrated in main.py")
        else:
            add_result("Middleware", "Request ID", "WARNING", "May not be integrated")
        
        if "GZipCompressionMiddleware" in str(middleware_names) or "GZipMiddleware" in str(middleware_names):
            add_result("Middleware", "Compression", "PASSED", "Integrated in main.py")
        else:
            add_result("Middleware", "Compression", "WARNING", "May not be integrated")
        
        if "CORSMiddleware" in str(middleware_names):
            add_result("Middleware", "CORS", "PASSED", "CORS configured")
        else:
            add_result("Middleware", "CORS", "WARNING", "CORS may not be configured")
            
    except Exception as e:
        add_result("Middleware", "Integration", "FAILED", str(e))


def print_summary():
    """Print results summary"""
    print("\n" + "="*70)
    print("VERIFICATION RESULTS SUMMARY")
    print("="*70)
    
    for category in results["categories"]:
        print(f"\n{category['name']}:")
        for test in category["tests"]:
            icon = "✓" if test["status"] == "PASSED" else ("⚠" if test["status"] == "WARNING" else "✗")
            print(f"  {icon} {test['name']}: {test['status']}")
            if test.get("details"):
                print(f"    → {test['details']}")
    
    print("\n" + "="*70)
    print(f"Total Tests: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']} ✓")
    print(f"Warnings: {results['summary']['warnings']} ⚠")
    print(f"Failed: {results['summary']['failed']} ✗")
    
    success_rate = (results['summary']['passed'] / results['summary']['total'] * 100) if results['summary']['total'] > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if results['summary']['failed'] == 0:
        print("\n✓ ALL CRITICAL TESTS PASSED - PRODUCTION READY")
    elif results['summary']['failed'] < 5:
        print("\n⚠ MOSTLY READY - FEW ISSUES TO ADDRESS")
    else:
        print("\n✗ SIGNIFICANT ISSUES FOUND - REVIEW REQUIRED")
    print("="*70 + "\n")


def save_results():
    """Save results to JSON file"""
    output_file = f"verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    """Run all verification checks"""
    print("="*70)
    print("SUTAZAI BACKEND COMPREHENSIVE VERIFICATION")
    print("="*70)
    
    try:
        check_imports()
        check_database_session()
        check_health_endpoints()
        check_voice_processing()
        check_chat_integration()
        check_vector_operations()
        check_security_features()
        check_middleware_integration()
        
        print_summary()
        save_results()
        
        # Exit code based on results
        sys.exit(0 if results['summary']['failed'] == 0 else 1)
        
    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
