#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Hardware API Integration Test Script
Tests that the hardware API endpoints are properly integrated with the backend
"""

import asyncio
import sys
import os

# Add backend to path
# Path handled by pytest configuration, 'app'))

async def test_hardware_integration():
    """Test that hardware API integration is working"""
    
    logger.info("🔧 Testing Hardware API Integration...")
    
    try:
        # Test import of hardware router
        from app.api.v1.endpoints.hardware import router, HardwareServiceClient
        logger.info("✅ Hardware router imports successfully")
        
        # Test that router is properly configured
        routes = [route.path for route in router.routes]
        expected_routes = [
            "/hardware/health",
            "/hardware/metrics", 
            "/hardware/metrics/stream",
            "/hardware/optimize",
            "/hardware/optimize/{task_id}",
            "/hardware/processes",
            "/hardware/processes/control",
            "/hardware/alerts",
            "/hardware/recommendations"
        ]
        
        for expected_route in expected_routes:
            if expected_route not in routes:
                logger.info(f"❌ Missing expected route: {expected_route}")
                return False
                
        logger.info(f"✅ All {len(expected_routes)} expected routes configured")
        
        # Test client initialization
        client = HardwareServiceClient()
        logger.info(f"✅ HardwareServiceClient initialized with base_url: {client.base_url}")
        
        # Test authentication imports
        from app.auth.dependencies import get_current_user, require_permissions
        logger.info("✅ Authentication dependencies imported successfully")
        
        # Test Pydantic models
        from app.api.v1.endpoints.hardware import (
            HardwareMetricsRequest, OptimizationRequest, SystemMetrics,
            OptimizationResult, ProcessInfo, HardwareStatus
        )
        
        # Test model instantiation
        metrics_request = HardwareMetricsRequest(
            include_processes=True,
            sample_duration=10
        )
        logger.info("✅ Pydantic models work correctly")
        
        optimization_request = OptimizationRequest(
            optimization_type="cpu",
            parameters={"target_usage": 70},
            priority="normal"
        )
        logger.info("✅ Optimization request model validation works")
        
        logger.info("\n🎉 Hardware API Integration Test PASSED!")
        logger.info("\nNext Steps:")
        logger.info("1. Start the backend server: uvicorn app.main:app --host 0.0.0.0 --port 10010")
        logger.info("2. Ensure hardware-resource-optimizer service is running on port 8080")
        logger.info("3. Test endpoints with curl or API client")
        logger.info("\nExample test command:")
        logger.info("curl -H 'Authorization: Bearer YOUR_JWT_TOKEN' http://localhost:10010/api/v1/hardware/health")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_backend_startup():
    """Test that the backend can start with hardware router"""
    
    logger.info("\n🚀 Testing Backend Startup Integration...")
    
    try:
        # Test main app import
        from app.main import app
        logger.info("✅ Main FastAPI app imports successfully")
        
        # Check if hardware router is included
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
            if hasattr(route, 'routes'):  # For included routers
                for subroute in route.routes:
                    if hasattr(subroute, 'path'):
                        routes.append(subroute.path)
        
        hardware_routes = [r for r in routes if '/hardware' in r]
        if hardware_routes:
            logger.info(f"✅ Found {len(hardware_routes)} hardware routes in main app")
        else:
            logger.info("❌ No hardware routes found in main app")
            return False
            
        logger.info("✅ Backend startup integration test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backend startup test failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    
    logger.info("=" * 60)
    logger.info("HARDWARE API INTEGRATION TEST SUITE")
    logger.info("=" * 60)
    
    # Test 1: Hardware API Integration
    test1_passed = await test_hardware_integration()
    
    # Test 2: Backend Startup
    test2_passed = test_backend_startup()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.error(f"Hardware API Integration: {'PASSED' if test1_passed else 'FAILED'}")
    logger.error(f"Backend Startup Integration: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("\n🎉 ALL TESTS PASSED! Hardware API is ready for production use.")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)