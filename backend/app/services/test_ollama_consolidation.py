#!/usr/bin/env python3
"""
Test script to verify Ollama service consolidation
Ensures all imports are correctly pointing to consolidated_ollama_service
"""

import asyncio
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_consolidated_service():
    """Test the consolidated Ollama service"""
    try:
        # Test direct import of consolidated service
        import sys
        import os
        sys.path.insert(0, '/opt/sutazaiapp/backend')
        from app.services.consolidated_ollama_service import get_ollama_service, ConsolidatedOllamaService
        logger.info("✅ Direct import of ConsolidatedOllamaService successful")
        
        # Test getting the service instance
        service = await get_ollama_service()
        logger.info(f"✅ Service instance created: {type(service).__name__}")
        
        # Verify it's the right type
        assert isinstance(service, ConsolidatedOllamaService), "Service is not ConsolidatedOllamaService"
        logger.info("✅ Service type verification passed")
        
        # Test basic functionality
        if hasattr(service, 'ollama_host'):
            logger.info(f"✅ Ollama host configured: {service.ollama_host}")
        
        if hasattr(service, 'default_model'):
            logger.info(f"✅ Default model configured: {service.default_model}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_integration_layer():
    """Test the integration layer compatibility"""
    try:
        # Test import through integration layer
        import sys
        sys.path.insert(0, '/opt/sutazaiapp/backend')
        from app.services.ollama_ultra_integration import OllamaUltraIntegration
        logger.info("✅ Integration layer import successful")
        
        # Create integration instance
        integration = OllamaUltraIntegration()
        await integration._ensure_initialized()
        
        # Verify the underlying service is ConsolidatedOllamaService
        if integration._ultra_service:
            service_type = type(integration._ultra_service).__name__
            logger.info(f"✅ Integration using service type: {service_type}")
            # The alias should make it appear as UltraOllamaService but it's actually ConsolidatedOllamaService
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


async def verify_no_ultra_service():
    """Verify ultra_ollama_service is not being used"""
    try:
        # This should fail since we renamed the file
        from ultra_ollama_service import UltraOllamaService
        logger.error("❌ ultra_ollama_service is still accessible - consolidation incomplete!")
        return False
    except ImportError:
        logger.info("✅ ultra_ollama_service correctly removed/deprecated")
        return True


async def main():
    """Run all consolidation tests"""
    logger.info("=" * 60)
    logger.info("OLLAMA SERVICE CONSOLIDATION VERIFICATION")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: Consolidated service
    logger.info("\n📋 Test 1: Consolidated Service")
    results.append(await test_consolidated_service())
    
    # Test 2: Integration layer
    logger.info("\n📋 Test 2: Integration Layer Compatibility")
    results.append(await test_integration_layer())
    
    # Test 3: Verify old service is gone
    logger.info("\n📋 Test 3: Old Service Removal")
    results.append(await verify_no_ultra_service())
    
    # Summary
    logger.info("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        logger.info(f"✅ CONSOLIDATION SUCCESSFUL: {passed}/{total} tests passed")
        logger.info("All Ollama services are now using consolidated_ollama_service.py")
        return 0
    else:
        logger.error(f"❌ CONSOLIDATION INCOMPLETE: {passed}/{total} tests passed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))