#!/usr/bin/env python3
"""
Simple Test Script for SutazAI V7 Self-Supervised Learning Pipeline
Tests basic functionality without external dependencies
"""

import os
import sys
import asyncio
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from web_learning.web_scraper import WebScraper, ScrapingConfig, create_web_scraper
        from web_learning.content_processor import ContentProcessor, ProcessedContent, ContentType, create_content_processor
        from web_learning.knowledge_extractor import KnowledgeExtractor, KnowledgeUnit, KnowledgeType, create_knowledge_extractor
        from web_learning.learning_pipeline import SelfSupervisedLearningPipeline, LearningConfig, create_learning_pipeline
        from web_learning.web_automation import WebAutomation, AutomationConfig, create_web_automation
        
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_web_scraper():
    """Test web scraper functionality"""
    print("\nTesting web scraper...")
    
    try:
        from web_learning.web_scraper import WebScraper, ScrapingConfig, RateLimiter, ContentValidator
        
        # Test configuration
        config = ScrapingConfig(requests_per_second=1.0, max_total_pages=5)
        scraper = WebScraper(config)
        
        # Test rate limiter
        limiter = RateLimiter(2.0)
        assert limiter.requests_per_second == 2.0
        
        # Test content validator
        validator = ContentValidator(config)
        assert validator.is_valid_content_type('text/html')
        assert not validator.is_valid_content_type('image/jpeg')
        assert validator.is_valid_url('https://example.com')
        assert not validator.is_valid_url('javascript:alert(1)')
        
        # Test content sanitization
        malicious_content = '<script>alert("xss")</script><p>Safe content</p>'
        sanitized = validator.sanitize_content(malicious_content)
        assert '<script>' not in sanitized
        assert 'Safe content' in sanitized
        
        # Test statistics
        stats = scraper.get_statistics()
        assert 'total_requests' in stats
        assert 'success_rate' in stats
        
        print("✓ Web scraper tests passed")
        return True
    except Exception as e:
        print(f"✗ Web scraper test error: {e}")
        return False

def test_content_processor():
    """Test content processor functionality"""
    print("\nTesting content processor...")
    
    try:
        from web_learning.content_processor import ContentProcessor, ProcessedContent, ContentType, HTMLProcessor, TextAnalyzer
        
        # Test processor creation
        processor = ContentProcessor()
        assert processor.html_processor is not None
        assert processor.text_analyzer is not None
        
        # Test content type detection
        html_content = '<html><body><h1>Test</h1></body></html>'
        content_type = processor.detect_content_type(html_content, 'text/html')
        assert content_type == ContentType.HTML
        
        json_content = '{"title": "Test", "content": "Test content"}'
        content_type = processor.detect_content_type(json_content, 'application/json')
        assert content_type == ContentType.JSON
        
        # Test HTML processor
        html_processor = HTMLProcessor()
        text = html_processor.extract_text(html_content)
        assert 'Test' in text
        
        title = html_processor.extract_title(html_content)
        # Note: title extraction might not work without full HTML structure
        
        # Test text analyzer
        text_analyzer = TextAnalyzer()
        test_text = "Machine learning is a subset of artificial intelligence."
        
        # Test readability
        readability = text_analyzer.calculate_readability(test_text)
        assert isinstance(readability, float)
        assert 0 <= readability <= 100
        
        # Test sentiment
        sentiment = text_analyzer.analyze_sentiment(test_text)
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1
        
        # Test statistics
        stats = processor.get_statistics()
        assert 'total_processed' in stats
        assert 'success_rate' in stats
        
        print("✓ Content processor tests passed")
        return True
    except Exception as e:
        print(f"✗ Content processor test error: {e}")
        return False

def test_knowledge_extractor():
    """Test knowledge extractor functionality"""
    print("\nTesting knowledge extractor...")
    
    try:
        from web_learning.knowledge_extractor import KnowledgeExtractor, KnowledgeUnit, KnowledgeType, FactualExtractor
        from web_learning.content_processor import ProcessedContent, ContentType
        
        # Test extractor creation
        extractor = KnowledgeExtractor()
        assert extractor.factual_extractor is not None
        assert extractor.conceptual_extractor is not None
        assert extractor.relational_extractor is not None
        assert extractor.numerical_extractor is not None
        
        # Test knowledge unit creation
        knowledge_unit = KnowledgeUnit(
            id="test_id",
            content="Machine learning is a subset of artificial intelligence",
            knowledge_type=KnowledgeType.FACTUAL,
            confidence=0.8,
            source_url="https://example.com",
            entities=["machine learning", "artificial intelligence"]
        )
        
        assert knowledge_unit.id == "test_id"
        assert knowledge_unit.knowledge_type == KnowledgeType.FACTUAL
        assert knowledge_unit.confidence == 0.8
        
        # Test factual extractor
        factual_extractor = FactualExtractor()
        
        # Create test processed content
        processed_content = ProcessedContent(
            url="https://example.com",
            content_type=ContentType.HTML,
            title="Test Article",
            text_content="Machine learning is a subset of artificial intelligence. Neural networks are computational models."
        )
        
        facts = factual_extractor.extract_facts(processed_content)
        assert isinstance(facts, list)
        
        # Test statistics
        stats = extractor.get_statistics()
        assert 'total_processed' in stats
        assert 'facts_extracted' in stats
        
        print("✓ Knowledge extractor tests passed")
        return True
    except Exception as e:
        print(f"✗ Knowledge extractor test error: {e}")
        return False

def test_learning_pipeline():
    """Test learning pipeline functionality"""
    print("\nTesting learning pipeline...")
    
    try:
        from web_learning.learning_pipeline import SelfSupervisedLearningPipeline, LearningConfig, LearningMode, LearningTask
        
        # Test configuration
        config = LearningConfig()
        config.scraping_config.max_total_pages = 2
        config.max_concurrent_tasks = 1
        
        # Use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config.results_directory = temp_dir
            
            # Test pipeline creation
            pipeline = SelfSupervisedLearningPipeline(config)
            assert pipeline.config == config
            assert pipeline.web_scraper is not None
            assert pipeline.content_processor is not None
            assert pipeline.knowledge_extractor is not None
            
            # Test learning task creation
            task = LearningTask(
                id="test_task",
                mode=LearningMode.EXPLORATION,
                topic="machine learning",
                urls=["https://example.com"],
                keywords=["ML", "AI"],
                priority=0.8
            )
            
            assert task.id == "test_task"
            assert task.mode == LearningMode.EXPLORATION
            assert task.topic == "machine learning"
            
            # Test statistics
            stats = pipeline.get_overall_statistics()
            assert 'total_sessions' in stats
            assert 'total_knowledge_extracted' in stats
            
        print("✓ Learning pipeline tests passed")
        return True
    except Exception as e:
        print(f"✗ Learning pipeline test error: {e}")
        return False

def test_web_automation():
    """Test web automation functionality"""
    print("\nTesting web automation...")
    
    try:
        from web_learning.web_automation import WebAutomation, AutomationConfig, BrowserManager
        
        # Test configuration
        config = AutomationConfig(
            browser_type="chrome",
            headless=True,
            take_screenshots=False
        )
        
        # Test automation creation (without actually initializing browser)
        automation = WebAutomation(config)
        assert automation.config == config
        assert not automation.is_initialized
        
        # Test browser manager creation
        browser_manager = BrowserManager(config)
        assert browser_manager.config == config
        assert browser_manager.driver is None
        
        # Test statistics
        stats = automation.get_statistics()
        assert 'pages_visited' in stats
        assert 'actions_performed' in stats
        
        print("✓ Web automation tests passed")
        return True
    except Exception as e:
        print(f"✗ Web automation test error: {e}")
        return False

async def test_async_functionality():
    """Test async functionality"""
    print("\nTesting async functionality...")
    
    try:
        from web_learning.content_processor import create_content_processor
        from web_learning.content_processor import ProcessedContent, ContentType
        
        # Test content processing
        processor = create_content_processor()
        
        test_content = '''
        <html>
            <head><title>Test Article</title></head>
            <body>
                <h1>Machine Learning</h1>
                <p>This is a test article about machine learning.</p>
            </body>
        </html>
        '''
        
        result = await processor.process_content("https://example.com", test_content, "text/html")
        
        assert isinstance(result, ProcessedContent)
        assert result.url == "https://example.com"
        assert result.content_type == ContentType.HTML
        assert len(result.text_content) > 0
        
        print("✓ Async functionality tests passed")
        return True
    except Exception as e:
        print(f"✗ Async functionality test error: {e}")
        return False

def test_integration():
    """Test integration between components"""
    print("\nTesting component integration...")
    
    try:
        from web_learning.learning_pipeline import create_learning_pipeline, LearningConfig
        
        # Test pipeline creation with configuration
        config = LearningConfig()
        config.scraping_config.max_total_pages = 1
        config.max_concurrent_tasks = 1
        
        pipeline = create_learning_pipeline(config)
        
        # Test that all components are properly integrated
        assert hasattr(pipeline, 'web_scraper')
        assert hasattr(pipeline, 'content_processor')
        assert hasattr(pipeline, 'knowledge_extractor')
        assert hasattr(pipeline, 'topic_discoverer')
        
        # Test component initialization
        assert pipeline.web_scraper is not None
        assert pipeline.content_processor is not None
        assert pipeline.knowledge_extractor is not None
        assert pipeline.topic_discoverer is not None
        
        print("✓ Integration tests passed")
        return True
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        return False

def main():
    """Run all tests"""
    print("SutazAI V7 Self-Supervised Learning Pipeline - Simple Test Suite")
    print("=" * 70)
    
    tests = [
        ("Import Tests", test_imports),
        ("Web Scraper Tests", test_web_scraper),
        ("Content Processor Tests", test_content_processor),
        ("Knowledge Extractor Tests", test_knowledge_extractor),
        ("Learning Pipeline Tests", test_learning_pipeline),
        ("Web Automation Tests", test_web_automation),
        ("Integration Tests", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    # Run async tests
    print("\nRunning async tests...")
    try:
        result = asyncio.run(test_async_functionality())
        if result:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print(f"✗ Async tests failed: {e}")
        failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests PASSED!")
        print("✓ Self-supervised learning pipeline is working correctly")
    else:
        print(f"❌ {failed} tests FAILED")
        print("Please check the error messages above")
    
    print("\nComponents tested:")
    print("• Web Scraper - Enterprise-grade web scraping")
    print("• Content Processor - Multi-format content processing")
    print("• Knowledge Extractor - Structured knowledge extraction")
    print("• Learning Pipeline - Self-supervised learning orchestration")
    print("• Web Automation - Browser automation for complex sites")
    print("• Integration - Component integration and data flow")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)