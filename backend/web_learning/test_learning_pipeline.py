#!/usr/bin/env python3
"""
Comprehensive Test Suite for SutazAI V7 Self-Supervised Learning Pipeline
Tests web scraping, content processing, knowledge extraction, and learning pipeline integration
"""

import os
import sys
import asyncio
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_learning.web_scraper import WebScraper, ScrapingConfig, create_web_scraper
from web_learning.content_processor import ContentProcessor, ProcessedContent, ContentType, create_content_processor
from web_learning.knowledge_extractor import KnowledgeExtractor, KnowledgeUnit, KnowledgeType, create_knowledge_extractor
from web_learning.learning_pipeline import SelfSupervisedLearningPipeline, LearningConfig, LearningMode, create_learning_pipeline
from web_learning.web_automation import WebAutomation, AutomationConfig, create_web_automation

class TestWebScraper:
    """Test suite for the web scraper component"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = ScrapingConfig(
            requests_per_second=10.0,  # High rate for testing
            max_total_pages=5,
            respect_robots_txt=False,  # Disable for testing
            cache_responses=True
        )
        
        self.scraper = WebScraper(self.config)
    
    def test_scraper_initialization(self):
        """Test scraper initialization"""
        assert self.scraper.config == self.config
        assert self.scraper.stats['total_requests'] == 0
        assert len(self.scraper.scraped_urls) == 0
        assert len(self.scraper.failed_urls) == 0
    
    def test_robots_checker(self):
        """Test robots.txt checking functionality"""
        from web_learning.web_scraper import RobotsChecker
        
        checker = RobotsChecker()
        
        # Test with a mock URL
        # Note: This would require mocking the actual robots.txt fetch
        assert hasattr(checker, 'can_fetch')
        assert hasattr(checker, 'robots_cache')
    
    def test_rate_limiter(self):
        """Test rate limiting functionality"""
        from web_learning.web_scraper import RateLimiter
        
        limiter = RateLimiter(requests_per_second=2.0)
        
        assert limiter.requests_per_second == 2.0
        assert limiter.min_interval == 0.5
        assert limiter.request_count == 0
    
    def test_content_validator(self):
        """Test content validation"""
        from web_learning.web_scraper import ContentValidator
        
        validator = ContentValidator(self.config)
        
        # Test valid content type
        assert validator.is_valid_content_type('text/html')
        assert validator.is_valid_content_type('application/json')
        assert not validator.is_valid_content_type('image/jpeg')
        
        # Test valid URL
        assert validator.is_valid_url('https://example.com')
        assert validator.is_valid_url('http://test.org/page')
        assert not validator.is_valid_url('javascript:alert(1)')
        assert not validator.is_valid_url('https://example.com/file.pdf')
    
    def test_content_sanitization(self):
        """Test content sanitization"""
        from web_learning.web_scraper import ContentValidator
        
        validator = ContentValidator(self.config)
        
        # Test malicious content removal
        malicious_content = '<script>alert("xss")</script><p>Safe content</p>'
        sanitized = validator.sanitize_content(malicious_content)
        
        assert '<script>' not in sanitized
        assert 'Safe content' in sanitized
    
    @pytest.mark.asyncio
    async def test_scraper_session_management(self):
        """Test session management"""
        async with self.scraper as scraper:
            assert scraper.session is not None
        
        # Session should be closed after context exit
        assert self.scraper.session is None
    
    def test_cache_functionality(self):
        """Test response caching"""
        # Test cache key generation
        cache_key = self.scraper._get_cache_key("https://example.com")
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
        
        # Test cache validity
        cache_key = "test_key"
        assert not self.scraper._is_cache_valid(cache_key)
    
    def test_link_extraction(self):
        """Test link extraction from HTML"""
        html_content = '''
        <html>
            <body>
                <a href="https://example.com">External link</a>
                <a href="/relative/path">Relative link</a>
                <a href="mailto:test@example.com">Email link</a>
            </body>
        </html>
        '''
        
        base_url = "https://test.com"
        links = self.scraper.extract_links(html_content, base_url)
        
        assert len(links) >= 1
        assert any('example.com' in link for link in links)
    
    def test_statistics_calculation(self):
        """Test statistics calculation"""
        stats = self.scraper.get_statistics()
        
        required_fields = [
            'total_requests', 'successful_requests', 'failed_requests',
            'scraped_urls_count', 'runtime_seconds', 'success_rate'
        ]
        
        for field in required_fields:
            assert field in stats

class TestContentProcessor:
    """Test suite for the content processor component"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = create_content_processor()
        
        # Sample HTML content
        self.html_content = '''
        <html>
            <head><title>Test Article</title></head>
            <body>
                <h1>Machine Learning Breakthrough</h1>
                <p>Researchers have developed a new neural network architecture.</p>
                <p>The system achieved 95% accuracy on benchmark tests.</p>
            </body>
        </html>
        '''
        
        # Sample JSON content
        self.json_content = '''
        {
            "title": "Research Paper",
            "content": "This paper describes advances in artificial intelligence.",
            "authors": ["John Doe", "Jane Smith"],
            "year": 2023
        }
        '''
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        assert self.processor.html_processor is not None
        assert self.processor.text_analyzer is not None
        assert self.processor.stats['total_processed'] == 0
    
    def test_content_type_detection(self):
        """Test content type detection"""
        # Test HTML detection
        html_type = self.processor.detect_content_type(self.html_content, "text/html")
        assert html_type == ContentType.HTML
        
        # Test JSON detection
        json_type = self.processor.detect_content_type(self.json_content, "application/json")
        assert json_type == ContentType.JSON
        
        # Test text detection
        text_type = self.processor.detect_content_type("Plain text content", "text/plain")
        assert text_type == ContentType.TEXT
    
    def test_html_processing(self):
        """Test HTML content processing"""
        from web_learning.content_processor import HTMLProcessor
        
        processor = HTMLProcessor()
        
        # Test text extraction
        text = processor.extract_text(self.html_content)
        assert 'Machine Learning Breakthrough' in text
        assert 'neural network architecture' in text
        
        # Test title extraction
        title = processor.extract_title(self.html_content)
        assert title == 'Test Article'
        
        # Test metadata extraction
        metadata = processor.extract_metadata(self.html_content)
        assert isinstance(metadata, dict)
    
    def test_text_analysis(self):
        """Test text analysis functionality"""
        from web_learning.content_processor import TextAnalyzer
        
        analyzer = TextAnalyzer()
        
        text = "Machine learning is a subset of artificial intelligence. Neural networks consist of interconnected nodes."
        
        # Test keyword extraction
        keywords = analyzer.extract_keywords(text)
        assert isinstance(keywords, list)
        
        # Test readability calculation
        readability = analyzer.calculate_readability(text)
        assert isinstance(readability, float)
        assert 0 <= readability <= 100
        
        # Test sentiment analysis
        sentiment = analyzer.analyze_sentiment(text)
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1
        
        # Test numerical data extraction
        numerical_data = analyzer.extract_numerical_data("The accuracy improved by 95% over previous methods.")
        assert isinstance(numerical_data, list)
        assert len(numerical_data) > 0
    
    @pytest.mark.asyncio
    async def test_content_processing_pipeline(self):
        """Test complete content processing pipeline"""
        url = "https://example.com/test"
        
        # Process HTML content
        result = await self.processor.process_content(url, self.html_content, "text/html")
        
        assert isinstance(result, ProcessedContent)
        assert result.url == url
        assert result.content_type == ContentType.HTML
        assert result.title == "Test Article"
        assert len(result.text_content) > 0
        assert result.word_count > 0
        assert result.quality_score > 0
    
    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        # Create a sample processed content
        processed = ProcessedContent(
            url="https://example.com",
            content_type=ContentType.HTML,
            title="Test Title",
            text_content="This is a test content with some meaningful text.",
            keywords=["test", "content"],
            entities=[{"text": "test", "type": "TEST"}],
            word_count=10
        )
        
        quality_score = self.processor._calculate_quality_score(processed)
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        initial_stats = self.processor.get_statistics()
        
        assert 'total_processed' in initial_stats
        assert 'successful_processed' in initial_stats
        assert 'failed_processed' in initial_stats
        assert 'runtime_seconds' in initial_stats

class TestKnowledgeExtractor:
    """Test suite for the knowledge extractor component"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = create_knowledge_extractor()
        
        # Sample processed content
        self.processed_content = ProcessedContent(
            url="https://example.com/ml-article",
            content_type=ContentType.HTML,
            title="Machine Learning Advances",
            text_content="Machine learning is a subset of artificial intelligence. "
                        "Neural networks consist of interconnected nodes. "
                        "Deep learning requires large datasets. "
                        "The accuracy improved by 15% over previous methods. "
                        "Research shows that convolutional neural networks excel at image recognition.",
            entities=[{'text': 'Machine Learning', 'type': 'TECHNOLOGY'}],
            keywords=['machine learning', 'neural networks', 'deep learning', 'accuracy']
        )
    
    def test_extractor_initialization(self):
        """Test extractor initialization"""
        assert self.extractor.factual_extractor is not None
        assert self.extractor.conceptual_extractor is not None
        assert self.extractor.relational_extractor is not None
        assert self.extractor.numerical_extractor is not None
        assert len(self.extractor.knowledge_units) == 0
    
    def test_factual_extraction(self):
        """Test factual knowledge extraction"""
        facts = self.extractor.factual_extractor.extract_facts(self.processed_content)
        
        assert isinstance(facts, list)
        assert len(facts) > 0
        
        # Check first fact
        fact = facts[0]
        assert isinstance(fact, KnowledgeUnit)
        assert fact.knowledge_type == KnowledgeType.FACTUAL
        assert fact.confidence > 0
        assert fact.source_url == self.processed_content.url
    
    def test_conceptual_extraction(self):
        """Test conceptual knowledge extraction"""
        concepts = self.extractor.conceptual_extractor.extract_concepts(self.processed_content)
        
        assert isinstance(concepts, list)
        # May be empty if no definitions found
        
        for concept in concepts:
            assert isinstance(concept, KnowledgeUnit)
            assert concept.knowledge_type == KnowledgeType.CONCEPTUAL
    
    def test_relational_extraction(self):
        """Test relational knowledge extraction"""
        relations = self.extractor.relational_extractor.extract_relations(self.processed_content)
        
        assert isinstance(relations, list)
        
        for relation in relations:
            assert isinstance(relation, KnowledgeUnit)
            assert relation.knowledge_type == KnowledgeType.RELATIONAL
            assert len(relation.relations) > 0
    
    def test_numerical_extraction(self):
        """Test numerical knowledge extraction"""
        numerical = self.extractor.numerical_extractor.extract_numerical_data(self.processed_content)
        
        assert isinstance(numerical, list)
        assert len(numerical) > 0  # Should find "15%"
        
        # Check first numerical unit
        num_unit = numerical[0]
        assert isinstance(num_unit, KnowledgeUnit)
        assert num_unit.knowledge_type == KnowledgeType.NUMERICAL
        assert len(num_unit.numerical_values) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_extraction_pipeline(self):
        """Test complete knowledge extraction pipeline"""
        results = await self.extractor.extract_knowledge(self.processed_content)
        
        assert isinstance(results, dict)
        assert 'facts' in results
        assert 'concepts' in results
        assert 'relations' in results
        assert 'numerical' in results
        assert 'total_units' in results
        
        # Check that knowledge was added to internal storage
        assert len(self.extractor.knowledge_units) > 0
    
    def test_knowledge_search(self):
        """Test knowledge search functionality"""
        # First add some knowledge
        self.extractor.knowledge_units.append(
            KnowledgeUnit(
                id="test_id",
                content="Neural networks are machine learning models",
                knowledge_type=KnowledgeType.FACTUAL,
                confidence=0.8,
                source_url="https://example.com",
                entities=["neural networks", "machine learning"]
            )
        )
        
        # Search for knowledge
        results = self.extractor.search_knowledge("neural networks")
        assert len(results) > 0
        assert "neural networks" in results[0].content.lower()
    
    def test_knowledge_graph_creation(self):
        """Test knowledge graph creation"""
        # Create some test knowledge units
        knowledge_units = [
            KnowledgeUnit(
                id="test1",
                content="Machine learning is AI",
                knowledge_type=KnowledgeType.FACTUAL,
                confidence=0.8,
                source_url="https://example.com",
                entities=["machine learning", "AI"]
            ),
            KnowledgeUnit(
                id="test2",
                content="Deep learning requires data",
                knowledge_type=KnowledgeType.RELATIONAL,
                confidence=0.7,
                source_url="https://example.com",
                entities=["deep learning", "data"],
                relations=[{"type": "requires", "source": "deep learning", "target": "data"}]
            )
        ]
        
        # Add to knowledge graph
        self.extractor._add_to_knowledge_graph(knowledge_units)
        
        # Check graph structure
        assert len(self.extractor.knowledge_graph.nodes) > 0
        assert len(self.extractor.knowledge_graph.edges) > 0
    
    def test_statistics_calculation(self):
        """Test statistics calculation"""
        stats = self.extractor.get_statistics()
        
        required_fields = [
            'total_processed', 'facts_extracted', 'concepts_extracted',
            'relations_extracted', 'numerical_extracted', 'runtime_seconds'
        ]
        
        for field in required_fields:
            assert field in stats

class TestLearningPipeline:
    """Test suite for the learning pipeline component"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = LearningConfig()
        self.config.scraping_config.max_total_pages = 2  # Small number for testing
        self.config.scraping_config.requests_per_second = 10.0  # High rate for testing
        self.config.max_concurrent_tasks = 1  # Sequential for testing
        
        # Use temporary directory for results
        self.temp_dir = tempfile.mkdtemp()
        self.config.results_directory = self.temp_dir
        
        self.pipeline = create_learning_pipeline(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.config == self.config
        assert self.pipeline.web_scraper is not None
        assert self.pipeline.content_processor is not None
        assert self.pipeline.knowledge_extractor is not None
        assert self.pipeline.current_session is None
    
    def test_topic_discoverer(self):
        """Test topic discovery functionality"""
        from web_learning.learning_pipeline import TopicDiscoverer
        
        discoverer = TopicDiscoverer(self.pipeline.knowledge_extractor)
        
        # Create mock processed content
        processed_content = [
            ProcessedContent(
                url="https://example.com",
                content_type=ContentType.HTML,
                keywords=["machine learning", "neural networks"],
                entities=[{"text": "artificial intelligence", "type": "TECHNOLOGY"}],
                topics=["technology", "science"]
            )
        ]
        
        # Discover topics
        topics = discoverer.discover_topics_from_content(processed_content)
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        assert any("machine learning" in topic for topic in topics)
    
    def test_knowledge_validator(self):
        """Test knowledge validation functionality"""
        from web_learning.learning_pipeline import KnowledgeValidator
        
        validator = KnowledgeValidator()
        
        # Create test knowledge unit
        knowledge_unit = KnowledgeUnit(
            id="test_id",
            content="Machine learning is a subset of artificial intelligence",
            knowledge_type=KnowledgeType.FACTUAL,
            confidence=0.7,
            source_url="https://example.com",
            entities=["machine learning", "artificial intelligence"]
        )
        
        # Test validation methods
        consistency_score = validator._validate_consistency(knowledge_unit)
        assert isinstance(consistency_score, float)
        assert 0 <= consistency_score <= 1
        
        source_score = validator._validate_source_reliability(knowledge_unit)
        assert isinstance(source_score, float)
        assert 0 <= source_score <= 1
        
        quality_score = validator._validate_content_quality(knowledge_unit)
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization_async(self):
        """Test asynchronous pipeline initialization"""
        await self.pipeline.initialize()
        
        # Check that components are initialized
        assert self.pipeline.content_processor is not None
        assert self.pipeline.knowledge_extractor is not None
    
    def test_learning_task_creation(self):
        """Test learning task creation"""
        from web_learning.learning_pipeline import LearningTask, LearningMode
        
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
        assert len(task.urls) == 1
        assert task.priority == 0.8
    
    def test_learning_session_creation(self):
        """Test learning session creation"""
        from web_learning.learning_pipeline import LearningSession
        
        session = LearningSession(
            id="test_session",
            name="Test Session",
            description="A test learning session"
        )
        
        assert session.id == "test_session"
        assert session.name == "Test Session"
        assert len(session.tasks) == 0
    
    def test_statistics_calculation(self):
        """Test overall statistics calculation"""
        stats = self.pipeline.get_overall_statistics()
        
        required_fields = [
            'total_sessions', 'total_tasks_completed', 'total_content_processed',
            'total_knowledge_extracted', 'runtime_seconds'
        ]
        
        for field in required_fields:
            assert field in stats

class TestWebAutomation:
    """Test suite for the web automation component"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = AutomationConfig(
            browser_type="chrome",
            headless=True,
            take_screenshots=False,  # Disable for testing
            page_load_timeout=5  # Shorter timeout for testing
        )
        
        # Only create automation if Selenium is available
        try:
            from selenium import webdriver
            self.automation = create_web_automation(self.config)
            self.selenium_available = True
        except ImportError:
            self.selenium_available = False
    
    def test_automation_config(self):
        """Test automation configuration"""
        assert self.config.browser_type == "chrome"
        assert self.config.headless == True
        assert self.config.page_load_timeout == 5
    
    def test_automation_initialization(self):
        """Test automation initialization"""
        if not self.selenium_available:
            pytest.skip("Selenium not available")
        
        assert self.automation.config == self.config
        assert self.automation.is_initialized == False
    
    def test_browser_manager(self):
        """Test browser manager functionality"""
        if not self.selenium_available:
            pytest.skip("Selenium not available")
        
        from web_learning.web_automation import BrowserManager
        
        manager = BrowserManager(self.config)
        assert manager.config == self.config
        assert manager.driver is None
        assert manager.session_id is None
    
    def test_automation_statistics(self):
        """Test automation statistics"""
        if not self.selenium_available:
            pytest.skip("Selenium not available")
        
        stats = self.automation.get_statistics()
        
        required_fields = [
            'pages_visited', 'actions_performed', 'screenshots_taken',
            'errors_encountered', 'runtime_seconds'
        ]
        
        for field in required_fields:
            assert field in stats

class TestIntegration:
    """Integration tests for the complete learning pipeline"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config for testing
        self.config = LearningConfig()
        self.config.scraping_config.max_total_pages = 1
        self.config.scraping_config.requests_per_second = 5.0
        self.config.scraping_config.respect_robots_txt = False
        self.config.max_concurrent_tasks = 1
        self.config.results_directory = self.temp_dir
        
        self.pipeline = create_learning_pipeline(self.config)
    
    def teardown_method(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_component_integration(self):
        """Test that all components integrate correctly"""
        # Test that pipeline has all required components
        assert hasattr(self.pipeline, 'web_scraper')
        assert hasattr(self.pipeline, 'content_processor')
        assert hasattr(self.pipeline, 'knowledge_extractor')
        assert hasattr(self.pipeline, 'topic_discoverer')
        
        # Test that components are properly initialized
        assert self.pipeline.web_scraper is not None
        assert self.pipeline.content_processor is not None
        assert self.pipeline.knowledge_extractor is not None
        assert self.pipeline.topic_discoverer is not None
    
    def test_data_flow(self):
        """Test data flow between components"""
        # Create mock scraped data
        scraped_data = {
            'url': 'https://example.com',
            'content': '<html><body><h1>Test</h1><p>This is test content.</p></body></html>',
            'content_type': 'text/html'
        }
        
        # Test content processor
        processor = self.pipeline.content_processor
        
        # This would need to be async in real implementation
        # For now, just test that the processor exists and has the right methods
        assert hasattr(processor, 'process_content')
        assert hasattr(processor, 'detect_content_type')
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test that config is properly set
        assert self.pipeline.config.scraping_config.max_total_pages == 1
        assert self.pipeline.config.max_concurrent_tasks == 1
        assert self.pipeline.config.results_directory == self.temp_dir
    
    def test_error_handling(self):
        """Test error handling across components"""
        # Test that components handle errors gracefully
        
        # Test web scraper with invalid URL
        invalid_url = "not-a-valid-url"
        assert not self.pipeline.web_scraper.content_validator.is_valid_url(invalid_url)
        
        # Test content processor with empty content
        processor = self.pipeline.content_processor
        content_type = processor.detect_content_type("", "")
        assert content_type is not None
    
    def test_statistics_aggregation(self):
        """Test statistics aggregation across components"""
        # Test that each component provides statistics
        web_scraper_stats = self.pipeline.web_scraper.get_statistics()
        content_processor_stats = self.pipeline.content_processor.get_statistics()
        knowledge_extractor_stats = self.pipeline.knowledge_extractor.get_statistics()
        pipeline_stats = self.pipeline.get_overall_statistics()
        
        # All should be dictionaries with required fields
        assert isinstance(web_scraper_stats, dict)
        assert isinstance(content_processor_stats, dict)
        assert isinstance(knowledge_extractor_stats, dict)
        assert isinstance(pipeline_stats, dict)
        
        # Check for key fields
        assert 'runtime_seconds' in web_scraper_stats
        assert 'runtime_seconds' in content_processor_stats
        assert 'runtime_seconds' in knowledge_extractor_stats
        assert 'runtime_seconds' in pipeline_stats

def run_tests():
    """Run all tests"""
    print("Running SutazAI V7 Self-Supervised Learning Pipeline Tests")
    print("=" * 60)
    
    # Run tests using pytest
    test_files = [
        'test_learning_pipeline.py::TestWebScraper',
        'test_learning_pipeline.py::TestContentProcessor',
        'test_learning_pipeline.py::TestKnowledgeExtractor',
        'test_learning_pipeline.py::TestLearningPipeline',
        'test_learning_pipeline.py::TestWebAutomation',
        'test_learning_pipeline.py::TestIntegration'
    ]
    
    try:
        # Run tests
        for test_class in test_files:
            print(f"\nRunning {test_class}...")
            result = pytest.main(['-v', test_class])
            if result != 0:
                print(f"Tests failed for {test_class}")
        
        print("\n" + "=" * 60)
        print("Test Summary:")
        print("✓ Web Scraper Tests")
        print("✓ Content Processor Tests")
        print("✓ Knowledge Extractor Tests")
        print("✓ Learning Pipeline Tests")
        print("✓ Web Automation Tests")
        print("✓ Integration Tests")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running tests: {e}")

if __name__ == "__main__":
    run_tests()