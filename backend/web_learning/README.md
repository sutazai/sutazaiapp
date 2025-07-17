# SutazAI V7 Self-Supervised Learning Pipeline

## Overview

The Self-Supervised Learning Pipeline is a comprehensive system for autonomous knowledge acquisition through web scraping, content processing, and knowledge extraction. It integrates with the advanced biological neural networks to provide brain-inspired learning capabilities.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Scraper   │───▶│Content Processor│───▶│Knowledge Extract│
│                 │    │                 │    │                 │
│ • Rate Limited  │    │ • HTML/JSON/XML │    │ • Factual       │
│ • Robots.txt    │    │ • Text Analysis │    │ • Conceptual    │
│ • Caching       │    │ • Sentiment     │    │ • Relational    │
│ • Validation    │    │ • Keywords      │    │ • Numerical     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Web Automation  │    │ Neural Networks │    │ Vector Database │
│                 │    │                 │    │                 │
│ • Browser       │    │ • Biological    │    │ • Semantic      │
│ • JavaScript    │    │ • Validation    │    │ • Search        │
│ • Interactions  │    │ • Learning      │    │ • Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### 1. Web Scraper (`web_scraper.py`)

Enterprise-grade web scraping with:
- **Rate limiting** and respectful crawling
- **Robots.txt compliance**
- **Content validation** and sanitization
- **Caching** for efficiency
- **Error handling** and retry mechanisms

**Key Features:**
- Configurable request rates
- Multiple content type support
- Automatic link extraction
- Security-focused design
- Comprehensive logging

### 2. Content Processor (`content_processor.py`)

Intelligent content processing with:
- **Multi-format support** (HTML, JSON, XML, text)
- **Text analysis** and keyword extraction
- **Entity recognition** and sentiment analysis
- **Summary generation**
- **Quality scoring**

**Key Features:**
- Automatic content type detection
- Advanced text analytics
- Metadata extraction
- Readability analysis
- Topic classification

### 3. Knowledge Extractor (`knowledge_extractor.py`)

Structured knowledge extraction with:
- **Factual knowledge** extraction
- **Conceptual understanding**
- **Relational mapping**
- **Numerical data** capture
- **Knowledge graph** construction

**Key Features:**
- Multiple knowledge types
- Biological neural validation
- Confidence scoring
- Graph-based storage
- Search capabilities

### 4. Learning Pipeline (`learning_pipeline.py`)

Self-supervised learning orchestration:
- **Automated learning sessions**
- **Topic discovery**
- **Knowledge validation**
- **Adaptive learning modes**
- **Progress tracking**

**Key Features:**
- Configurable learning strategies
- Concurrent processing
- Session management
- Statistics and monitoring
- Result persistence

### 5. Web Automation (`web_automation.py`)

Browser automation for complex sites:
- **JavaScript execution**
- **Dynamic content loading**
- **Form interactions**
- **Screenshot capture**
- **Error recovery**

**Key Features:**
- Headless browser support
- Element waiting strategies
- Action chains
- Performance monitoring
- Security controls

## Installation

### Prerequisites

```bash
# Core dependencies
pip install aiohttp beautifulsoup4 numpy torch

# Optional dependencies for enhanced features
pip install nltk selenium pyppeteer

# For testing
pip install pytest pytest-asyncio

# For NLTK (if using text analysis)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

### Browser Setup (for Web Automation)

```bash
# For Chrome
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
apt-get update
apt-get install -y google-chrome-stable

# Download ChromeDriver
wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/LATEST_RELEASE/chromedriver_linux64.zip
unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/
```

## Usage

### Basic Usage

```python
import asyncio
from web_learning import create_learning_pipeline, LearningConfig

async def main():
    # Create configuration
    config = LearningConfig()
    config.scraping_config.requests_per_second = 1.0
    config.scraping_config.max_total_pages = 50
    
    # Create pipeline
    pipeline = create_learning_pipeline(config)
    
    # Initialize
    await pipeline.initialize()
    
    # Start learning session
    session_id = await pipeline.start_learning_session(
        session_name="AI Research",
        description="Learn about artificial intelligence developments",
        topics=["machine learning", "neural networks", "deep learning"]
    )
    
    # Get results
    stats = pipeline.get_overall_statistics()
    print(f"Extracted {stats['total_knowledge_extracted']} knowledge units")
    
    # Shutdown
    await pipeline.shutdown()

asyncio.run(main())
```

### Web Scraping Only

```python
from web_learning import create_web_scraper, ScrapingConfig

async def scrape_example():
    config = ScrapingConfig(
        requests_per_second=0.5,
        max_total_pages=10,
        respect_robots_txt=True
    )
    
    async with create_web_scraper(config) as scraper:
        # Single URL
        result = await scraper.fetch_url("https://example.com")
        
        # Multiple URLs
        urls = ["https://site1.com", "https://site2.com"]
        results = await scraper.scrape_urls(urls)
        
        # Website crawl
        crawl_results = await scraper.crawl_website(
            "https://example.com", 
            max_depth=2
        )
        
        # Statistics
        stats = scraper.get_statistics()
        print(f"Success rate: {stats['success_rate']:.1f}%")

asyncio.run(scrape_example())
```

### Content Processing

```python
from web_learning import create_content_processor

async def process_content():
    processor = create_content_processor()
    
    html_content = """
    <html>
        <head><title>AI Research</title></head>
        <body>
            <h1>Machine Learning Breakthrough</h1>
            <p>Researchers have achieved 95% accuracy...</p>
        </body>
    </html>
    """
    
    # Process content
    result = await processor.process_content(
        "https://example.com/article",
        html_content,
        "text/html"
    )
    
    print(f"Title: {result.title}")
    print(f"Keywords: {result.keywords}")
    print(f"Summary: {result.summary}")
    print(f"Quality: {result.quality_score}")

asyncio.run(process_content())
```

### Knowledge Extraction

```python
from web_learning import create_knowledge_extractor

async def extract_knowledge():
    extractor = create_knowledge_extractor()
    await extractor.initialize_neural_engine()
    
    # Use processed content from above
    knowledge = await extractor.extract_knowledge(processed_content)
    
    print(f"Facts: {len(knowledge['facts'])}")
    print(f"Concepts: {len(knowledge['concepts'])}")
    print(f"Relations: {len(knowledge['relations'])}")
    
    # Search knowledge
    results = extractor.search_knowledge("neural networks")
    for result in results:
        print(f"- {result.content} (confidence: {result.confidence:.2f})")

asyncio.run(extract_knowledge())
```

### Web Automation

```python
from web_learning import create_web_automation, AutomationConfig

async def automate_web():
    config = AutomationConfig(
        browser_type="chrome",
        headless=True,
        take_screenshots=True
    )
    
    automation = create_web_automation(config)
    
    try:
        await automation.initialize()
        
        # Extract dynamic content
        results = await automation.extract_dynamic_content(
            "https://example.com",
            wait_conditions=[
                {'type': 'element', 'selector': '.content', 'timeout': 10}
            ]
        )
        
        # Automate interactions
        interactions = [
            {'type': 'click', 'selector': 'button.load-more'},
            {'type': 'wait', 'selector': '.new-content', 'timeout': 5},
            {'type': 'find', 'selector': '.article'}
        ]
        
        interaction_results = await automation.automate_page_interaction(
            "https://example.com",
            interactions
        )
        
        print(f"Performed {interaction_results['interactions_performed']} interactions")
        
    finally:
        await automation.shutdown()

asyncio.run(automate_web())
```

## Configuration

### Scraping Configuration

```python
from web_learning import ScrapingConfig

config = ScrapingConfig(
    # Rate limiting
    requests_per_second=1.0,
    max_concurrent_requests=5,
    
    # Timeouts
    connection_timeout=10.0,
    read_timeout=30.0,
    
    # Scope
    max_depth=3,
    max_pages_per_domain=100,
    max_total_pages=1000,
    
    # Security
    respect_robots_txt=True,
    verify_ssl=True,
    
    # Content filtering
    max_page_size=10*1024*1024,  # 10MB
    allowed_content_types=['text/html', 'application/json'],
    
    # Caching
    cache_responses=True,
    cache_duration=3600  # 1 hour
)
```

### Learning Configuration

```python
from web_learning import LearningConfig

config = LearningConfig(
    # Processing
    max_concurrent_tasks=3,
    session_duration_hours=2,
    
    # Neural network
    neural_config={
        'use_advanced_biological_modeling': True,
        'network': {
            'population_sizes': {
                'sensory': 128,
                'l2_3_pyramidal': 256,
                'l5_pyramidal': 128,
                'fast_spiking': 64,
                'dopaminergic': 32,
                'output': 64
            }
        }
    },
    
    # Features
    enable_topic_discovery=True,
    enable_knowledge_validation=True,
    
    # Storage
    save_intermediate_results=True,
    results_directory="/opt/sutazaiapp/data/learning_results"
)
```

## Testing

### Running Tests

```bash
# Run all tests
cd /opt/sutazaiapp/backend/web_learning
python -m pytest test_learning_pipeline.py -v

# Run specific test class
python -m pytest test_learning_pipeline.py::TestWebScraper -v

# Run with coverage
python -m pytest test_learning_pipeline.py --cov=web_learning --cov-report=html
```

### Test Structure

```
tests/
├── test_web_scraper.py         # Web scraping tests
├── test_content_processor.py   # Content processing tests  
├── test_knowledge_extractor.py # Knowledge extraction tests
├── test_learning_pipeline.py   # Pipeline integration tests
├── test_web_automation.py      # Web automation tests
└── test_integration.py         # End-to-end tests
```

## Performance

### Benchmarks

| Component | Throughput | Memory Usage | CPU Usage |
|-----------|------------|--------------|-----------|
| Web Scraper | 1-10 pages/sec | 50-100MB | 10-30% |
| Content Processor | 5-20 pages/sec | 100-200MB | 20-50% |
| Knowledge Extractor | 2-10 pages/sec | 200-500MB | 30-70% |
| Neural Validation | 1-5 pages/sec | 500MB-1GB | 50-90% |

### Optimization Tips

1. **Adjust concurrency**: Increase `max_concurrent_tasks` for faster processing
2. **Tune rate limiting**: Balance speed with server respect
3. **Enable caching**: Reduce redundant requests
4. **Filter content**: Use `allowed_content_types` to skip unwanted content
5. **Batch processing**: Process multiple URLs in batches
6. **Memory management**: Monitor memory usage and adjust batch sizes

## Monitoring

### Built-in Statistics

```python
# Web scraper statistics
stats = scraper.get_statistics()
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
print(f"Average response time: {stats['avg_response_time']:.2f}s")

# Content processor statistics
stats = processor.get_statistics()
print(f"Processing rate: {stats['processing_rate']:.1f} pages/sec")
print(f"Quality score: {stats['avg_quality_score']:.2f}")

# Knowledge extractor statistics
stats = extractor.get_statistics()
print(f"Extraction rate: {stats['extraction_rate']:.1f} units/sec")
print(f"Knowledge graph size: {stats['knowledge_graph_nodes']} nodes")

# Learning pipeline statistics
stats = pipeline.get_overall_statistics()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Knowledge extracted: {stats['total_knowledge_extracted']}")
```

### Integration with SutazAI Monitoring

The pipeline integrates with the SutazAI monitoring system:

```python
from utils.monitoring_integration import create_monitoring_system

# Create monitoring system
monitoring = create_monitoring_system(
    system_id="learning_pipeline",
    enable_neural_monitoring=True
)

# Monitor learning session
with monitoring.monitor_inference("learning_session"):
    session_id = await pipeline.start_learning_session(
        session_name="Research Session",
        description="Learn about recent developments"
    )
```

## Security

### Built-in Security Features

1. **Content sanitization**: Removes malicious scripts and code
2. **URL validation**: Blocks suspicious and malicious URLs
3. **Rate limiting**: Prevents overwhelming target servers
4. **SSL verification**: Ensures secure connections
5. **Robots.txt respect**: Follows site crawling guidelines
6. **Content type filtering**: Only processes safe content types

### Security Best Practices

```python
# Secure configuration
config = ScrapingConfig(
    # Security settings
    respect_robots_txt=True,
    verify_ssl=True,
    max_page_size=10*1024*1024,  # Prevent huge downloads
    
    # Content filtering
    allowed_content_types=['text/html', 'application/json'],
    exclude_file_extensions=['.exe', '.zip', '.rar'],
    
    # Rate limiting
    requests_per_second=1.0,  # Respectful crawling
    max_requests_per_hour=1000,
    
    # User agent
    user_agent="SutazAI/7.0 (Educational/Research Bot; contact@sutazai.com)"
)
```

## Troubleshooting

### Common Issues

1. **Rate limiting errors**: Reduce `requests_per_second`
2. **Memory issues**: Decrease `max_concurrent_tasks` or batch sizes
3. **Timeout errors**: Increase timeout values
4. **Robot.txt blocking**: Set `respect_robots_txt=False` (use responsibly)
5. **SSL errors**: Set `verify_ssl=False` (not recommended for production)

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Component-specific logging
logger = logging.getLogger("web_learning")
logger.setLevel(logging.DEBUG)
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_learning_session():
    # Your learning pipeline code here
    pass

# Profile the session
cProfile.run('profile_learning_session()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(20)
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/sutazai/sutazai-v7.git
cd sutazai-v7/backend/web_learning

# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest

# Run linting
flake8 .
black .
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings to all public methods
- Include comprehensive error handling
- Write unit tests for new features

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Run full test suite
4. Submit pull request with description
5. Address review feedback

## License

This project is part of SutazAI V7 and is subject to the project's license terms.

## Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the SutazAI documentation

---

**SutazAI V7 Self-Supervised Learning Pipeline**  
*Autonomous Knowledge Acquisition with Biological Neural Networks*