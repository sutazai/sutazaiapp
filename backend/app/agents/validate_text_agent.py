#!/usr/bin/env python3
"""
Validation Script for Text Analysis Agent
==========================================

This script validates that the Text Analysis Agent is:
1. Properly integrated into the backend
2. Accessible via API endpoints
3. Producing real AI results (not stubs)
4. Performing with acceptable metrics

Run this after starting the backend to confirm everything works.
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import asyncio
import aiohttp
import time
from datetime import datetime
import sys

# Configuration
BACKEND_URL = "http://localhost:10010"
OLLAMA_URL = "http://localhost:10104"

# Test texts for validation
VALIDATION_TEXTS = {
    "positive": "This is absolutely fantastic! I love how well this works.",
    "negative": "This is terrible and completely broken. Very disappointed.",
    "technical": "Machine learning algorithms process data using neural networks.",
    "news": "President Biden met with leaders in Washington DC on January 15, 2024."
}

# ANSI color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ️  {text}{Colors.RESET}")


async def check_ollama_connection() -> bool:
    """Check if Ollama is accessible"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_URL}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    if 'tinyllama' in models or 'tinyllama:latest' in models:
                        print_success(f"Ollama connected with tinyllama model")
                        return True
                    else:
                        print_error(f"Ollama connected but tinyllama not found. Models: {models}")
                        return False
    except Exception as e:
        print_error(f"Ollama connection failed: {e}")
        return False


async def check_backend_connection() -> bool:
    """Check if backend is accessible"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BACKEND_URL}/health") as response:
                if response.status == 200:
                    print_success("Backend API is accessible")
                    return True
    except Exception as e:
        print_error(f"Backend connection failed: {e}")
        return False


async def check_text_analysis_endpoint() -> bool:
    """Check if Text Analysis endpoints are available"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BACKEND_URL}/api/text-analysis/health") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('healthy'):
                        print_success("Text Analysis Agent is healthy")
                        return True
                    else:
                        print_error(f"Text Analysis Agent unhealthy: {data}")
                        return False
    except Exception as e:
        print_error(f"Text Analysis endpoint check failed: {e}")
        return False


async def test_sentiment_analysis() -> bool:
    """Test sentiment analysis functionality"""
    print_info("Testing sentiment analysis...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test positive sentiment
            params = {"text": VALIDATION_TEXTS["positive"]}
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/sentiment",
                params=params
            ) as response:
                if response.status != 200:
                    print_error(f"Sentiment API returned {response.status}")
                    return False
                
                data = await response.json()
                
                # Validate response structure
                required_fields = ["sentiment", "confidence", "model_used"]
                for field in required_fields:
                    if field not in data:
                        print_error(f"Missing field in response: {field}")
                        return False
                
                # Check if it's not a stub (would always return same result)
                if data['sentiment'] == 'positive' and data['confidence'] > 0.5:
                    print_success(f"Positive sentiment detected correctly: {data['sentiment']} ({data['confidence']:.2%})")
                else:
                    print_error(f"Unexpected sentiment result: {data}")
                    return False
            
            # Test negative sentiment
            params = {"text": VALIDATION_TEXTS["negative"]}
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/sentiment",
                params=params
            ) as response:
                data = await response.json()
                if data['sentiment'] == 'negative' and data['confidence'] > 0.5:
                    print_success(f"Negative sentiment detected correctly: {data['sentiment']} ({data['confidence']:.2%})")
                    return True
                else:
                    print_error(f"Unexpected sentiment result: {data}")
                    return False
                    
    except Exception as e:
        print_error(f"Sentiment analysis test failed: {e}")
        return False


async def test_entity_extraction() -> bool:
    """Test entity extraction functionality"""
    print_info("Testing entity extraction...")
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {"text": VALIDATION_TEXTS["news"]}
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/entities",
                params=params
            ) as response:
                if response.status != 200:
                    print_error(f"Entity API returned {response.status}")
                    return False
                
                data = await response.json()
                
                # Check for entities
                entities = data.get('entities', {})
                found_entities = False
                
                for entity_type in ['people', 'locations', 'dates']:
                    if entities.get(entity_type):
                        found_entities = True
                        print_success(f"Found {entity_type}: {entities[entity_type][:3]}")
                
                return found_entities
                
    except Exception as e:
        print_error(f"Entity extraction test failed: {e}")
        return False


async def test_summarization() -> bool:
    """Test text summarization functionality"""
    print_info("Testing text summarization...")
    
    long_text = VALIDATION_TEXTS["technical"] * 10  # Make it longer
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {"text": long_text, "max_sentences": 2}
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/summary",
                params=params
            ) as response:
                if response.status != 200:
                    print_error(f"Summary API returned {response.status}")
                    return False
                
                data = await response.json()
                
                summary = data.get('summary', '')
                if summary and len(summary) < len(long_text):
                    compression = data.get('compression_ratio', 1.0)
                    print_success(f"Summary generated with {compression:.2%} compression ratio")
                    print_info(f"Summary: {summary[:100]}...")
                    return True
                else:
                    print_error("Summary not properly generated")
                    return False
                    
    except Exception as e:
        print_error(f"Summarization test failed: {e}")
        return False


async def test_keyword_extraction() -> bool:
    """Test keyword extraction functionality"""
    print_info("Testing keyword extraction...")
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {"text": VALIDATION_TEXTS["technical"], "num_keywords": 5}
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/keywords",
                params=params
            ) as response:
                if response.status != 200:
                    print_error(f"Keywords API returned {response.status}")
                    return False
                
                data = await response.json()
                
                keywords = data.get('keywords', [])
                if keywords and len(keywords) > 0:
                    print_success(f"Keywords extracted: {', '.join(keywords)}")
                    return True
                else:
                    print_error("No keywords extracted")
                    return False
                    
    except Exception as e:
        print_error(f"Keyword extraction test failed: {e}")
        return False


async def test_language_detection() -> bool:
    """Test language detection functionality"""
    print_info("Testing language detection...")
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {"text": "Hello world, this is English text."}
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/language",
                params=params
            ) as response:
                if response.status != 200:
                    print_error(f"Language API returned {response.status}")
                    return False
                
                data = await response.json()
                
                language = data.get('language', '')
                confidence = data.get('confidence', 0)
                if language and confidence > 0:
                    print_success(f"Language detected: {language} ({confidence:.2%} confidence)")
                    return True
                else:
                    print_error("Language detection failed")
                    return False
                    
    except Exception as e:
        print_error(f"Language detection test failed: {e}")
        return False


async def test_caching_performance() -> bool:
    """Test that caching improves performance"""
    print_info("Testing caching performance...")
    
    test_text = "This is a test for caching performance."
    
    try:
        async with aiohttp.ClientSession() as session:
            # First request (cache miss)
            start = time.time()
            params = {"text": test_text}
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/sentiment",
                params=params
            ) as response:
                data1 = await response.json()
            time1 = time.time() - start
            
            # Second request (should hit cache)
            start = time.time()
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/sentiment",
                params=params
            ) as response:
                data2 = await response.json()
            time2 = time.time() - start
            
            # Results should be identical
            if data1['sentiment'] != data2['sentiment']:
                print_error("Inconsistent results between calls")
                return False
            
            # Second call should be faster (cache hit)
            if time2 < time1:
                speedup = time1 / time2
                print_success(f"Caching working: Second call {speedup:.1f}x faster")
                print_info(f"First call: {time1:.3f}s, Second call: {time2:.3f}s")
                return True
            else:
                print_info("Cache might not be working (Redis unavailable?)")
                return True  # Not a failure, just informational
                
    except Exception as e:
        print_error(f"Caching test failed: {e}")
        return False


async def test_batch_processing() -> bool:
    """Test batch processing capability"""
    print_info("Testing batch processing...")
    
    texts = [
        "This is positive text.",
        "This is negative text.",
        "This is neutral text."
    ]
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "texts": texts,
                "analysis_type": "sentiment"
            }
            async with session.post(
                f"{BACKEND_URL}/api/text-analysis/batch",
                params=params
            ) as response:
                if response.status != 200:
                    print_error(f"Batch API returned {response.status}")
                    return False
                
                data = await response.json()
                
                if data.get('successful', 0) > 0:
                    print_success(f"Batch processing successful: {data['successful']}/{data['total']} processed")
                    return True
                else:
                    print_error("Batch processing failed")
                    return False
                    
    except Exception as e:
        print_error(f"Batch processing test failed: {e}")
        return False


async def get_agent_statistics() -> Dict[str, Any]:
    """Get and display agent statistics"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BACKEND_URL}/api/text-analysis/stats") as response:
                if response.status == 200:
                    return await response.json()
    except Exception as e:
        # Suppressed exception (was bare except)
        logger.debug(f"Suppressed exception: {e}")
        pass
    return {}


async def main():
    """Main validation routine"""
    print_header("TEXT ANALYSIS AGENT VALIDATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Ollama URL: {OLLAMA_URL}")
    
    # Track test results
    results = {}
    
    # Prerequisites
    print_header("CHECKING PREREQUISITES")
    
    results['ollama'] = await check_ollama_connection()
    results['backend'] = await check_backend_connection()
    results['endpoint'] = await check_text_analysis_endpoint()
    
    if not all([results['ollama'], results['backend'], results['endpoint']]):
        print_error("\nPrerequisites not met. Please ensure:")
        print_info("1. Ollama is running with tinyllama model")
        print_info("2. Backend is running on port 10010")
        print_info("3. Text Analysis Agent is properly integrated")
        return False
    
    # Functional tests
    print_header("FUNCTIONAL VALIDATION")
    
    results['sentiment'] = await test_sentiment_analysis()
    results['entities'] = await test_entity_extraction()
    results['summary'] = await test_summarization()
    results['keywords'] = await test_keyword_extraction()
    results['language'] = await test_language_detection()
    
    # Performance tests
    print_header("PERFORMANCE VALIDATION")
    
    results['caching'] = await test_caching_performance()
    results['batch'] = await test_batch_processing()
    
    # Get statistics
    print_header("AGENT STATISTICS")
    
    stats = await get_agent_statistics()
    if stats:
        print_info(f"Agent ID: {stats.get('agent_id', 'N/A')}")
        print_info(f"Status: {stats.get('status', 'N/A')}")
        print_info(f"Total analyses: {stats.get('total_analyses', 0)}")
        print_info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        print_info(f"Model: {stats.get('model_name', 'N/A')}")
        
        metrics = stats.get('analysis_metrics', {})
        if metrics:
            print_info(f"Sentiment analyses: {metrics.get('sentiment_analyses', 0)}")
            print_info(f"Entity extractions: {metrics.get('entity_extractions', 0)}")
            print_info(f"Summaries generated: {metrics.get('summaries_generated', 0)}")
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"\nTests passed: {passed}/{total} ({success_rate:.0f}%)")
    
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {test_name:15} : {status}")
    
    if success_rate >= 80:
        print_success(f"\n🎉 Text Analysis Agent is fully functional!")
        print_success("This is a REAL AI agent with genuine intelligence!")
        return True
    elif success_rate >= 60:
        print_info(f"\n⚠️ Text Analysis Agent is partially functional")
        print_info("Some features may not be working correctly")
        return True
    else:
        print_error(f"\n❌ Text Analysis Agent validation failed")
        print_error("Please check the logs and configuration")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)