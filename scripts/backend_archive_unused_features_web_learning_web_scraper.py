#!/usr/bin/env python3
"""
Enterprise-Grade Web Scraper for SutazAI V7
Implements secure, rate-limited, and intelligent web scraping with respect for robots.txt
"""

import os
import sys
import asyncio
import aiohttp
import urllib.robotparser
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, robots
from bs4 import BeautifulSoup
import ssl
import certifi
from datetime import datetime, timedelta
import json
import re
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.secure_config import SecureConfigManager
from utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class ScrapingConfig:
    """Configuration for web scraping operations"""
    # Rate limiting
    requests_per_second: float = 1.0
    max_concurrent_requests: int = 5
    delay_between_requests: float = 1.0
    
    # Timeouts
    connection_timeout: float = 10.0
    read_timeout: float = 30.0
    total_timeout: float = 60.0
    
    # User agent and headers
    user_agent: str = "SutazAI/7.0 (Educational/Research Bot; contact@sutazai.com)"
    respect_robots_txt: bool = True
    
    # Content filtering
    max_page_size: int = 10 * 1024 * 1024  # 10MB
    allowed_content_types: List[str] = field(default_factory=lambda: [
        'text/html', 'text/plain', 'application/json', 'text/xml'
    ])
    
    # Depth and scope
    max_depth: int = 3
    max_pages_per_domain: int = 100
    max_total_pages: int = 1000
    
    # Security
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_redirects: int = 5
    
    # Storage
    cache_responses: bool = True
    cache_duration: int = 3600  # 1 hour
    
    # Filtering
    exclude_file_extensions: List[str] = field(default_factory=lambda: [
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif',
        '.svg', '.webp', '.mp4', '.avi', '.mov', '.mp3', '.wav'
    ])
    
    # Ethical guidelines
    max_requests_per_hour: int = 1000
    respect_crawl_delay: bool = True

class RobotsChecker:
    """Manages robots.txt compliance"""
    
    def __init__(self):
        self.robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(hours=1)
    
    async def can_fetch(self, url: str, user_agent: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            # Check cache
            if robots_url in self.robots_cache:
                if datetime.now() - self.cache_expiry[robots_url] < self.cache_duration:
                    rp = self.robots_cache[robots_url]
                    return rp.can_fetch(user_agent, url)
            
            # Fetch and parse robots.txt
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(robots_url, timeout=5) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            rp.read_robots_content(robots_content)
                        else:
                            # If robots.txt doesn't exist, assume all allowed
                            return True
            except Exception:
                # If can't fetch robots.txt, assume all allowed
                return True
            
            # Cache the result
            self.robots_cache[robots_url] = rp
            self.cache_expiry[robots_url] = datetime.now()
            
            return rp.can_fetch(user_agent, url)
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True  # Default to allowing if check fails

class RateLimiter:
    """Rate limiter for web scraping"""
    
    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.request_count = 0
        self.hour_start = time.time()
    
    async def wait(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        
        # Check hourly limit
        if current_time - self.hour_start > 3600:
            self.request_count = 0
            self.hour_start = current_time
        
        # Wait for minimum interval
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

class ContentValidator:
    """Validates and sanitizes scraped content"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                 for pattern in self.malicious_patterns]
    
    def is_valid_content_type(self, content_type: str) -> bool:
        """Check if content type is allowed"""
        if not content_type:
            return False
        
        content_type = content_type.lower().split(';')[0].strip()
        return any(allowed in content_type for allowed in self.config.allowed_content_types)
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and safe"""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Check for file extensions
            path = parsed.path.lower()
            if any(path.endswith(ext) for ext in self.config.exclude_file_extensions):
                return False
            
            # Check for suspicious patterns
            if any(pattern in url.lower() for pattern in ['javascript:', 'data:', 'vbscript:']):
                return False
            
            return True
            
        except Exception:
            return False
    
    def sanitize_content(self, content: str) -> str:
        """Sanitize content to remove potentially malicious code"""
        if not content:
            return ""
        
        # Remove malicious patterns
        for pattern in self.compiled_patterns:
            content = pattern.sub('', content)
        
        return content

class WebScraper:
    """
    Enterprise-grade web scraper with biological neural network integration
    """
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.robots_checker = RobotsChecker()
        self.rate_limiter = RateLimiter(config.requests_per_second)
        self.content_validator = ContentValidator(config)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Tracking
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.domain_counters: Dict[str, int] = {}
        
        # Cache
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'robots_blocked': 0,
            'cache_hits': 0,
            'content_filtered': 0,
            'start_time': datetime.now()
        }
        
        # Security
        self.secure_config = SecureConfigManager()
        
        logger.info(f"WebScraper initialized with {config.requests_per_second} RPS limit")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
    
    async def start_session(self):
        """Start the HTTP session"""
        if self.session is None:
            # SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            if not self.config.verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Connection limits
            connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent_requests * 2,
                limit_per_host=self.config.max_concurrent_requests,
                ssl=ssl_context,
                enable_cleanup_closed=True
            )
            
            # Timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self.config.total_timeout,
                connect=self.config.connection_timeout,
                sock_read=self.config.read_timeout
            )
            
            # Headers
            headers = {
                'User-Agent': self.config.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
                max_redirects=self.config.max_redirects if self.config.follow_redirects else 0
            )
    
    async def close_session(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached response is still valid"""
        if not self.config.cache_responses:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        return cache_age.total_seconds() < self.config.cache_duration
    
    async def fetch_url(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch a single URL with all safety checks
        
        Returns:
            Dict containing response data or None if failed
        """
        # Validate URL
        if not self.content_validator.is_valid_url(url):
            logger.warning(f"Invalid URL: {url}")
            return None
        
        # Check if already scraped
        if url in self.scraped_urls:
            logger.debug(f"URL already scraped: {url}")
            return None
        
        # Check domain limits
        domain = urlparse(url).netloc
        if self.domain_counters.get(domain, 0) >= self.config.max_pages_per_domain:
            logger.warning(f"Domain limit reached for {domain}")
            return None
        
        # Check total pages limit
        if len(self.scraped_urls) >= self.config.max_total_pages:
            logger.warning("Total pages limit reached")
            return None
        
        # Check cache
        cache_key = self._get_cache_key(url)
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {url}")
            self.stats['cache_hits'] += 1
            return self.response_cache[cache_key]
        
        # Check robots.txt
        if self.config.respect_robots_txt:
            if not await self.robots_checker.can_fetch(url, self.config.user_agent):
                logger.info(f"Robots.txt blocks access to {url}")
                self.stats['robots_blocked'] += 1
                return None
        
        # Rate limiting
        await self.rate_limiter.wait()
        
        # Ensure session is started
        if self.session is None:
            await self.start_session()
        
        try:
            self.stats['total_requests'] += 1
            
            logger.debug(f"Fetching {url}")
            async with self.session.get(url, **kwargs) as response:
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if not self.content_validator.is_valid_content_type(content_type):
                    logger.warning(f"Invalid content type for {url}: {content_type}")
                    self.stats['content_filtered'] += 1
                    return None
                
                # Check content length
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > self.config.max_page_size:
                    logger.warning(f"Content too large for {url}: {content_length} bytes")
                    self.stats['content_filtered'] += 1
                    return None
                
                # Read content
                content = await response.text()
                
                # Check actual content size
                if len(content) > self.config.max_page_size:
                    logger.warning(f"Content too large for {url}: {len(content)} bytes")
                    self.stats['content_filtered'] += 1
                    return None
                
                # Sanitize content
                sanitized_content = self.content_validator.sanitize_content(content)
                
                # Prepare response data
                response_data = {
                    'url': url,
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'content': sanitized_content,
                    'content_type': content_type,
                    'timestamp': datetime.now().isoformat(),
                    'size': len(sanitized_content)
                }
                
                # Update tracking
                self.scraped_urls.add(url)
                self.domain_counters[domain] = self.domain_counters.get(domain, 0) + 1
                self.stats['successful_requests'] += 1
                
                # Cache response
                if self.config.cache_responses:
                    self.response_cache[cache_key] = response_data
                    self.cache_timestamps[cache_key] = datetime.now()
                
                logger.info(f"Successfully scraped {url} ({len(sanitized_content)} chars)")
                return response_data
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            self.failed_urls.add(url)
            self.stats['failed_requests'] += 1
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            self.failed_urls.add(url)
            self.stats['failed_requests'] += 1
            return None
    
    async def scrape_urls(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs concurrently
        
        Args:
            urls: List of URLs to scrape
            **kwargs: Additional arguments for fetch_url
            
        Returns:
            List of successful responses
        """
        if not urls:
            return []
        
        logger.info(f"Starting to scrape {len(urls)} URLs")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def fetch_with_semaphore(url: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.fetch_url(url, **kwargs)
        
        # Execute all requests
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, dict):
                successful_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Exception in scraping: {result}")
        
        logger.info(f"Successfully scraped {len(successful_results)} out of {len(urls)} URLs")
        return successful_results
    
    def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract links from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                
                if self.content_validator.is_valid_url(full_url):
                    links.append(full_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}")
            return []
    
    async def crawl_website(self, start_url: str, max_depth: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Crawl a website starting from a given URL
        
        Args:
            start_url: Starting URL for crawling
            max_depth: Maximum depth to crawl (overrides config)
            
        Returns:
            List of scraped page data
        """
        max_depth = max_depth or self.config.max_depth
        
        logger.info(f"Starting website crawl from {start_url} with max depth {max_depth}")
        
        # Initialize crawl state
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        visited_urls = set()
        all_results = []
        
        while urls_to_visit and len(all_results) < self.config.max_total_pages:
            # Get current batch
            current_batch = []
            next_batch = []
            
            for url, depth in urls_to_visit:
                if depth <= max_depth and url not in visited_urls:
                    current_batch.append(url)
                    visited_urls.add(url)
                    
                    if len(current_batch) >= self.config.max_concurrent_requests:
                        break
                else:
                    if depth <= max_depth:
                        next_batch.append((url, depth))
            
            # Update remaining URLs
            urls_to_visit = next_batch + urls_to_visit[len(current_batch):]
            
            if not current_batch:
                break
            
            # Scrape current batch
            batch_results = await self.scrape_urls(current_batch)
            all_results.extend(batch_results)
            
            # Extract links from successful results for next depth level
            if max_depth > 0:
                for result in batch_results:
                    if result and 'content' in result:
                        links = self.extract_links(result['content'], result['url'])
                        for link in links:
                            if link not in visited_urls:
                                # Calculate depth for this link
                                current_depth = next((d for u, d in urls_to_visit if u == result['url']), 0)
                                urls_to_visit.append((link, current_depth + 1))
            
            logger.info(f"Crawled {len(current_batch)} pages, {len(all_results)} total, "
                       f"{len(urls_to_visit)} URLs remaining")
        
        logger.info(f"Crawl completed: {len(all_results)} pages scraped")
        return all_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        runtime = datetime.now() - self.stats['start_time']
        
        return {
            **self.stats,
            'scraped_urls_count': len(self.scraped_urls),
            'failed_urls_count': len(self.failed_urls),
            'domain_counts': dict(self.domain_counters),
            'runtime_seconds': runtime.total_seconds(),
            'success_rate': (self.stats['successful_requests'] / 
                           max(1, self.stats['total_requests'])) * 100,
            'cache_hit_rate': (self.stats['cache_hits'] / 
                             max(1, self.stats['total_requests'])) * 100
        }
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save scraping results to file"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'statistics': self.get_statistics(),
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

# Factory function for easy instantiation
def create_web_scraper(
    requests_per_second: float = 1.0,
    max_pages: int = 100,
    respect_robots: bool = True,
    **kwargs
) -> WebScraper:
    """
    Factory function to create a configured WebScraper
    
    Args:
        requests_per_second: Rate limit for requests
        max_pages: Maximum pages to scrape
        respect_robots: Whether to respect robots.txt
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured WebScraper instance
    """
    config = ScrapingConfig(
        requests_per_second=requests_per_second,
        max_total_pages=max_pages,
        respect_robots_txt=respect_robots,
        **kwargs
    )
    
    return WebScraper(config)

# Example usage
if __name__ == "__main__":
    async def main():
        # Example configuration
        config = ScrapingConfig(
            requests_per_second=0.5,
            max_total_pages=50,
            max_depth=2,
            respect_robots_txt=True
        )
        
        async with WebScraper(config) as scraper:
            # Single URL
            result = await scraper.fetch_url("https://example.com")
            if result:
                print(f"Scraped {result['url']}: {len(result['content'])} characters")
            
            # Multiple URLs
            urls = ["https://httpbin.org/json", "https://httpbin.org/html"]
            results = await scraper.scrape_urls(urls)
            print(f"Scraped {len(results)} URLs successfully")
            
            # Website crawl
            crawl_results = await scraper.crawl_website("https://example.com", max_depth=1)
            print(f"Crawled {len(crawl_results)} pages")
            
            # Statistics
            stats = scraper.get_statistics()
            print(f"Success rate: {stats['success_rate']:.1f}%")
    
    # Run example
    asyncio.run(main())