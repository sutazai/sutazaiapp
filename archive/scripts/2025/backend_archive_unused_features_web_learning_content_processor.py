#!/usr/bin/env python3
"""
Content Processor for SutazAI V7 Self-Supervised Learning
Processes and analyzes web content for knowledge extraction
"""

import os
import sys
import re
import json
import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import mimetypes

try:
    from bs4 import BeautifulSoup
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content that can be processed"""
    HTML = "html"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    PDF = "pdf"
    MARKDOWN = "markdown"
    RSS = "rss"
    UNKNOWN = "unknown"

@dataclass
class ProcessedContent:
    """Processed content with extracted information"""
    url: str
    content_type: ContentType
    title: str = ""
    text_content: str = ""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    sentiment: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    structured_data: Dict[str, Any] = field(default_factory=dict)
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    word_count: int = 0
    reading_time: int = 0  # in minutes
    quality_score: float = 0.0

@dataclass
class ContentAnalysis:
    """Analysis results for content"""
    complexity_score: float = 0.0
    readability_score: float = 0.0
    information_density: float = 0.0
    uniqueness_score: float = 0.0
    relevance_score: float = 0.0
    factual_indicators: List[str] = field(default_factory=list)
    opinion_indicators: List[str] = field(default_factory=list)
    temporal_references: List[str] = field(default_factory=list)
    numerical_data: List[Dict[str, Any]] = field(default_factory=list)

class HTMLProcessor:
    """Processes HTML content"""
    
    def __init__(self):
        self.content_tags = ['p', 'div', 'article', 'section', 'main', 'span']
        self.header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        self.exclude_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']
        
    def extract_text(self, html_content: str) -> str:
        """Extract clean text from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted tags
            for tag in soup(self.exclude_tags):
                tag.decompose()
            
            # Extract text from content areas
            text_parts = []
            
            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
            
            if main_content:
                text_parts.append(main_content.get_text(separator=' ', strip=True))
            else:
                # Fall back to extracting from all content tags
                for tag in soup.find_all(self.content_tags):
                    text = tag.get_text(separator=' ', strip=True)
                    if text and len(text) > 20:  # Filter out very short texts
                        text_parts.append(text)
            
            # Join and clean text
            full_text = ' '.join(text_parts)
            
            # Clean up whitespace
            full_text = re.sub(r'\s+', ' ', full_text)
            full_text = re.sub(r'\n+', '\n', full_text)
            
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def extract_title(self, html_content: str) -> str:
        """Extract title from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try different title sources
            title_sources = [
                soup.find('title'),
                soup.find('h1'),
                soup.find('meta', property='og:title'),
                soup.find('meta', name='twitter:title')
            ]
            
            for source in title_sources:
                if source:
                    title = source.get_text(strip=True) if hasattr(source, 'get_text') else source.get('content', '')
                    if title and len(title) > 0:
                        return title
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return ""
    
    def extract_metadata(self, html_content: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = {}
            
            # Standard meta tags
            for meta in soup.find_all('meta'):
                if meta.get('name'):
                    metadata[meta.get('name')] = meta.get('content', '')
                elif meta.get('property'):
                    metadata[meta.get('property')] = meta.get('content', '')
            
            # Structured data (JSON-LD)
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    structured_data = json.loads(script.string)
                    metadata['structured_data'] = structured_data
                except:
                    pass
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract links from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http'):
                    links.append(href)
                elif href.startswith('/'):
                    # Resolve relative URLs
                    from urllib.parse import urljoin
                    full_url = urljoin(base_url, href)
                    links.append(full_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []
    
    def extract_images(self, html_content: str, base_url: str) -> List[str]:
        """Extract image URLs from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            images = []
            
            for img in soup.find_all('img', src=True):
                src = img['src']
                if src.startswith('http'):
                    images.append(src)
                elif src.startswith('/'):
                    from urllib.parse import urljoin
                    full_url = urljoin(base_url, src)
                    images.append(full_url)
            
            return list(set(images))
            
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return []

class TextAnalyzer:
    """Analyzes text content for various features"""
    
    def __init__(self):
        self.lemmatizer = None
        self.stop_words = None
        
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                logger.warning("NLTK data not available. Some features will be limited.")
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords from text"""
        if not NLTK_AVAILABLE or not text:
            return []
        
        try:
            # Tokenize and clean
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and len(word) > 3]
            
            # Remove stop words
            if self.stop_words:
                words = [word for word in words if word not in self.stop_words]
            
            # Lemmatize
            if self.lemmatizer:
                words = [self.lemmatizer.lemmatize(word) for word in words]
            
            # Count frequency
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, freq in keywords[:max_keywords]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        if not NLTK_AVAILABLE or not text:
            return []
        
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity recognition
            entities = []
            chunks = ne_chunk(pos_tags)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join([token for token, pos in chunk])
                    entities.append({
                        'text': entity_name,
                        'type': chunk.label(),
                        'confidence': 0.8  # Placeholder confidence
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def calculate_readability(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease)"""
        if not text:
            return 0.0
        
        try:
            # Count sentences, words, and syllables
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            # Approximate syllable count
            syllables = 0
            for word in words:
                syllables += self._count_syllables(word)
            
            # Flesch Reading Ease formula
            if len(sentences) > 0 and len(words) > 0:
                score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
                return max(0, min(100, score))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        
        if word[0] in vowels:
            count += 1
        
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        
        if word.endswith("e"):
            count -= 1
        
        if count == 0:
            count = 1
        
        return count
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (simplified implementation)"""
        if not text:
            return 0.0
        
        # Simple sentiment analysis based on word lists
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'best', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'worst', 'horrible', 'disgusting', 'annoying', 'stupid']
        
        words = word_tokenize(text.lower()) if NLTK_AVAILABLE else text.lower().split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_sentiment_words
    
    def extract_numerical_data(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical data from text"""
        numerical_data = []
        
        # Pattern for numbers with units
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(percent|%)', 'percentage'),
            (r'(\d+(?:\.\d+)?)\s*(million|billion|trillion)', 'large_number'),
            (r'\$(\d+(?:\.\d+)?)', 'currency'),
            (r'(\d+(?:\.\d+)?)\s*(kg|km|m|cm|mm|g|lb|ft|in)', 'measurement'),
            (r'(\d{4})', 'year'),
            (r'(\d+(?:\.\d+)?)', 'number')
        ]
        
        for pattern, data_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                numerical_data.append({
                    'value': match.group(1),
                    'type': data_type,
                    'context': text[max(0, match.start()-50):match.end()+50],
                    'position': match.start()
                })
        
        return numerical_data

class ContentProcessor:
    """
    Main content processor that handles different content types
    and integrates with the biological neural network
    """
    
    def __init__(self):
        self.html_processor = HTMLProcessor()
        self.text_analyzer = TextAnalyzer()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'content_types': {},
            'start_time': datetime.now()
        }
        
        logger.info("ContentProcessor initialized")
    
    def detect_content_type(self, content: str, content_type_header: str = "") -> ContentType:
        """Detect content type from content and headers"""
        try:
            # Check header first
            if content_type_header:
                header_lower = content_type_header.lower()
                if 'html' in header_lower:
                    return ContentType.HTML
                elif 'json' in header_lower:
                    return ContentType.JSON
                elif 'xml' in header_lower:
                    return ContentType.XML
                elif 'text' in header_lower:
                    return ContentType.TEXT
            
            # Analyze content
            content_lower = content.lower().strip()
            
            if content_lower.startswith('<!doctype html') or content_lower.startswith('<html'):
                return ContentType.HTML
            elif content_lower.startswith('{') and content_lower.endswith('}'):
                return ContentType.JSON
            elif content_lower.startswith('<') and content_lower.endswith('>'):
                return ContentType.XML
            elif content_lower.startswith('<?xml'):
                return ContentType.XML
            elif '# ' in content or content.count('*') > 5:
                return ContentType.MARKDOWN
            else:
                return ContentType.TEXT
                
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return ContentType.UNKNOWN
    
    async def process_content(self, url: str, content: str, content_type_header: str = "") -> ProcessedContent:
        """
        Process content and extract all relevant information
        
        Args:
            url: Source URL of the content
            content: Raw content to process
            content_type_header: Content-Type header value
            
        Returns:
            ProcessedContent object with extracted information
        """
        self.stats['total_processed'] += 1
        
        try:
            # Detect content type
            content_type = self.detect_content_type(content, content_type_header)
            
            # Update stats
            self.stats['content_types'][content_type.value] = \
                self.stats['content_types'].get(content_type.value, 0) + 1
            
            # Initialize processed content
            processed = ProcessedContent(
                url=url,
                content_type=content_type
            )
            
            # Process based on content type
            if content_type == ContentType.HTML:
                await self._process_html_content(processed, content)
            elif content_type == ContentType.JSON:
                await self._process_json_content(processed, content)
            elif content_type == ContentType.TEXT:
                await self._process_text_content(processed, content)
            elif content_type == ContentType.XML:
                await self._process_xml_content(processed, content)
            else:
                # Default text processing
                await self._process_text_content(processed, content)
            
            # Common analysis for all content types
            await self._analyze_content(processed)
            
            # Calculate quality metrics
            processed.quality_score = self._calculate_quality_score(processed)
            
            self.stats['successful_processed'] += 1
            
            logger.debug(f"Successfully processed content from {url}")
            return processed
            
        except Exception as e:
            logger.error(f"Error processing content from {url}: {e}")
            self.stats['failed_processed'] += 1
            
            # Return minimal processed content
            return ProcessedContent(
                url=url,
                content_type=ContentType.UNKNOWN,
                text_content=content[:1000] if content else "",
                quality_score=0.0
            )
    
    async def _process_html_content(self, processed: ProcessedContent, content: str):
        """Process HTML content"""
        processed.title = self.html_processor.extract_title(content)
        processed.text_content = self.html_processor.extract_text(content)
        processed.metadata = self.html_processor.extract_metadata(content)
        processed.links = self.html_processor.extract_links(content, processed.url)
        processed.images = self.html_processor.extract_images(content, processed.url)
        
        # Extract structured data
        if 'structured_data' in processed.metadata:
            processed.structured_data = processed.metadata['structured_data']
    
    async def _process_json_content(self, processed: ProcessedContent, content: str):
        """Process JSON content"""
        try:
            data = json.loads(content)
            processed.structured_data = data
            
            # Extract text from JSON values
            text_parts = []
            self._extract_text_from_json(data, text_parts)
            processed.text_content = ' '.join(text_parts)
            
            # Try to find title
            title_keys = ['title', 'name', 'headline', 'subject']
            for key in title_keys:
                if key in data and isinstance(data[key], str):
                    processed.title = data[key]
                    break
                    
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON content from {processed.url}")
            processed.text_content = content
    
    async def _process_text_content(self, processed: ProcessedContent, content: str):
        """Process plain text content"""
        processed.text_content = content
        
        # Try to extract title from first line
        lines = content.split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100:  # Likely a title
                processed.title = first_line
    
    async def _process_xml_content(self, processed: ProcessedContent, content: str):
        """Process XML content"""
        try:
            soup = BeautifulSoup(content, 'xml')
            
            # Extract text content
            processed.text_content = soup.get_text(separator=' ', strip=True)
            
            # Try to find title
            title_tag = soup.find('title') or soup.find('name')
            if title_tag:
                processed.title = title_tag.get_text(strip=True)
                
        except Exception as e:
            logger.warning(f"Error parsing XML from {processed.url}: {e}")
            processed.text_content = content
    
    def _extract_text_from_json(self, data: Any, text_parts: List[str]):
        """Recursively extract text from JSON data"""
        if isinstance(data, dict):
            for value in data.values():
                self._extract_text_from_json(value, text_parts)
        elif isinstance(data, list):
            for item in data:
                self._extract_text_from_json(item, text_parts)
        elif isinstance(data, str) and len(data) > 10:
            text_parts.append(data)
    
    async def _analyze_content(self, processed: ProcessedContent):
        """Perform comprehensive content analysis"""
        if not processed.text_content:
            return
        
        # Basic text statistics
        processed.word_count = len(processed.text_content.split())
        processed.reading_time = max(1, processed.word_count // 200)  # Assume 200 WPM
        
        # Extract keywords
        processed.keywords = self.text_analyzer.extract_keywords(processed.text_content)
        
        # Extract entities
        processed.entities = self.text_analyzer.extract_entities(processed.text_content)
        
        # Sentiment analysis
        processed.sentiment = self.text_analyzer.analyze_sentiment(processed.text_content)
        
        # Extract numerical data
        numerical_data = self.text_analyzer.extract_numerical_data(processed.text_content)
        processed.metadata['numerical_data'] = numerical_data
        
        # Generate summary (simple extractive approach)
        processed.summary = self._generate_summary(processed.text_content)
        
        # Topic extraction (simplified)
        processed.topics = self._extract_topics(processed.text_content, processed.keywords)
    
    def _generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary"""
        if not text or not NLTK_AVAILABLE:
            return text[:500] + "..." if len(text) > 500 else text
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return text
            
            # Score sentences based on keyword frequency
            word_freq = {}
            words = word_tokenize(text.lower())
            
            for word in words:
                if word.isalpha() and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score each sentence
            sentence_scores = {}
            for sentence in sentences:
                score = 0
                words = word_tokenize(sentence.lower())
                for word in words:
                    if word in word_freq:
                        score += word_freq[word]
                
                if len(words) > 0:
                    sentence_scores[sentence] = score / len(words)
            
            # Select top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
            
            # Maintain original order
            summary_sentences = []
            for sentence in sentences:
                if any(sentence == s for s, score in top_sentences):
                    summary_sentences.append(sentence)
                    if len(summary_sentences) >= max_sentences:
                        break
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:500] + "..." if len(text) > 500 else text
    
    def _extract_topics(self, text: str, keywords: List[str]) -> List[str]:
        """Extract topics from text using keywords"""
        if not keywords:
            return []
        
        # Simple topic extraction based on keyword clustering
        topics = []
        
        # Group related keywords
        topic_groups = {
            'technology': ['software', 'computer', 'digital', 'internet', 'web', 'app', 'system', 'data', 'programming'],
            'science': ['research', 'study', 'experiment', 'discovery', 'analysis', 'method', 'theory'],
            'business': ['market', 'company', 'industry', 'economic', 'finance', 'revenue', 'profit', 'investment'],
            'health': ['medical', 'health', 'patient', 'treatment', 'disease', 'medicine', 'doctor', 'hospital'],
            'education': ['learning', 'student', 'school', 'university', 'education', 'teaching', 'academic']
        }
        
        for topic, topic_keywords in topic_groups.items():
            matches = sum(1 for keyword in keywords if keyword in topic_keywords)
            if matches >= 2:
                topics.append(topic)
        
        return topics
    
    def _calculate_quality_score(self, processed: ProcessedContent) -> float:
        """Calculate content quality score"""
        score = 0.0
        
        # Length factor
        if processed.word_count > 100:
            score += 0.2
        if processed.word_count > 500:
            score += 0.2
        
        # Title presence
        if processed.title:
            score += 0.1
        
        # Metadata richness
        if processed.metadata:
            score += 0.1
        
        # Keyword extraction success
        if processed.keywords:
            score += 0.1
        
        # Entity extraction success
        if processed.entities:
            score += 0.1
        
        # Structured data presence
        if processed.structured_data:
            score += 0.1
        
        # Links and images
        if processed.links:
            score += 0.05
        if processed.images:
            score += 0.05
        
        return min(1.0, score)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        runtime = datetime.now() - self.stats['start_time']
        
        return {
            **self.stats,
            'runtime_seconds': runtime.total_seconds(),
            'success_rate': (self.stats['successful_processed'] / 
                           max(1, self.stats['total_processed'])) * 100,
            'average_processing_time': runtime.total_seconds() / max(1, self.stats['total_processed'])
        }

# Factory function
def create_content_processor() -> ContentProcessor:
    """Create a configured ContentProcessor instance"""
    return ContentProcessor()

# Example usage
if __name__ == "__main__":
    async def main():
        processor = create_content_processor()
        
        # Example HTML content
        html_content = """
        <html>
        <head><title>Test Article</title></head>
        <body>
            <h1>Machine Learning Breakthrough</h1>
            <p>Researchers at MIT have developed a new neural network architecture that shows 
            promising results in natural language processing tasks.</p>
            <p>The system achieved 95% accuracy on benchmark tests, representing a significant 
            improvement over previous methods.</p>
        </body>
        </html>
        """
        
        # Process content
        result = await processor.process_content("https://example.com/article", html_content, "text/html")
        
        print(f"Title: {result.title}")
        print(f"Word Count: {result.word_count}")
        print(f"Keywords: {result.keywords}")
        print(f"Summary: {result.summary}")
        print(f"Quality Score: {result.quality_score}")
        print(f"Topics: {result.topics}")
        
        # Statistics
        stats = processor.get_statistics()
        print(f"Processing Success Rate: {stats['success_rate']:.1f}%")
    
    asyncio.run(main())