"""
Web Learning Module for SutazAI V7
Self-supervised learning pipeline with web scraping and browser automation
"""

from .web_scraper import WebScraper, ScrapingConfig
from .content_processor import ContentProcessor, ContentType
from .learning_pipeline import SelfSupervisedLearningPipeline
from .knowledge_extractor import KnowledgeExtractor
from .adaptive_learner import AdaptiveLearner
from .web_automation import WebAutomation

__all__ = [
    "WebScraper",
    "ScrapingConfig", 
    "ContentProcessor",
    "ContentType",
    "SelfSupervisedLearningPipeline",
    "KnowledgeExtractor",
    "AdaptiveLearner",
    "WebAutomation"
]