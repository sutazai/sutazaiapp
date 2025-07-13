#!/usr/bin/env python3
"""
Enhanced ML/NLP Service for SutazAI
Integrates multiple frameworks: spaCy, NLTK, Transformers, OpenCV, etc.
"""

import asyncio
import os
import io
import base64
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

# Core ML/AI imports
try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering, AutoModelForCausalLM
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available")

try:
    import cv2
    import PIL.Image
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV/PIL not available")

try:
    import gensim
    from gensim.models import Word2Vec, Doc2Vec
    from gensim.models.ldamodel import LdaModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.warning("Gensim not available")


class TaskType(str, Enum):
    """ML/NLP task types"""
    # Text Processing
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "ner"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    
    # Document Processing
    DOCUMENT_ANALYSIS = "document_analysis"
    TEXT_EXTRACTION = "text_extraction"
    KEYWORD_EXTRACTION = "keyword_extraction"
    
    # Image Processing
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_ENHANCEMENT = "image_enhancement"
    
    # Embeddings and Similarity
    TEXT_EMBEDDING = "text_embedding"
    SIMILARITY_SEARCH = "similarity_search"
    SEMANTIC_SEARCH = "semantic_search"
    
    # Topic Modeling
    TOPIC_MODELING = "topic_modeling"
    CLUSTERING = "clustering"


@dataclass
class MLTask:
    """ML/NLP task definition"""
    task_type: TaskType
    input_data: Any
    parameters: Dict[str, Any] = None
    model_name: str = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class MLNLPService:
    """Comprehensive ML/NLP service integrating multiple frameworks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.pipelines = {}
        self.initialized_frameworks = set()
        
        # Framework-specific components
        self.nlp_spacy = None
        self.sentiment_analyzer = None
        self.lemmatizer = None
        
    async def initialize(self) -> bool:
        """Initialize all available ML/NLP frameworks"""
        try:
            success = True
            
            # Initialize NLTK
            if NLTK_AVAILABLE:
                success &= await self._initialize_nltk()
            
            # Initialize spaCy
            if SPACY_AVAILABLE:
                success &= await self._initialize_spacy()
            
            # Initialize Transformers pipelines
            if TRANSFORMERS_AVAILABLE:
                success &= await self._initialize_transformers()
            
            logger.info(f"ML/NLP Service initialized. Frameworks: {list(self.initialized_frameworks)}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize ML/NLP service: {e}")
            return False
    
    async def _initialize_nltk(self) -> bool:
        """Initialize NLTK components"""
        try:
            # Download required NLTK data
            nltk_downloads = [
                'vader_lexicon', 'punkt', 'stopwords', 
                'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'
            ]
            
            for dataset in nltk_downloads:
                try:
                    nltk.download(dataset, quiet=True)
                except Exception:
                    pass  # Continue if download fails
            
            # Initialize components
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            
            self.initialized_frameworks.add("nltk")
            logger.info("NLTK initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"NLTK initialization failed: {e}")
            return False
    
    async def _initialize_spacy(self) -> bool:
        """Initialize spaCy models"""
        try:
            # Try to load English model
            model_names = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
            
            for model_name in model_names:
                try:
                    self.nlp_spacy = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            
            if self.nlp_spacy is None:
                # Fallback to blank English model
                self.nlp_spacy = English()
                logger.info("Using blank spaCy English model")
            
            self.initialized_frameworks.add("spacy")
            return True
            
        except Exception as e:
            logger.error(f"spaCy initialization failed: {e}")
            return False
    
    async def _initialize_transformers(self) -> bool:
        """Initialize Transformers pipelines"""
        try:
            # Initialize common pipelines
            pipeline_configs = {
                "sentiment": ("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english"),
                "classification": ("text-classification", "distilbert-base-uncased-finetuned-sst-2-english"),
                "ner": ("ner", "dbmdz/bert-large-cased-finetuned-conll03-english"),
                "qa": ("question-answering", "distilbert-base-cased-distilled-squad"),
                "summarization": ("summarization", "facebook/bart-large-cnn"),
                "generation": ("text-generation", "gpt2"),
                "feature_extraction": ("feature-extraction", "sentence-transformers/all-MiniLM-L6-v2")
            }
            
            for name, (task, model) in pipeline_configs.items():
                try:
                    self.pipelines[name] = pipeline(task, model=model, return_all_scores=True)
                    logger.info(f"Loaded pipeline: {name} -> {model}")
                except Exception as e:
                    logger.warning(f"Failed to load pipeline {name}: {e}")
            
            self.initialized_frameworks.add("transformers")
            return True
            
        except Exception as e:
            logger.error(f"Transformers initialization failed: {e}")
            return False
    
    async def process_task(self, task: MLTask) -> Dict[str, Any]:
        """Process ML/NLP task"""
        try:
            if task.task_type == TaskType.SENTIMENT_ANALYSIS:
                return await self._sentiment_analysis(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.TEXT_CLASSIFICATION:
                return await self._text_classification(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.NAMED_ENTITY_RECOGNITION:
                return await self._named_entity_recognition(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.QUESTION_ANSWERING:
                return await self._question_answering(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.TEXT_GENERATION:
                return await self._text_generation(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.TEXT_SUMMARIZATION:
                return await self._text_summarization(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.TEXT_EMBEDDING:
                return await self._text_embedding(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.DOCUMENT_ANALYSIS:
                return await self._document_analysis(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.KEYWORD_EXTRACTION:
                return await self._keyword_extraction(task.input_data, task.parameters)
            
            elif task.task_type == TaskType.TOPIC_MODELING:
                return await self._topic_modeling(task.input_data, task.parameters)
            
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported task type: {task.task_type}"
                }
                
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _sentiment_analysis(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sentiment analysis"""
        results = {}
        
        # NLTK VADER sentiment
        if "nltk" in self.initialized_frameworks:
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            results["nltk_vader"] = vader_scores
        
        # Transformers sentiment
        if "transformers" in self.initialized_frameworks and "sentiment" in self.pipelines:
            transformer_result = self.pipelines["sentiment"](text)
            results["transformers"] = transformer_result
        
        # Aggregate results
        if results:
            # Determine overall sentiment
            overall_sentiment = "neutral"
            confidence = 0.5
            
            if "nltk_vader" in results:
                compound = results["nltk_vader"]["compound"]
                if compound >= 0.05:
                    overall_sentiment = "positive"
                    confidence = abs(compound)
                elif compound <= -0.05:
                    overall_sentiment = "negative"
                    confidence = abs(compound)
            
            return {
                "status": "completed",
                "result": {
                    "overall_sentiment": overall_sentiment,
                    "confidence": confidence,
                    "detailed_scores": results
                }
            }
        
        return {
            "status": "error",
            "error": "No sentiment analysis frameworks available"
        }
    
    async def _text_classification(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform text classification"""
        if "transformers" in self.initialized_frameworks and "classification" in self.pipelines:
            result = self.pipelines["classification"](text)
            return {
                "status": "completed",
                "result": result
            }
        
        return {
            "status": "error",
            "error": "Text classification not available"
        }
    
    async def _named_entity_recognition(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform named entity recognition"""
        results = {}
        
        # spaCy NER
        if "spacy" in self.initialized_frameworks:
            doc = self.nlp_spacy(text)
            spacy_entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": spacy.explain(ent.label_)
                }
                for ent in doc.ents
            ]
            results["spacy"] = spacy_entities
        
        # Transformers NER
        if "transformers" in self.initialized_frameworks and "ner" in self.pipelines:
            transformer_entities = self.pipelines["ner"](text)
            results["transformers"] = transformer_entities
        
        return {
            "status": "completed",
            "result": results
        }
    
    async def _question_answering(self, input_data: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform question answering"""
        question = input_data.get("question")
        context = input_data.get("context")
        
        if not question or not context:
            return {
                "status": "error",
                "error": "Both question and context are required"
            }
        
        if "transformers" in self.initialized_frameworks and "qa" in self.pipelines:
            result = self.pipelines["qa"](question=question, context=context)
            return {
                "status": "completed",
                "result": result
            }
        
        return {
            "status": "error",
            "error": "Question answering not available"
        }
    
    async def _text_generation(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text"""
        max_length = params.get("max_length", 100)
        temperature = params.get("temperature", 0.7)
        
        if "transformers" in self.initialized_frameworks and "generation" in self.pipelines:
            result = self.pipelines["generation"](
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=50256
            )
            return {
                "status": "completed",
                "result": result
            }
        
        return {
            "status": "error",
            "error": "Text generation not available"
        }
    
    async def _text_summarization(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize text"""
        max_length = params.get("max_length", 130)
        min_length = params.get("min_length", 30)
        
        if "transformers" in self.initialized_frameworks and "summarization" in self.pipelines:
            result = self.pipelines["summarization"](
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return {
                "status": "completed",
                "result": result
            }
        
        return {
            "status": "error",
            "error": "Text summarization not available"
        }
    
    async def _text_embedding(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text embeddings"""
        if "transformers" in self.initialized_frameworks and "feature_extraction" in self.pipelines:
            embeddings = self.pipelines["feature_extraction"](text)
            # Average token embeddings
            if embeddings and len(embeddings) > 0:
                avg_embedding = np.mean(embeddings[0], axis=0).tolist()
                return {
                    "status": "completed",
                    "result": {
                        "embedding": avg_embedding,
                        "dimension": len(avg_embedding)
                    }
                }
        
        return {
            "status": "error",
            "error": "Text embedding not available"
        }
    
    async def _document_analysis(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive document analysis"""
        analysis = {}
        
        # Basic text statistics
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('.')
        words = word_tokenize(text) if NLTK_AVAILABLE else text.split()
        
        analysis["statistics"] = {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "average_words_per_sentence": len(words) / max(len(sentences), 1)
        }
        
        # Language detection and processing
        if "spacy" in self.initialized_frameworks:
            doc = self.nlp_spacy(text)
            analysis["linguistic"] = {
                "pos_tags": [(token.text, token.pos_) for token in doc[:20]],  # First 20 tokens
                "dependency_structure": [(token.text, token.dep_, token.head.text) for token in doc[:10]]
            }
        
        # Sentiment analysis
        sentiment_task = MLTask(TaskType.SENTIMENT_ANALYSIS, text)
        sentiment_result = await self.process_task(sentiment_task)
        if sentiment_result["status"] == "completed":
            analysis["sentiment"] = sentiment_result["result"]
        
        # Named entities
        ner_task = MLTask(TaskType.NAMED_ENTITY_RECOGNITION, text)
        ner_result = await self.process_task(ner_task)
        if ner_result["status"] == "completed":
            analysis["entities"] = ner_result["result"]
        
        return {
            "status": "completed",
            "result": analysis
        }
    
    async def _keyword_extraction(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract keywords from text"""
        keywords = []
        
        if "spacy" in self.initialized_frameworks:
            doc = self.nlp_spacy(text)
            # Extract tokens that are not stop words, punctuation, or spaces
            keywords = [
                token.lemma_.lower() 
                for token in doc 
                if not token.is_stop and not token.is_punct and not token.is_space
                and len(token.text) > 2
            ]
            
            # Get frequency counts
            from collections import Counter
            keyword_freq = Counter(keywords)
            top_keywords = keyword_freq.most_common(params.get("top_k", 10))
            
            return {
                "status": "completed",
                "result": {
                    "keywords": [{"word": word, "frequency": freq} for word, freq in top_keywords],
                    "total_unique_keywords": len(keyword_freq)
                }
            }
        
        return {
            "status": "error",
            "error": "Keyword extraction not available"
        }
    
    async def _topic_modeling(self, texts: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform topic modeling on multiple texts"""
        if not GENSIM_AVAILABLE:
            return {
                "status": "error",
                "error": "Gensim not available for topic modeling"
            }
        
        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                if NLTK_AVAILABLE:
                    tokens = word_tokenize(text.lower())
                    # Remove stopwords and short tokens
                    stop_words = set(stopwords.words('english'))
                    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
                else:
                    tokens = text.lower().split()
                
                processed_texts.append(tokens)
            
            # Create dictionary and corpus
            dictionary = Dictionary(processed_texts)
            corpus = [dictionary.doc2bow(text) for text in processed_texts]
            
            # Train LDA model
            num_topics = params.get("num_topics", 5)
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            # Extract topics
            topics = []
            for idx, topic in lda_model.print_topics(-1):
                topics.append({
                    "topic_id": idx,
                    "words": topic
                })
            
            return {
                "status": "completed",
                "result": {
                    "topics": topics,
                    "num_topics": num_topics,
                    "perplexity": lda_model.log_perplexity(corpus)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Topic modeling failed: {str(e)}"
            }
    
    def get_available_frameworks(self) -> List[str]:
        """Get list of available ML/NLP frameworks"""
        return list(self.initialized_frameworks)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "status": "healthy" if self.initialized_frameworks else "unhealthy",
            "available_frameworks": list(self.initialized_frameworks),
            "available_pipelines": list(self.pipelines.keys()) if hasattr(self, 'pipelines') else [],
            "supported_tasks": [task.value for task in TaskType]
        }


# Factory function for service registration
def create_ml_nlp_service(config: Dict[str, Any] = None) -> MLNLPService:
    """Factory function to create ML/NLP service"""
    return MLNLPService(config or {})