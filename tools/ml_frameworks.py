import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import asyncio
from abc import ABC, abstractmethod

# Framework imports (will be installed as needed)
try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from config import config

logger = logging.getLogger(__name__)

@dataclass
class MLModelInfo:
    name: str
    framework: str
    model_type: str
    capabilities: List[str]
    model_path: Optional[str] = None
    is_loaded: bool = False
    performance_metrics: Dict[str, float] = None
    memory_usage: Optional[int] = None

@dataclass
class NLPProcessingResult:
    text: str
    tokens: List[str]
    entities: List[Dict[str, Any]]
    sentiment: Dict[str, float]
    embeddings: Optional[np.ndarray] = None
    language: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = None

class MLFrameworkManager:
    """Unified manager for multiple ML/NLP frameworks."""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.framework_status = {
            'pytorch': HAS_PYTORCH,
            'tensorflow': HAS_TENSORFLOW,
            'spacy': HAS_SPACY,
            'nltk': HAS_NLTK,
            'transformers': HAS_TRANSFORMERS,
            'onnx': HAS_ONNX
        }
        self.nlp_pipelines = {}
        self._initialize_frameworks()
    
    def _initialize_frameworks(self):
        """Initialize available frameworks."""
        try:
            if HAS_SPACY:
                self._initialize_spacy()
            if HAS_NLTK:
                self._initialize_nltk()
            if HAS_TRANSFORMERS:
                self._initialize_transformers()
            if HAS_ONNX:
                self._initialize_onnx()
        except Exception as e:
            logger.error(f"Error initializing ML frameworks: {e}")
    
    def _initialize_spacy(self):
        """Initialize spaCy models."""
        try:
            # Try to load English model
            self.nlp_pipelines['spacy_en'] = spacy.load('en_core_web_sm')
            logger.info("spaCy English model loaded successfully")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
    
    def _initialize_nltk(self):
        """Initialize NLTK components."""
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            logger.info("NLTK components downloaded successfully")
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")
    
    def _initialize_transformers(self):
        """Initialize Transformers models."""
        try:
            # Load sentiment analysis pipeline
            self.nlp_pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Load text generation pipeline
            self.nlp_pipelines['text_generation'] = pipeline(
                "text-generation",
                model="gpt2"
            )
            
            # Load embeddings model
            self.nlp_pipelines['embeddings'] = pipeline(
                "feature-extraction",
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            logger.info("Transformers pipelines loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Transformers models: {e}")
    
    def _initialize_onnx(self):
        """Initialize ONNX runtime."""
        try:
            # ONNX runtime is available
            self.onnx_session = None
            logger.info("ONNX runtime available")
        except Exception as e:
            logger.error(f"Error initializing ONNX: {e}")
    
    async def process_text_comprehensive(self, text: str) -> NLPProcessingResult:
        """Comprehensive text processing using all available frameworks."""
        result = NLPProcessingResult(
            text=text,
            tokens=[],
            entities=[],
            sentiment={},
            keywords=[]
        )
        
        try:
            # spaCy processing
            if 'spacy_en' in self.nlp_pipelines:
                result = await self._process_with_spacy(text, result)
            
            # NLTK processing
            if HAS_NLTK:
                result = await self._process_with_nltk(text, result)
            
            # Transformers processing
            if HAS_TRANSFORMERS:
                result = await self._process_with_transformers(text, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive text processing: {e}")
            return result
    
    async def _process_with_spacy(self, text: str, result: NLPProcessingResult) -> NLPProcessingResult:
        """Process text with spaCy."""
        try:
            nlp = self.nlp_pipelines['spacy_en']
            doc = nlp(text)
            
            # Extract tokens
            result.tokens = [token.text for token in doc]
            
            # Extract named entities
            for ent in doc.ents:
                result.entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0
                })
            
            # Extract keywords (using noun phrases)
            result.keywords = [chunk.text for chunk in doc.noun_chunks]
            
            # Language detection
            result.language = doc.lang_
            
        except Exception as e:
            logger.error(f"Error processing with spaCy: {e}")
        
        return result
    
    async def _process_with_nltk(self, text: str, result: NLPProcessingResult) -> NLPProcessingResult:
        """Process text with NLTK."""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Sentiment analysis
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = sia.polarity_scores(text)
            result.sentiment.update({
                'nltk_compound': sentiment_scores['compound'],
                'nltk_positive': sentiment_scores['pos'],
                'nltk_negative': sentiment_scores['neg'],
                'nltk_neutral': sentiment_scores['neu']
            })
            
            # Enhanced tokenization if not already done
            if not result.tokens:
                result.tokens = word_tokenize(text)
            
        except Exception as e:
            logger.error(f"Error processing with NLTK: {e}")
        
        return result
    
    async def _process_with_transformers(self, text: str, result: NLPProcessingResult) -> NLPProcessingResult:
        """Process text with Transformers."""
        try:
            # Sentiment analysis
            if 'sentiment' in self.nlp_pipelines:
                sentiment_result = self.nlp_pipelines['sentiment'](text)
                if sentiment_result:
                    result.sentiment.update({
                        'transformers_label': sentiment_result[0]['label'],
                        'transformers_score': sentiment_result[0]['score']
                    })
            
            # Generate embeddings
            if 'embeddings' in self.nlp_pipelines:
                embeddings = self.nlp_pipelines['embeddings'](text)
                if embeddings:
                    result.embeddings = np.array(embeddings[0])
            
        except Exception as e:
            logger.error(f"Error processing with Transformers: {e}")
        
        return result
    
    async def generate_text(self, prompt: str, max_length: int = 100, framework: str = "transformers") -> str:
        """Generate text using specified framework."""
        try:
            if framework == "transformers" and 'text_generation' in self.nlp_pipelines:
                result = self.nlp_pipelines['text_generation'](
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7
                )
                return result[0]['generated_text']
            
            # Fallback to basic completion
            return f"{prompt} [Generated text would continue here...]"
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"{prompt} [Error in text generation]"
    
    async def analyze_code_quality(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code quality using ML techniques."""
        try:
            analysis_result = {
                'complexity_score': 0.0,
                'readability_score': 0.0,
                'security_issues': [],
                'suggestions': [],
                'performance_hints': []
            }
            
            # Basic complexity analysis
            lines = code.split('\n')
            analysis_result['complexity_score'] = min(100.0, len(lines) * 2.5)
            
            # Use NLP to analyze comments and variable names
            nlp_result = await self.process_text_comprehensive(code)
            
            # Readability based on NLP analysis
            if nlp_result.sentiment:
                analysis_result['readability_score'] = max(0.0, 
                    nlp_result.sentiment.get('nltk_compound', 0.0) * 100)
            
            # Generate suggestions using text analysis
            if 'def ' in code or 'class ' in code:
                analysis_result['suggestions'].append(
                    "Consider adding more descriptive docstrings"
                )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing code quality: {e}")
            return {'error': str(e)}
    
    async def create_onnx_model(self, pytorch_model: Any, input_shape: tuple, model_name: str) -> bool:
        """Convert PyTorch model to ONNX format."""
        try:
            if not HAS_PYTORCH or not HAS_ONNX:
                logger.error("PyTorch or ONNX not available for model conversion")
                return False
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Export to ONNX
            onnx_path = Path(config.storage.models_path) / f"{model_name}.onnx"
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            logger.info(f"Model exported to ONNX: {onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ONNX model: {e}")
            return False
    
    async def load_onnx_model(self, model_path: str) -> bool:
        """Load ONNX model for inference."""
        try:
            if not HAS_ONNX:
                logger.error("ONNX runtime not available")
                return False
            
            self.onnx_session = ort.InferenceSession(model_path)
            logger.info(f"ONNX model loaded: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            return False
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get status of all ML frameworks."""
        return {
            'frameworks': self.framework_status,
            'loaded_models': list(self.loaded_models.keys()),
            'available_pipelines': list(self.nlp_pipelines.keys()),
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, str]:
        """Get approximate memory usage of loaded models."""
        try:
            import psutil
            process = psutil.Process()
            return {
                'total_memory': f"{process.memory_info().rss / 1024 / 1024:.2f} MB",
                'cpu_percent': f"{process.cpu_percent():.2f}%"
            }
        except ImportError:
            return {'info': 'psutil not available for memory monitoring'}
    
    async def benchmark_frameworks(self, test_text: str) -> Dict[str, Any]:
        """Benchmark different frameworks on the same task."""
        benchmarks = {}
        
        try:
            # Benchmark spaCy
            if 'spacy_en' in self.nlp_pipelines:
                start_time = datetime.utcnow()
                nlp = self.nlp_pipelines['spacy_en']
                doc = nlp(test_text)
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                benchmarks['spacy'] = {
                    'processing_time': processing_time,
                    'entities_found': len(doc.ents),
                    'tokens_processed': len(doc)
                }
            
            # Benchmark Transformers
            if 'sentiment' in self.nlp_pipelines:
                start_time = datetime.utcnow()
                result = self.nlp_pipelines['sentiment'](test_text)
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                benchmarks['transformers_sentiment'] = {
                    'processing_time': processing_time,
                    'confidence': result[0]['score'] if result else 0.0
                }
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error benchmarking frameworks: {e}")
            return {'error': str(e)}

class TensorFlowIntegration:
    """TensorFlow-specific integration for advanced ML tasks."""
    
    def __init__(self):
        self.models = {}
        self.available = HAS_TENSORFLOW
    
    async def create_text_classifier(self, num_classes: int = 2) -> bool:
        """Create a TensorFlow text classification model."""
        try:
            if not HAS_TENSORFLOW:
                return False
            
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(10000, 64),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.models['text_classifier'] = model
            logger.info("TensorFlow text classifier created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating TensorFlow model: {e}")
            return False

class PyTorchIntegration:
    """PyTorch-specific integration for deep learning tasks."""
    
    def __init__(self):
        self.models = {}
        self.available = HAS_PYTORCH
    
    async def create_neural_network(self, input_size: int, hidden_size: int, output_size: int) -> bool:
        """Create a PyTorch neural network."""
        try:
            if not HAS_PYTORCH:
                return False
            
            class SimpleNN(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(SimpleNN, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, output_size)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
            
            model = SimpleNN(input_size, hidden_size, output_size)
            self.models['simple_nn'] = model
            logger.info("PyTorch neural network created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating PyTorch model: {e}")
            return False

# Global instances
ml_framework_manager = MLFrameworkManager()
tensorflow_integration = TensorFlowIntegration()
pytorch_integration = PyTorchIntegration()

# Convenience functions
async def process_text(text: str) -> NLPProcessingResult:
    """Process text using the best available framework."""
    return await ml_framework_manager.process_text_comprehensive(text)

async def generate_text(prompt: str, max_length: int = 100) -> str:
    """Generate text using available frameworks."""
    return await ml_framework_manager.generate_text(prompt, max_length)

async def analyze_code(code: str, language: str = "python") -> Dict[str, Any]:
    """Analyze code quality using ML techniques."""
    return await ml_framework_manager.analyze_code_quality(code, language)

async def get_ml_status() -> Dict[str, Any]:
    """Get status of all ML frameworks and models."""
    return ml_framework_manager.get_framework_status()