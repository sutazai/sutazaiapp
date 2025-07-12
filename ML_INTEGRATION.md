# SutazAI ML Framework Integration

This document describes the comprehensive ML/NLP framework integration in SutazAI v3+ system.

## Overview

SutazAI now includes advanced machine learning and natural language processing capabilities through integration with multiple industry-standard frameworks:

### Core Frameworks
- **PyTorch** - Deep learning and neural networks
- **TensorFlow** - Machine learning platform with GPU acceleration
- **Transformers** - State-of-the-art NLP models (BERT, GPT, etc.)
- **spaCy** - Industrial-strength NLP
- **NLTK** - Comprehensive NLP toolkit
- **ONNX** - Model portability and optimization

## Features

### 🤖 Intelligent Code Analysis
- **ML-powered code quality assessment**
- **Security vulnerability detection using AI**
- **Performance optimization suggestions**
- **Automated documentation generation**

### 📝 Advanced Text Processing
- **Multi-framework sentiment analysis**
- **Named entity recognition**
- **Language detection (165+ languages)**
- **Semantic similarity and embeddings**
- **Text summarization**

### 🔍 Semantic Search
- **Vector-based document search**
- **Context-aware information retrieval**
- **Cross-lingual search capabilities**

### 🛡️ Security Intelligence
- **AI-powered vulnerability scanning**
- **Pattern-based threat detection**
- **Code security scoring**

### ⚡ Performance Optimization
- **Code complexity analysis**
- **Performance bottleneck detection**
- **Optimization recommendations**

## Installation

### Quick Setup
```bash
# Install ML frameworks
cd /opt/sutazaiapp
python install_ml_frameworks.py

# Or install manually
pip install -r requirements-ml.txt

# Download spaCy models
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('all')"
```

### GPU Support
For CUDA-enabled systems:
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

## API Endpoints

### Text Analysis
```bash
POST /api/v1/ml/analyze/text
{
    "text": "Your text here",
    "include_embeddings": true
}
```

### Code Analysis
```bash
POST /api/v1/ml/analyze/code
{
    "code": "def hello(): print('world')",
    "language": "python",
    "detailed": true
}
```

### Security Analysis
```bash
POST /api/v1/ml/analyze/security
{
    "code": "your code here",
    "language": "python"
}
```

### Text Generation
```bash
POST /api/v1/ml/generate/text
{
    "prompt": "Write a function that",
    "max_length": 100,
    "framework": "transformers"
}
```

### Semantic Search
```bash
POST /api/v1/ml/search/semantic
{
    "query": "machine learning algorithms",
    "limit": 10,
    "similarity_threshold": 0.7
}
```

## Configuration

### Environment Variables
```bash
# GPU Settings
USE_GPU=true
MAX_GPU_MEMORY=8GB
TORCH_DEVICE=auto

# Model Cache
TRANSFORMERS_CACHE=./data/transformers_cache
SPACY_MODEL_CACHE=./data/spacy_models

# Processing Limits
MAX_TEXT_LENGTH=1000000
MAX_BATCH_SIZE=32
```

### Framework Configuration
```python
ML_CONFIG = {
    "pytorch": {
        "device": "auto",
        "memory_fraction": 0.8
    },
    "transformers": {
        "model_max_length": 512,
        "batch_size": 32
    },
    "spacy": {
        "model_name": "en_core_web_sm",
        "disable": ["parser", "tagger"]  # for production
    }
}
```

## Usage Examples

### Python Agent Integration
```python
from agents.ml_agent import ml_analysis_agent

# Analyze code
task = {
    "type": "analyze_code",
    "code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    "language": "python"
}

result = await ml_analysis_agent.execute_task(task)
print(f"Quality Score: {result['analysis']['quality_score']}")
```

### Direct Framework Usage
```python
from tools.ml_frameworks import process_text, analyze_code

# Text processing
text_result = await process_text("AI will revolutionize software development")
print(f"Sentiment: {text_result.sentiment}")
print(f"Entities: {text_result.entities}")

# Code analysis
code_result = await analyze_code("print('hello world')", "python")
print(f"Complexity: {code_result['complexity_score']}")
```

## Supported Models

### Pre-trained Models
- **BERT**: `bert-base-uncased`
- **RoBERTa**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **GPT-2**: `gpt2`
- **Sentence Transformers**: `all-MiniLM-L6-v2`
- **spaCy English**: `en_core_web_sm`

### Custom Model Support
- Load custom PyTorch models
- Convert models to ONNX format
- Support for Hugging Face model hub

## Performance Monitoring

### Framework Status
```bash
GET /api/v1/ml/status
```

### Benchmark Frameworks
```bash
POST /api/v1/ml/benchmark
{
    "test_text": "Sample text for benchmarking"
}
```

### Available Models
```bash
GET /api/v1/ml/models/available
```

## Best Practices

### Memory Management
- Use GPU when available for large models
- Enable memory growth for TensorFlow
- Cache models to avoid reload overhead
- Monitor memory usage with built-in tools

### Performance Optimization
- Batch process when possible
- Use ONNX for inference optimization
- Disable unused spaCy components in production
- Set appropriate timeout values

### Security Considerations
- Validate input text length
- Sanitize code inputs
- Monitor resource usage
- Use secure model sources

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
config.MAX_BATCH_SIZE = 16

# Use CPU fallback
config.TORCH_DEVICE = "cpu"
```

#### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

#### NLTK Data Missing
```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
```

### Performance Issues
- Check GPU availability: `torch.cuda.is_available()`
- Monitor memory: `GET /api/v1/ml/status`
- Reduce model complexity in production
- Use framework-specific optimizations

## Advanced Features

### ONNX Model Conversion
```python
# Convert PyTorch to ONNX
await ml_framework_manager.create_onnx_model(
    pytorch_model=model,
    input_shape=(1, 784),
    model_name="custom_model"
)
```

### Multi-language Support
- 165+ languages with spaCy/Polyglot
- Language detection
- Cross-lingual embeddings
- Translation capabilities (with additional models)

### Custom Pipelines
```python
# Create custom analysis pipeline
pipeline = Pipeline([
    ("tokenizer", SpacyTokenizer()),
    ("embedder", TransformerEmbedder()),
    ("classifier", CustomClassifier())
])
```

## Integration Examples

### Chat Enhancement
```python
# Enhanced chat with ML analysis
async def enhanced_chat_response(message):
    # Analyze message sentiment and entities
    analysis = await process_text(message)
    
    # Generate context-aware response
    if analysis.sentiment.get('nltk_compound', 0) < -0.5:
        response_style = "supportive"
    else:
        response_style = "informative"
    
    return await generate_response(message, style=response_style)
```

### Code Review Automation
```python
# Automated code review
async def review_code(code, language):
    analysis = await analyze_code(code, language)
    security = await assess_security(code)
    
    return {
        "quality_score": analysis["quality_score"],
        "security_issues": security["vulnerabilities"],
        "recommendations": analysis["suggestions"] + security["recommendations"]
    }
```

## Roadmap

### Planned Enhancements
- [ ] Computer vision integration (OpenCV, YOLO)
- [ ] Audio processing (speech-to-text, TTS)
- [ ] Time series analysis
- [ ] Reinforcement learning capabilities
- [ ] AutoML integration
- [ ] Model fine-tuning APIs

### Version History
- **v3.0**: Initial ML framework integration
- **v3.1**: ONNX support and optimization
- **v3.2**: Multi-language NLP support
- **v3.3**: Advanced security analysis

## Support

For issues and questions:
- Check troubleshooting section
- Review framework documentation
- Monitor system logs
- Use built-in diagnostic tools

## License

This integration maintains compatibility with all included framework licenses:
- PyTorch: BSD License
- TensorFlow: Apache 2.0
- Transformers: Apache 2.0
- spaCy: MIT License
- NLTK: Apache 2.0