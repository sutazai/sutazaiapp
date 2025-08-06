# SutazAI Multi-Modal Fusion System

A comprehensive multi-modal fusion coordination system that enables the SutazAI platform to seamlessly process and understand multiple data modalities simultaneously, including text, voice, visual data, and sensor inputs.

## 🎯 Overview

The Multi-Modal Fusion System provides advanced capabilities for integrating heterogeneous data streams through sophisticated fusion architectures. It supports real-time processing with high concurrency, designed to work with SutazAI's 69 AI agents and existing infrastructure.

### Key Features

- **Multiple Fusion Strategies**: Early, Late, and Hybrid fusion approaches
- **Temporal Synchronization**: Alignment of data across different sampling rates
- **Cross-Modal Learning**: Advanced attention mechanisms and transfer learning
- **Real-Time Processing**: High-performance pipeline supporting 1000+ req/s
- **Auto-Scaling**: Dynamic worker management based on system load
- **Comprehensive Monitoring**: Real-time visualization and debugging tools
- **SutazAI Integration**: Seamless integration with existing agent ecosystem

## 🏗️ Architecture

### Core Components

1. **Multi-Modal Fusion Coordinator** (`core/multi_modal_fusion_coordinator.py`)
   - Main fusion processing engine
   - Supports early, late, and hybrid fusion strategies
   - Temporal synchronization across modalities
   - Integration with Ollama/gpt-oss for text processing

2. **Unified Representation Framework** (`core/unified_representation.py`)
   - Cross-modal embedding alignment
   - Semantic space unification
   - Hierarchical representation learning
   - Integration with knowledge graphs

3. **Cross-Modal Learning System** (`core/cross_modal_learning.py`)
   - Contrastive learning across modalities
   - Dynamic attention mechanisms
   - Transfer learning between modalities
   - Adaptive representation optimization

4. **Real-Time Processing Pipeline** (`pipeline/realtime_fusion_pipeline.py`)
   - High-performance streaming processing
   - Auto-scaling and load balancing
   - WebSocket-based real-time updates
   - Support for 174 concurrent consumers (aligned with Ollama config)

5. **Visualization Tools** (`visualization/fusion_visualizer.py`)
   - Real-time monitoring dashboard
   - Cross-modal attention visualization
   - Performance metrics and debugging
   - Streamlit-based web interface

## 🚀 Quick Start

### Installation

1. **Install Dependencies**
   ```bash
   cd /opt/sutazaiapp/fusion
   pip install -r requirements.txt
   ```

2. **Configure System**
   ```bash
   # Copy and customize configuration
   cp /opt/sutazaiapp/config/fusion_config.yaml /opt/sutazaiapp/config/fusion_config.local.yaml
   ```

3. **Initialize Database**
   ```python
   from fusion.visualization.fusion_visualizer import DataCollector, VisualizationConfig
   
   config = VisualizationConfig()
   collector = DataCollector(config)
   # Database tables will be created automatically
   ```

### Basic Usage

```python
import asyncio
from fusion import (
    MultiModalFusionCoordinator,
    ModalityType,
    ModalityData,
    FusionStrategy
)

async def main():
    # Initialize fusion coordinator
    coordinator = MultiModalFusionCoordinator()
    
    # Create multi-modal data
    text_data = ModalityData(
        modality_type=ModalityType.TEXT,
        data="Hello, this is a test message.",
        confidence=0.95
    )
    
    voice_data = ModalityData(
        modality_type=ModalityType.VOICE,
        data="[audio_data]",
        confidence=0.85
    )
    
    # Process fusion
    result = await coordinator.process_multi_modal_input(
        modality_data={
            ModalityType.TEXT: text_data,
            ModalityType.VOICE: voice_data
        },
        fusion_strategy=FusionStrategy.HYBRID
    )
    
    print(f"Fusion completed with confidence: {result.confidence_scores}")

# Run the example
asyncio.run(main())
```

### Real-Time Pipeline

```python
import asyncio
from fusion.pipeline.realtime_fusion_pipeline import (
    RealTimeFusionPipeline,
    ProcessingRequest,
    ProcessingPriority
)

async def main():
    # Initialize pipeline
    pipeline = RealTimeFusionPipeline()
    
    # Start processing
    await pipeline.start_pipeline()
    
    # Create processing request
    request = ProcessingRequest(
        request_id="test_001",
        modality_data={
            ModalityType.TEXT: text_data,
            ModalityType.VOICE: voice_data
        },
        fusion_strategy=FusionStrategy.EARLY,
        priority=ProcessingPriority.HIGH
    )
    
    # Process request
    response = await pipeline.process_request(request)
    print(f"Processing latency: {response.processing_latency:.3f}s")
    
    # Stop pipeline
    await pipeline.stop_pipeline()

asyncio.run(main())
```

### Visualization Dashboard

```bash
# Start the visualization dashboard
cd /opt/sutazaiapp/fusion
streamlit run visualization/fusion_visualizer.py --server.port 8501
```

Then open http://localhost:8501 to access the dashboard.

## 📊 Performance Specifications

### Throughput and Latency
- **Target Throughput**: 1000+ requests/second
- **Latency**: < 100ms for simple fusion operations
- **Concurrent Users**: Supports 174 concurrent consumers (Ollama alignment)
- **Auto-Scaling**: 2-32 workers based on system load

### Resource Requirements
- **Memory**: 4-8GB RAM for optimal performance
- **CPU**: 8-12 cores recommended
- **Storage**: 2GB for caching and temporary data
- **Network**: WebSocket support for real-time monitoring

### Supported Modalities
- **Text**: Natural language processing via Ollama/gpt-oss
- **Voice**: Audio processing with spectral analysis
- **Visual**: Image/video processing with CNN features
- **Sensor**: Time-series data with statistical features
- **Structured**: Database and API data integration

## 🔧 Configuration

### Main Configuration (`/opt/sutazaiapp/config/fusion_config.yaml`)

```yaml
# Temporal synchronization
temporal_window: 2.0
sync_tolerance: 0.2

# Feature representation
feature_dim: 768
embedding_model: "nomic-embed-text"

# Processing configuration
max_workers: 12
batch_size: 32
queue_max_size: 1000

# Integration endpoints
ollama_url: "http://ollama:11434"
jarvis_url: "http://jarvis:8080" 
chromadb_url: "http://chromadb:8000"
neo4j_url: "bolt://neo4j:7687"

# Performance optimization
enable_gpu: false
mixed_precision: false
compile_models: true
```

### Modality-Specific Settings

```yaml
modalities:
  text:
    max_length: 8192
    chunk_size: 512
    confidence_threshold: 0.7
  
  voice:
    sample_rate: 16000
    max_duration: 30
    feature_extraction: "mfcc"
    confidence_threshold: 0.6
  
  visual:
    max_resolution: [1024, 1024]
    feature_extraction: "cnn"
    confidence_threshold: 0.8
```

## 🔗 SutazAI Integration

### Agent Orchestration
The fusion system integrates with SutazAI's 69 AI agents through the agent orchestration system:

```python
from fusion import integrate_with_sutazai

# Configure integration
integration_config = integrate_with_sutazai(
    agent_orchestrator_url="http://backend:8000/api/v1/agents",
    ollama_url="http://ollama:11434"
)

# Initialize with integration
coordinator = MultiModalFusionCoordinator(integration_config)
```

### Knowledge Graph Integration
Connects to Neo4j for semantic understanding:

```python
# Semantic mapping through knowledge graph
semantic_mapper = SemanticSpaceMapper(
    knowledge_graph_path="bolt://neo4j:7687"
)
```

### Vector Database Integration
Utilizes ChromaDB and Qdrant for efficient embedding storage:

```python
# Configure vector storage
vector_config = {
    "chromadb_url": "http://chromadb:8000",
    "qdrant_url": "http://qdrant:6333",
    "collection_name": "sutazai_multimodal"
}
```

## 📈 Monitoring and Debugging

### Real-Time Metrics
- Processing throughput and latency
- Memory and CPU usage
- Queue sizes and worker status
- Error rates and types

### Visualization Features
- Interactive fusion result analysis
- Cross-modal attention heatmaps
- Representation space visualization (t-SNE, PCA, UMAP)
- Performance trend analysis

### Health Checks
```python
# Pipeline health check
health_status = await pipeline.health_check()
print(f"System status: {health_status['status']}")
```

### WebSocket Monitoring
```javascript
// Connect to real-time metrics
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = function(event) {
    const metrics = JSON.parse(event.data);
    console.log('Throughput:', metrics.throughput_per_second);
};
```

## 🧪 Testing and Validation

### Unit Tests
```bash
cd /opt/sutazaiapp/fusion
python -m pytest tests/ -v --cov=fusion
```

### Integration Tests
```bash
# Test with SutazAI infrastructure
python -m pytest tests/integration/ -v
```

### Performance Tests
```bash
# Load testing
python tests/performance/load_test.py --requests 1000 --concurrency 50
```

### Example Test
```python
import pytest
from fusion import MultiModalFusionCoordinator, ModalityType, ModalityData

@pytest.mark.asyncio
async def test_basic_fusion():
    coordinator = MultiModalFusionCoordinator()
    
    text_data = ModalityData(
        modality_type=ModalityType.TEXT,
        data="Test message",
        confidence=0.9
    )
    
    result = await coordinator.process_multi_modal_input({
        ModalityType.TEXT: text_data
    })
    
    assert result.fusion_result is not None
    assert result.fusion_result.confidence_scores[ModalityType.TEXT] == 0.9
```

## 📚 API Reference

### Core Classes

#### MultiModalFusionCoordinator
Main fusion processing engine supporting multiple strategies.

**Methods:**
- `process_multi_modal_input()`: Process multi-modal data
- `get_processing_statistics()`: Get performance metrics
- `shutdown()`: Graceful shutdown

#### UnifiedRepresentationFramework
Cross-modal representation learning and management.

**Methods:**
- `create_unified_representation()`: Create unified embeddings
- `find_similar_representations()`: Similarity search
- `get_representation_analytics()`: Analysis tools

#### RealTimeFusionPipeline
High-performance streaming processing pipeline.

**Methods:**
- `start_pipeline()`: Start processing
- `process_request()`: Handle single request
- `process_stream()`: Handle streaming data
- `get_pipeline_status()`: Status information

### Data Types

#### ModalityData
Container for modality-specific data with metadata.

#### FusionResult
Result of fusion operation with confidence scores and metadata.

#### UnifiedRepresentation
Unified cross-modal representation with semantic features.

## 🤝 Contributing

### Development Setup
```bash
# Clone and setup development environment
git clone <repository>
cd fusion-system
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Maintain test coverage > 90%
- Document all public APIs

### Pull Request Process
1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Submit PR with detailed description

## 📝 License

This multi-modal fusion system is part of the SutazAI platform. All rights reserved.

## 🆘 Support

### Documentation
- API Reference: `/docs/api/`
- Examples: `/examples/`
- Tutorials: `/docs/tutorials/`

### Issues and Questions
- GitHub Issues: Create detailed bug reports
- Discussions: General questions and feature requests
- Email: Technical support contact

### Performance Tuning
For optimal performance with your specific workload:
1. Profile your fusion operations
2. Adjust worker counts and batch sizes
3. Optimize modality-specific preprocessing
4. Monitor memory usage and garbage collection

---

**SutazAI Multi-Modal Fusion System v1.0.0**  
Advanced multi-modal processing for the next generation of AI systems.