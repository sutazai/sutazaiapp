# SutazaiApp Technical Architecture

This document outlines the technical architecture of SutazaiApp, explaining key components, data flows, and design decisions.

## System Overview

SutazaiApp is built as a modular, service-oriented architecture with the following primary components:

1. **FastAPI Backend**: Core API server that exposes all functionality
2. **Document Processing Services**: Specialized services for handling different document types
3. **Vector Store**: Manages document embeddings and semantic search capabilities
4. **Code Generation Service**: AI-powered code generation and improvement
5. **AI Orchestrator**: Monitors and optimizes system performance

## Component Architecture

### FastAPI Backend

The FastAPI backend serves as the central integration point for all services. It provides:

- RESTful API endpoints for all functionality
- Request validation and error handling
- Authentication and authorization
- Background task processing
- Automatic API documentation via Swagger UI
- Cross-origin resource sharing (CORS) support

### Document Processing Services

Document processing is handled by specialized services:

- **PDF Processor**: Extracts text and metadata from PDF files, with optional OCR support
- **DOCX Processor**: Extracts text and metadata from Word documents
- **Document Router**: Routes document processing requests to the appropriate processor

The document processing pipeline includes:
1. File upload and validation
2. Text extraction and metadata parsing
3. OCR processing for images (when enabled)
4. Text chunking for semantic indexing
5. Vector embedding generation
6. Storage in both vector database and document store

### Vector Store

The vector store uses Qdrant for efficient similarity search:

- Document chunks are converted to embeddings using Sentence Transformers
- Embeddings are stored with metadata and source information
- Similarity search is performed using cosine similarity
- Search results include original text, metadata, and relevance scores

### Code Generation Service

The code generation service leverages large language models for code creation:

- **Code Generator**: Creates code based on natural language specifications
- **Code Analyzer**: Identifies issues and improvement opportunities in generated code
- **Code Improver**: Enhances code based on identified issues

The code generation process follows these steps:
1. Specification parsing and analysis
2. Initial code generation
3. Static analysis for quality issues
4. Iterative improvement based on identified issues
5. Final code formatting and documentation

### AI Orchestrator

The AI Orchestrator monitors and optimizes system performance:

- **Performance Metrics**: Tracks response times, resource usage, and error rates
- **Self-Improvement**: Analyzes performance data to identify optimization opportunities
- **Resource Management**: Adapts resource allocation based on workload patterns

## Data Flow

1. **Document Processing**:
   ```
   Client -> API -> Document Router -> Document Processor -> Vector Store -> Database
                                                          -> Document Storage
   ```

2. **Vector Search**:
   ```
   Client -> API -> Vector Store -> Client
   ```

3. **Code Generation**:
   ```
   Client -> API -> Code Generator -> Code Analyzer -> Code Improver -> Client
                                                    -> Database
   ```

## Database Schema

SutazaiApp uses PostgreSQL with pgvector extension for primary data storage:

### User

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login TIMESTAMP
);
```

### APIKey

```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_used TIMESTAMP,
    enabled BOOLEAN NOT NULL DEFAULT TRUE
);
```

### Document

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    filename VARCHAR(255) NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    author VARCHAR(255),
    page_count INTEGER,
    processed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    full_text TEXT,
    vectorized BOOLEAN NOT NULL DEFAULT FALSE
);
```

### CodeGeneration

```sql
CREATE TABLE code_generations (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    spec_text TEXT NOT NULL,
    language VARCHAR(50) NOT NULL,
    generated_code TEXT NOT NULL,
    issues JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    generation_time_ms INTEGER
);
```

## Technology Stack

- **Backend**: Python 3.8+, FastAPI, Pydantic
- **Database**: PostgreSQL with pgvector extension
- **Vector Database**: Qdrant
- **ML Models**: Sentence Transformers, Hugging Face models
- **Document Processing**: PyMuPDF, python-docx, Tesseract OCR
- **Code Analysis**: Pylint, Bandit
- **Authentication**: JWT, OAuth2
- **Infrastructure**: Docker, Docker Compose

## Deployment Architecture

SutazaiApp can be deployed in various configurations:

### Single-Server Deployment

Simplest deployment, suitable for small installations:

```
┌─────────────────────────────────┐
│           Application Server    │
│                                 │
│  ┌─────────┐    ┌────────────┐  │
│  │ FastAPI │    │PostgreSQL  │  │
│  │ Backend │    │Database    │  │
│  └─────────┘    └────────────┘  │
│                                 │
│  ┌─────────┐    ┌────────────┐  │
│  │ Qdrant  │    │ML Models   │  │
│  │ Vector  │    │            │  │
│  │ Database│    │            │  │
│  └─────────┘    └────────────┘  │
└─────────────────────────────────┘
```

### Distributed Deployment

For larger deployments with higher availability requirements:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  API Server │   │  API Server │   │  API Server │
│  (FastAPI)  │   │  (FastAPI)  │   │  (FastAPI)  │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └─────────┬───────┴────────┬────────┘
                 │                │
       ┌─────────▼──────┐  ┌──────▼─────────┐
       │  PostgreSQL    │  │  Qdrant Vector │
       │  Database      │  │  Database      │
       └────────────────┘  └────────────────┘
                │                │
       ┌────────▼─────────────────▼────────┐
       │          Model Server             │
       │  (AI Models and Processing)       │
       └─────────────────────────────────┘
```

## Security Considerations

- API authentication via JWT tokens
- HTTPS for all external communications
- Database connection encryption
- API rate limiting to prevent abuse
- Input validation on all endpoints
- Document isolation between users
- Secure model serving
- Regular security updates

## Scalability

SutazaiApp is designed to scale in several dimensions:

- **Horizontal Scaling**: Multiple API servers behind a load balancer
- **Vertical Scaling**: Increasing resources for computation-intensive components
- **Database Scaling**: PostgreSQL replication and sharding for high throughput
- **Vector Database Scaling**: Qdrant's distributed deployment capabilities
- **Computation Offloading**: Model serving on dedicated GPU servers

## Monitoring and Maintenance

The system includes comprehensive monitoring:

- Prometheus metrics collection
- Performance tracking via AI Orchestrator
- Log aggregation and analysis
- Automated alerts for system issues
- Health check endpoints for each service
- Backup and recovery procedures 