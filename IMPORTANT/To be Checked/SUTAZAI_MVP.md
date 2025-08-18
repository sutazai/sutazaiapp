Contents
Executive SummarySystem ArchitectureCore PrinciplesComponent InteractionContainerized DeploymentDocker ComposeService DefinitionsCore ComponentsChatbotCode AssistantResearch ToolOllama IntegrationData StorageImplementation
Local AI Assistant:
A Containerized MVP Architecture
Building a privacy-centric, multi-functional AI system that runs entirely on local hardware using Docker, Ollama, and modern microservices architecture.
Multi-Purpose AI
Chatbot, code assistant, and research tool in one unified interface
Privacy First
All data remains on your machine with local LLM inference
Containerized
Docker-based deployment for portability and reproducibility
Lightweight
Optimized for limited hardware with TinyLlama and GPT-OSS:20b
Executive Summary
Key Achievements
•	Designed a containerized, multi-functional AI assistant that runs entirely on local hardware
•	Implemented three core use cases: conversational chatbot, code assistant, and research tool with RAG
•	Leveraged Ollama framework for local LLM management with support for TinyLlama and GPT-OSS:20b
This document outlines a functional Minimum Viable Product (MVP) for a multi-purpose local AI assistant. The system is designed as a containerized application using Docker and Docker Compose, enabling it to run entirely on local hardware. It supports three primary use cases: a conversational chatbot, a code assistant, and a research tool powered by a Retrieval-Augmented Generation (RAG) pipeline.
The architecture is modular, leveraging a microservices approach to ensure scalability and maintainability. The core AI capabilities are provided by local Large Language Models (LLMs) managed through the Ollama framework, with support for lightweight models like TinyLlama and more powerful options like GPT-OSS:20b, depending on hardware capabilities.
The system is designed with a local-first, privacy-centric philosophy, ensuring all user data remains on the user's machine. This approach addresses growing concerns about data privacy while providing powerful AI capabilities for personal and professional use.
System Architecture Overview
The proposed system is a containerized, multi-functional AI assistant designed to operate entirely on local hardware, addressing privacy concerns and leveraging the capabilities of open-source Large Language Models (LLMs). The architecture is built on a microservices model, ensuring modularity and scalability, with Docker Compose orchestrating the various components.
System Architecture Flow
Core Principles for a Lightweight MVP
Modularity
Microservices architecture allows independent development, deployment, and scaling of each component. This approach is well-documented in successful AI application deployments [146].
Containerization
Docker and Docker Compose ensure portability, reproducibility, and simplified management across different environments. Each microservice is encapsulated with all necessary dependencies [147].
Local-First
All user data remains on the user's machine, ensuring complete privacy and security. Local LLMs provide powerful AI capabilities without external cloud services [162].
High-Level Component Interaction
User Interface as Central Interaction Point
A single, integrated web-based UI built with React or Vue.js serves as the primary interaction point. The UI communicates with backend services through a RESTful API, providing a consistent experience across all use cases.
API Gateway for Routing Requests
An API Gateway acts as a single entry point, routing requests to specialized backend services. This simplifies client-side code and provides a central point for authentication, logging, and rate limiting.
Specialized Backend Services
•	Chatbot Service: Handles general-purpose conversational queries
•	Code Assistant Service: Specialized for code generation and explanation
•	Research Tool Service: Implements RAG pipeline for document interaction
Ollama as Central LLM Engine
Ollama serves as the engine for running local LLMs, providing a unified interface for model management and inference. Backend services communicate with Ollama via REST API for text generation [207].
Containerized Deployment Strategy
The deployment strategy leverages Docker and Docker Compose to ensure the system is portable, reproducible, and easy to manage across different environments. Each component is encapsulated within its own Docker container, including all necessary dependencies and configuration files.
Docker Compose Orchestration
# docker-compose.yml
version: '3.8'

services:
  web-ui:
    build: ./web-ui
    ports:
      - "3000:3000"
    depends_on:
      - api-gateway
    volumes:
      - ./web-ui:/app
      - /app/node_modules

  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - chatbot-service
      - code-assistant-service
      - research-tool-service

  chatbot-service:
    build: ./chatbot-service
    environment:
      - OLLAMA_HOST=ollama:11434
    depends_on:
      - ollama
      - database

  code-assistant-service:
    build: ./code-assistant-service
    environment:
      - OLLAMA_HOST=ollama:11434
    depends_on:
      - ollama
      - database

  research-tool-service:
    build: ./research-tool-service
    environment:
      - OLLAMA_HOST=ollama:11434
      - VECTOR_DB_HOST=vector-db
    depends_on:
      - ollama
      - vector-db
      - database

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama

  vector-db:
    image: chromadb/chroma
    ports:
      - "8001:8000"
    volumes:
      - vector_data:/data

  database:
    image: postgres:13
    environment:
      - POSTGRES_USER=ai_assistant
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=ai_assistant
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  ollama_models:
  vector_data:
  postgres_data:
                            
Service Definitions
Each component is defined as a separate service with specific configurations, dependencies, and volume mounts.
Inter-Container Communication
Docker Compose creates a private network where services communicate using service names as hostnames.
Data Persistence
Named volumes ensure data persistence across container restarts for models, vector data, and user data.
Service Definitions
Web UI Service (React/Next.js)
Modern JavaScript framework with Nginx server, exposing port 80. Mounts source code for hot-reloading during development.
API Gateway Service (FastAPI/Node.js)
High-performance web framework listening on port 8000, implementing routing logic and cross-cutting concerns.
Backend Services
Three specialized services built with FastAPI:
•	Chatbot Service: Manages conversational queries and history
•	Code Assistant Service: Handles code generation and explanation
•	Research Tool Service: Implements RAG pipeline with vector database integration
Ollama Service
Official Ollama Docker image exposing port 11434, with volume mount for persistent model storage.
Vector Database Service (Faiss/ChromaDB)
Lightweight vector database for storing document embeddings, with persistent volume for vector data.
Core Components
The Chatbot Component
Architecture & Implementation
•	User Interface: React/Vue.js with chat-like experience and responsive design
•	Backend API: FastAPI/Node.js for processing chat requests and managing conversation history
•	Ollama Integration: Official Python library for streaming responses with context
Local LLM Integration
General-Purpose Models: TinyLlama for lightweight conversation
Model Management: Ollama framework for easy switching
Context Handling: Conversation history maintained for coherent responses
Key Features
Streaming responses for real-time interaction, conversation history management, and flexible model selection based on hardware capabilities.
The Code Assistant Component
Specialized Architecture
•	Code Editor UI: Dedicated interface with syntax highlighting
•	Code Processing API: Specialized backend for code-related queries
•	Code-Specific LLMs: CodeLlama or Mistral-Instruct for programming tasks
Capabilities
Code Generation
Generate code snippets from natural language descriptions
Code Explanation
Explain existing code functionality and structure
Boilerplate Automation
Automate generation of common boilerplate code
The Research Tool Component
RAG Pipeline Architecture
Document Processing
•	• File upload and parsing
•	• Text chunking for context
•	• Metadata extraction
Vector Embeddings
•	• Sentence Transformers
•	• Semantic representation
•	• Dimensionality optimization
Retrieval & Generation
•	• Similarity search (Faiss)
•	• Context augmentation
•	• Prompt engineering
LangChain Integration
The RAG pipeline uses LangChain for orchestration, providing tools and abstractions for building sophisticated document interaction systems with local LLMs.
Local LLM Integration with Ollama
Ollama serves as the core engine for local LLM deployment and management, functioning analogously to Docker for AI models. It simplifies the process of running open-source LLMs on local hardware, abstracting away complexities of model configuration and resource optimization [207].
Model Management Engine
Docker Container Setup
Ollama runs in a dedicated container with volume mounting for persistent model storage at /root/.ollama
Model Selection
Support for TinyLlama (lightweight) and GPT-OSS:20b (powerful) with ollama pull commands
REST API Access
Exposes API at http://ollama:11434 for inter-service communication
Python Integration
# Install Ollama Python library
pip install ollama

# Basic usage example
import ollama

# Generate text response
response = ollama.generate(
    model='tinyllama',
    prompt='Hello, how are you?'
)

# Chat with context
messages = [
    {'role': 'user', 'content': 'Hello!'},
    {'role': 'assistant', 'content': 'Hi there!'},
    {'role': 'user', 'content': 'How are you?'}
]

response = ollama.chat(
    model='tinyllama',
    messages=messages,
    stream=True  # For real-time responses
)
Streaming responses for real-time interaction
Function calling for agentic behavior [208]
Multi-modal support for advanced models
Advanced Feature: Function Calling
Function calling enables sophisticated AI agents by allowing LLMs to interact with external tools and APIs:
•	• Code execution in sandboxed environments
•	• Vector database queries for research tool
•	• External API integrations for extended capabilities
Data Storage and Management
User Data Storage
Relational Database
SQLite or PostgreSQL for structured data storage:
•	• Chat histories and conversation context
•	• User sessions and preferences
•	• Application configuration
Persistent Volumes
Docker volumes ensure data persistence across container restarts and updates.
Research Knowledge Base
Vector Database
Faiss or ChromaDB for efficient vector operations:
•	• Document embeddings storage
•	• Similarity search indices
•	• Metadata management
Document Storage
File system storage for original documents with metadata indexing.
Data Flow Architecture
Implementation Roadmap
Phase 1: Foundation
•	• Docker Compose setup
•	• Ollama service integration
•	• Basic UI framework
•	• API Gateway implementation
•	• Chatbot service development
Phase 2: Enhancement
•	• Code Assistant development
•	• Vector database setup
•	• RAG pipeline implementation
•	• Research tool integration
•	• User data persistence
Phase 3: Optimization
•	• Performance tuning
•	• Advanced function calling
•	• Model switching logic
•	• UI/UX refinement
•	• Documentation & testing
Success Metrics
Technical Metrics
•	• Response latency < 500ms for chat interactions
•	• Document processing time < 1s per page
•	• Memory usage optimized for 8GB RAM systems
•	• Startup time < 30s for all services
User Experience Metrics
•	• Unified interface for all three use cases
•	• Seamless switching between functions
•	• Intuitive document management
•	• Context-aware conversation flow
Technical Considerations
Hardware Requirements
Minimum: 8GB RAM, 20GB disk space, modern CPU
Recommended: 16GB+ RAM, GPU acceleration, SSD storage
Model Selection Strategy
Lightweight: TinyLlama for general tasks (2GB RAM)
Powerful: GPT-OSS:20b for complex tasks (16GB+ RAM)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


<!DOCTYPE html>
<html lang="en">

  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local AI Assistant: A Containerized MVP Architecture</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tiempos+Headline:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <style>
        :root {
            --primary: #0f766e;
            --primary-light: #14b8a6;
            --secondary: #ea580c;
            --accent: #f59e0b;
            --neutral: #374151;
            --base-100: #fefefe;
            --base-200: #f8fafc;
            --base-300: #e2e8f0;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            color: var(--neutral);
            overflow-x: hidden;
        }
        
        .font-serif {
            font-family: 'Tiempos Headline', serif;
        }
        
        .toc-fixed {
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            width: 280px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-right: 1px solid var(--base-300);
            z-index: 1000;
            overflow-y: auto;
            padding: 2rem 1.5rem;
        }
        
        .main-content {
            margin-left: 280px;
            min-height: 100vh;
        }
        
        .hero-section {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
            position: relative;
            overflow: hidden;
        }
        
        .hero-overlay {
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.3);
            z-index: 1;
        }
        
        .hero-content {
            position: relative;
            z-index: 2;
        }
        
        .bento-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin-top: 3rem;
        }
        
        .bento-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 1rem;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        .bento-card:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }
        
        .section-card {
            background: white;
            border-radius: 1rem;
            padding: 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--base-300);
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left: 4px solid var(--accent);
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1.5rem 0;
        }
        
        .citation-link {
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
            border-bottom: 1px dotted var(--primary);
            transition: all 0.2s ease;
        }
        
        .citation-link:hover {
            color: var(--secondary);
            border-bottom-color: var(--secondary);
        }
        
        .toc-link {
            display: block;
            padding: 0.5rem 0;
            color: var(--neutral);
            text-decoration: none;
            border-left: 2px solid transparent;
            padding-left: 1rem;
            transition: all 0.2s ease;
        }
        
        .toc-link:hover, .toc-link.active {
            color: var(--primary);
            border-left-color: var(--primary);
            background: rgba(15, 118, 110, 0.05);
        }
        
        .toc-sub {
            margin-left: 1rem;
            font-size: 0.875rem;
        }
        
        .architecture-diagram {
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
            border: 1px solid var(--base-300);
        }
        
        /* Mermaid Chart Styles */
        .mermaid-container {
            display: flex;
            justify-content: center;
            min-height: 300px;
            max-height: 800px;
            background: #ffffff;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 30px;
            margin: 30px 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            position: relative;
            overflow: hidden;
        }

        .mermaid-container .mermaid {
            width: 100%;
            max-width: 100%;
            height: 100%;
            cursor: grab;
            transition: transform 0.3s ease;
            transform-origin: center center;
            display: flex;
            justify-content: center;
            align-items: center;
            touch-action: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
        }

        .mermaid-container .mermaid svg {
            max-width: 100%;
            height: 100%;
            display: block;
            margin: 0 auto;
        }

        .mermaid-container .mermaid:active {
            cursor: grabbing;
        }

        .mermaid-container.zoomed .mermaid {
            height: 100%;
            width: 100%;
            cursor: grab;
        }

        .mermaid-controls {
            position: absolute;
            top: 15px;
            right: 15px;
            display: flex;
            gap: 10px;
            z-index: 20;
            background: rgba(255, 255, 255, 0.95);
            padding: 8px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .mermaid-control-btn {
            background: #ffffff;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            padding: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #374151;
            font-size: 14px;
            min-width: 36px;
            height: 36px;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .mermaid-control-btn:hover {
            background: #f8fafc;
            border-color: #3b82f6;
            color: #3b82f6;
            transform: translateY(-1px);
        }

        .mermaid-control-btn:active {
            transform: scale(0.95);
        }

        @media (max-width: 1024px) {
            .toc-fixed {
                transform: translateX(-100%);
                transition: transform 0.3s ease;
            }
            
            .toc-fixed.open {
                transform: translateX(0);
            }
            
            .main-content {
                margin-left: 0;
            }
            
            .bento-grid {
                grid-template-columns: 1fr;
            }

            .mermaid-control-btn:not(.reset-zoom) {
                display: none;
            }
            .mermaid-controls {
                top: auto;
                bottom: 15px;
                right: 15px;
            }
        }

        /* Prevent horizontal overflow on small screens */
        @media (max-width: 768px) {
            .hero-section h1 {
                font-size: 2.5rem;
                line-height: 1.2;
            }
            .hero-section p {
                font-size: 1rem;
            }
            .hero-section .container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .bento-card {
                padding: 1rem;
            }
            .section-card {
                padding: 1.5rem;
            }
        }

        @media (max-width: 480px) {
            .hero-section h1 {
                font-size: 2rem;
            }
        }
    </style>
  </head>

  <body>
    <!-- Table of Contents -->
    <nav class="toc-fixed">
      <div class="mb-6">
        <h3 class="font-serif font-bold text-lg text-gray-800 mb-4">Contents</h3>
        <div class="space-y-1">
          <a href="#executive-summary" class="toc-link">Executive Summary</a>
          <a href="#system-architecture" class="toc-link">System Architecture</a>
          <a href="#core-principles" class="toc-link toc-sub">Core Principles</a>
          <a href="#component-interaction" class="toc-link toc-sub">Component Interaction</a>
          <a href="#containerized-deployment" class="toc-link">Containerized Deployment</a>
          <a href="#docker-compose" class="toc-link toc-sub">Docker Compose</a>
          <a href="#service-definitions" class="toc-link toc-sub">Service Definitions</a>
          <a href="#core-components" class="toc-link">Core Components</a>
          <a href="#chatbot" class="toc-link toc-sub">Chatbot</a>
          <a href="#code-assistant" class="toc-link toc-sub">Code Assistant</a>
          <a href="#research-tool" class="toc-link toc-sub">Research Tool</a>
          <a href="#ollama-integration" class="toc-link">Ollama Integration</a>
          <a href="#data-storage" class="toc-link">Data Storage</a>
          <a href="#implementation" class="toc-link">Implementation</a>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="main-content">
      <!-- Hero Section -->
      <section class="hero-section text-white py-20">
        <div class="hero-overlay"></div>
        <div class="hero-content container mx-auto px-6">
          <div class="max-w-4xl">
            <h1 class="font-serif text-5xl font-bold mb-6 leading-tight">
              <em>Local AI Assistant:</em>
              <br>
              A Containerized MVP Architecture
            </h1>
            <p class="text-xl text-gray-200 mb-8 leading-relaxed">
              Building a privacy-centric, multi-functional AI system that runs entirely on local hardware using Docker, Ollama, and modern microservices architecture.
            </p>
          </div>

          <!-- Bento Grid -->
          <div class="bento-grid">
            <div class="bento-card">
              <div class="flex items-center mb-4">
                <i class="fas fa-robot text-2xl mr-3"></i>
                <h3 class="font-serif text-xl font-semibold">Multi-Purpose AI</h3>
              </div>
              <p class="text-gray-200">Chatbot, code assistant, and research tool in one unified interface</p>
            </div>

            <div class="bento-card">
              <div class="flex items-center mb-4">
                <i class="fas fa-shield-alt text-2xl mr-3"></i>
                <h3 class="font-serif text-xl font-semibold">Privacy First</h3>
              </div>
              <p class="text-gray-200">All data remains on your machine with local LLM inference</p>
            </div>

            <div class="bento-card">
              <div class="flex items-center mb-4">
                <i class="fas fa-cube text-2xl mr-3"></i>
                <h3 class="font-serif text-xl font-semibold">Containerized</h3>
              </div>
              <p class="text-gray-200">Docker-based deployment for portability and reproducibility</p>
            </div>

            <div class="bento-card">
              <div class="flex items-center mb-4">
                <i class="fas fa-microchip text-2xl mr-3"></i>
                <h3 class="font-serif text-xl font-semibold">Lightweight</h3>
              </div>
              <p class="text-gray-200">Optimized for limited hardware with TinyLlama and GPT-OSS:20b</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Executive Summary -->
      <section id="executive-summary" class="py-16 bg-white">
        <div class="container mx-auto px-6">
          <div class="section-card">
            <h2 class="font-serif text-3xl font-bold mb-6 text-gray-800">Executive Summary</h2>

            <div class="highlight-box">
              <h3 class="font-semibold text-lg mb-3">Key Achievements</h3>
              <ul class="space-y-2">
                <li class="flex items-start">
                  <i class="fas fa-check-circle text-green-600 mt-1 mr-2"></i>
                  Designed a containerized, multi-functional AI assistant that runs entirely on local hardware
                </li>
                <li class="flex items-start">
                  <i class="fas fa-check-circle text-green-600 mt-1 mr-2"></i>
                  Implemented three core use cases: conversational chatbot, code assistant, and research tool with RAG
                </li>
                <li class="flex items-start">
                  <i class="fas fa-check-circle text-green-600 mt-1 mr-2"></i>
                  Leveraged Ollama framework for local LLM management with support for TinyLlama and GPT-OSS:20b
                </li>
              </ul>
            </div>

            <p class="text-lg leading-relaxed mb-6">
              This document outlines a functional Minimum Viable Product (MVP) for a multi-purpose local AI assistant. The system is designed as a containerized application using Docker and Docker Compose, enabling it to run entirely on local hardware. It supports three primary use cases: a conversational chatbot, a code assistant, and a research tool powered by a Retrieval-Augmented Generation (RAG) pipeline.
            </p>

            <p class="text-lg leading-relaxed mb-6">
              The architecture is modular, leveraging a microservices approach to ensure scalability and maintainability. The core AI capabilities are provided by local Large Language Models (LLMs) managed through the Ollama framework, with support for lightweight models like TinyLlama and more powerful options like GPT-OSS:20b, depending on hardware capabilities.
            </p>

            <p class="text-lg leading-relaxed">
              The system is designed with a local-first, privacy-centric philosophy, ensuring all user data remains on the user's machine. This approach addresses growing concerns about data privacy while providing powerful AI capabilities for personal and professional use.
            </p>
          </div>
        </div>
      </section>

      <!-- System Architecture -->
      <section id="system-architecture" class="py-16 bg-gray-50">
        <div class="container mx-auto px-6">
          <div class="section-card">
            <h2 class="font-serif text-3xl font-bold mb-8 text-gray-800">System Architecture Overview</h2>

            <p class="text-lg leading-relaxed mb-8">
              The proposed system is a containerized, multi-functional AI assistant designed to operate entirely on local hardware, addressing privacy concerns and leveraging the capabilities of open-source Large Language Models (LLMs). The architecture is built on a microservices model, ensuring modularity and scalability, with Docker Compose orchestrating the various components.
            </p>

            <!-- Architecture Diagram -->
            <div class="architecture-diagram">
              <h3 class="font-serif text-xl font-semibold mb-6 text-center">System Architecture Flow</h3>
              <div class="mermaid-container">
                <div class="mermaid-controls">
                  <button class="mermaid-control-btn zoom-in" title="放大">
                    <i class="fas fa-search-plus"></i>
                  </button>
                  <button class="mermaid-control-btn zoom-out" title="缩小">
                    <i class="fas fa-search-minus"></i>
                  </button>
                  <button class="mermaid-control-btn reset-zoom" title="重置">
                    <i class="fas fa-expand-arrows-alt"></i>
                  </button>
                  <button class="mermaid-control-btn fullscreen" title="全屏查看">
                    <i class="fas fa-expand"></i>
                  </button>
                </div>
                <div class="mermaid">
                  graph TB
                  A["User Interface
                  <br />React/Next.js"] --> B["API Gateway
                  <br />FastAPI/Node.js"]
                  B --> C["Chatbot Service
                  <br />Python FastAPI"]
                  B --> D["Code Assistant Service
                  <br />Python FastAPI"]
                  B --> E["Research Tool Service
                  <br />Python FastAPI"]
                  C --> F["Ollama Service
                  <br />LLM Engine"]
                  D --> F
                  E --> F
                  E --> G["Vector Database
                  <br />Faiss/ChromaDB"]
                  H["SQLite/PostgreSQL
                  <br />User Data"] --> C
                  H --> D
                  I["Document Storage
                  <br />File System"] --> E

                  style A fill:#e0f2fe,stroke:#0277bd,stroke-width:2px,color:#000
                  style B fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
                  style C fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000
                  style D fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000
                  style E fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000
                  style F fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
                  style G fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
                  style H fill:#f1f8e9,stroke:#689f38,stroke-width:2px,color:#000
                  style I fill:#f1f8e9,stroke:#689f38,stroke-width:2px,color:#000
                </div>
              </div>
            </div>

            <div id="core-principles" class="mb-12">
              <h3 class="font-serif text-2xl font-semibold mb-6 text-gray-800">Core Principles for a Lightweight MVP</h3>

              <div class="grid md:grid-cols-3 gap-6">
                <div class="bg-white p-6 rounded-lg border border-gray-200">
                  <div class="flex items-center mb-4">
                    <i class="fas fa-puzzle-piece text-2xl text-teal-600 mr-3"></i>
                    <h4 class="font-semibold text-lg">Modularity</h4>
                  </div>
                  <p class="text-gray-600">
                    Microservices architecture allows independent development, deployment, and scaling of each component. This approach is well-documented in successful AI application deployments <a href="https://towardsdatascience.com/designing-building-deploying-an-ai-chat-app-from-scratch-part-1-f1ebf5232d4d/" class="citation-link" target="_blank">[146]</a>.
                  </p>
                </div>

                <div class="bg-white p-6 rounded-lg border border-gray-200">
                  <div class="flex items-center mb-4">
                    <i class="fas fa-docker text-2xl text-blue-600 mr-3"></i>
                    <h4 class="font-semibold text-lg">Containerization</h4>
                  </div>
                  <p class="text-gray-600">
                    Docker and Docker Compose ensure portability, reproducibility, and simplified management across different environments. Each microservice is encapsulated with all necessary dependencies <a href="https://www.theseus.fi/bitstream/handle/10024/861594/Zurcher_Alexandre.pdf?sequence=2" class="citation-link" target="_blank">[147]</a>.
                  </p>
                </div>

                <div class="bg-white p-6 rounded-lg border border-gray-200">
                  <div class="flex items-center mb-4">
                    <i class="fas fa-home text-2xl text-green-600 mr-3"></i>
                    <h4 class="font-semibold text-lg">Local-First</h4>
                  </div>
                  <p class="text-gray-600">
                    All user data remains on the user's machine, ensuring complete privacy and security. Local LLMs provide powerful AI capabilities without external cloud services <a href="https://cobusgreyling.medium.com/run-openai-gpt-oss-locally-with-ollama-50f7e40482f7" class="citation-link" target="_blank">[162]</a>.
                  </p>
                </div>
              </div>
            </div>

            <div id="component-interaction">
              <h3 class="font-serif text-2xl font-semibold mb-6 text-gray-800">High-Level Component Interaction</h3>

              <div class="space-y-6">
                <div class="bg-gray-50 p-6 rounded-lg">
                  <h4 class="font-semibold text-lg mb-3">User Interface as Central Interaction Point</h4>
                  <p class="text-gray-700">
                    A single, integrated web-based UI built with React or Vue.js serves as the primary interaction point. The UI communicates with backend services through a RESTful API, providing a consistent experience across all use cases.
                  </p>
                </div>

                <div class="bg-gray-50 p-6 rounded-lg">
                  <h4 class="font-semibold text-lg mb-3">API Gateway for Routing Requests</h4>
                  <p class="text-gray-700">
                    An API Gateway acts as a single entry point, routing requests to specialized backend services. This simplifies client-side code and provides a central point for authentication, logging, and rate limiting.
                  </p>
                </div>

                <div class="bg-gray-50 p-6 rounded-lg">
                  <h4 class="font-semibold text-lg mb-3">Specialized Backend Services</h4>
                  <ul class="space-y-2 text-gray-700">
                    <li><strong>Chatbot Service:</strong> Handles general-purpose conversational queries</li>
                    <li><strong>Code Assistant Service:</strong> Specialized for code generation and explanation</li>
                    <li><strong>Research Tool Service:</strong> Implements RAG pipeline for document interaction</li>
                  </ul>
                </div>

                <div class="bg-gray-50 p-6 rounded-lg">
                  <h4 class="font-semibold text-lg mb-3">Ollama as Central LLM Engine</h4>
                  <p class="text-gray-700">
                    Ollama serves as the engine for running local LLMs, providing a unified interface for model management and inference. Backend services communicate with Ollama via REST API for text generation <a href="https://www.cohorte.co/blog/using-ollama-with-python-step-by-step-guide" class="citation-link" target="_blank">[207]</a>.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Containerized Deployment -->
      <section id="containerized-deployment" class="py-16 bg-white">
        <div class="container mx-auto px-6">
          <div class="section-card">
            <h2 class="font-serif text-3xl font-bold mb-8 text-gray-800">Containerized Deployment Strategy</h2>

            <p class="text-lg leading-relaxed mb-8">
              The deployment strategy leverages Docker and Docker Compose to ensure the system is portable, reproducible, and easy to manage across different environments. Each component is encapsulated within its own Docker container, including all necessary dependencies and configuration files.
            </p>

            <div id="docker-compose" class="mb-12">
              <h3 class="font-serif text-2xl font-semibold mb-6 text-gray-800">Docker Compose Orchestration</h3>

              <div class="bg-gray-900 text-green-400 p-6 rounded-lg mb-6 overflow-x-auto">
                <pre class="text-sm">
# docker-compose.yml
version: '3.8'

services:
  web-ui:
    build: ./web-ui
    ports:
      - "3000:3000"
    depends_on:
      - api-gateway
    volumes:
      - ./web-ui:/app
      - /app/node_modules

  api-gateway:
    build: ./api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - chatbot-service
      - code-assistant-service
      - research-tool-service

  chatbot-service:
    build: ./chatbot-service
    environment:
      - OLLAMA_HOST=ollama:11434
    depends_on:
      - ollama
      - database

  code-assistant-service:
    build: ./code-assistant-service
    environment:
      - OLLAMA_HOST=ollama:11434
    depends_on:
      - ollama
      - database

  research-tool-service:
    build: ./research-tool-service
    environment:
      - OLLAMA_HOST=ollama:11434
      - VECTOR_DB_HOST=vector-db
    depends_on:
      - ollama
      - vector-db
      - database

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama

  vector-db:
    image: chromadb/chroma
    ports:
      - "8001:8000"
    volumes:
      - vector_data:/data

  database:
    image: postgres:13
    environment:
      - POSTGRES_USER=ai_assistant
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=ai_assistant
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  ollama_models:
  vector_data:
  postgres_data:
                            </pre>
              </div>

              <div class="grid md:grid-cols-3 gap-6">
                <div class="bg-blue-50 p-6 rounded-lg">
                  <h4 class="font-semibold text-lg mb-3 text-blue-800">Service Definitions</h4>
                  <p class="text-blue-700 text-sm">
                    Each component is defined as a separate service with specific configurations, dependencies, and volume mounts.
                  </p>
                </div>

                <div class="bg-green-50 p-6 rounded-lg">
                  <h4 class="font-semibold text-lg mb-3 text-green-800">Inter-Container Communication</h4>
                  <p class="text-green-700 text-sm">
                    Docker Compose creates a private network where services communicate using service names as hostnames.
                  </p>
                </div>

                <div class="bg-purple-50 p-6 rounded-lg">
                  <h4 class="font-semibold text-lg mb-3 text-purple-800">Data Persistence</h4>
                  <p class="text-purple-700 text-sm">
                    Named volumes ensure data persistence across container restarts for models, vector data, and user data.
                  </p>
                </div>
              </div>
            </div>

            <div id="service-definitions">
              <h3 class="font-serif text-2xl font-semibold mb-6 text-gray-800">Service Definitions</h3>

              <div class="space-y-6">
                <div class="border-l-4 border-blue-500 pl-6">
                  <h4 class="font-semibold text-lg mb-2">Web UI Service (React/Next.js)</h4>
                  <p class="text-gray-700">
                    Modern JavaScript framework with Nginx server, exposing port 80. Mounts source code for hot-reloading during development.
                  </p>
                </div>

                <div class="border-l-4 border-green-500 pl-6">
                  <h4 class="font-semibold text-lg mb-2">API Gateway Service (FastAPI/Node.js)</h4>
                  <p class="text-gray-700">
                    High-performance web framework listening on port 8000, implementing routing logic and cross-cutting concerns.
                  </p>
                </div>

                <div class="border-l-4 border-purple-500 pl-6">
                  <h4 class="font-semibold text-lg mb-2">Backend Services</h4>
                  <p class="text-gray-700 mb-3">Three specialized services built with FastAPI:</p>
                  <ul class="list-disc list-inside text-gray-600 space-y-1">
                    <li>Chatbot Service: Manages conversational queries and history</li>
                    <li>Code Assistant Service: Handles code generation and explanation</li>
                    <li>Research Tool Service: Implements RAG pipeline with vector database integration</li>
                  </ul>
                </div>

                <div class="border-l-4 border-orange-500 pl-6">
                  <h4 class="font-semibold text-lg mb-2">Ollama Service</h4>
                  <p class="text-gray-700">
                    Official Ollama Docker image exposing port 11434, with volume mount for persistent model storage.
                  </p>
                </div>

                <div class="border-l-4 border-pink-500 pl-6">
                  <h4 class="font-semibold text-lg mb-2">Vector Database Service (Faiss/ChromaDB)</h4>
                  <p class="text-gray-700">
                    Lightweight vector database for storing document embeddings, with persistent volume for vector data.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Core Components -->
      <section id="core-components" class="py-16 bg-gray-50">
        <div class="container mx-auto px-6">
          <div class="section-card">
            <h2 class="font-serif text-3xl font-bold mb-8 text-gray-800">Core Components</h2>

            <div id="chatbot" class="mb-12">
              <h3 class="font-serif text-2xl font-semibold mb-6 text-gray-800">The Chatbot Component</h3>

              <div class="grid md:grid-cols-2 gap-8 mb-8">
                <div>
                  <h4 class="font-semibold text-lg mb-4">Architecture & Implementation</h4>
                  <ul class="space-y-3 text-gray-700">
                    <li class="flex items-start">
                      <i class="fas fa-desktop text-blue-500 mt-1 mr-3"></i>
                      <span><strong>User Interface:</strong> React/Vue.js with chat-like experience and responsive design</span>
                    </li>
                    <li class="flex items-start">
                      <i class="fas fa-server text-green-500 mt-1 mr-3"></i>
                      <span><strong>Backend API:</strong> FastAPI/Node.js for processing chat requests and managing conversation history</span>
                    </li>
                    <li class="flex items-start">
                      <i class="fas fa-robot text-purple-500 mt-1 mr-3"></i>
                      <span><strong>Ollama Integration:</strong> Official Python library for streaming responses with context</span>
                    </li>
                  </ul>
                </div>

                <div>
                  <h4 class="font-semibold text-lg mb-4">Local LLM Integration</h4>
                  <div class="bg-white p-4 rounded-lg border">
                    <div class="mb-3">
                      <span class="font-medium">General-Purpose Models:</span>
                      <span class="text-gray-600 ml-2">TinyLlama for lightweight conversation</span>
                    </div>
                    <div class="mb-3">
                      <span class="font-medium">Model Management:</span>
                      <span class="text-gray-600 ml-2">Ollama framework for easy switching</span>
                    </div>
                    <div>
                      <span class="font-medium">Context Handling:</span>
                      <span class="text-gray-600 ml-2">Conversation history maintained for coherent responses</span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="highlight-box">
                <h4 class="font-semibold mb-2">Key Features</h4>
                <p>Streaming responses for real-time interaction, conversation history management, and flexible model selection based on hardware capabilities.</p>
              </div>
            </div>

            <div id="code-assistant" class="mb-12">
              <h3 class="font-serif text-2xl font-semibold mb-6 text-gray-800">The Code Assistant Component</h3>

              <div class="grid md:grid-cols-2 gap-8 mb-8">
                <div>
                  <h4 class="font-semibold text-lg mb-4">Specialized Architecture</h4>
                  <ul class="space-y-3 text-gray-700">
                    <li class="flex items-start">
                      <i class="fas fa-code text-blue-500 mt-1 mr-3"></i>
                      <span><strong>Code Editor UI:</strong> Dedicated interface with syntax highlighting</span>
                    </li>
                    <li class="flex items-start">
                      <i class="fas fa-cogs text-green-500 mt-1 mr-3"></i>
                      <span><strong>Code Processing API:</strong> Specialized backend for code-related queries</span>
                    </li>
                    <li class="flex items-start">
                      <i class="fas fa-microchip text-purple-500 mt-1 mr-3"></i>
                      <span><strong>Code-Specific LLMs:</strong> CodeLlama or Mistral-Instruct for programming tasks</span>
                    </li>
                  </ul>
                </div>

                <div>
                  <h4 class="font-semibold text-lg mb-4">Capabilities</h4>
                  <div class="space-y-3">
                    <div class="bg-white p-4 rounded-lg border">
                      <div class="flex items-center mb-2">
                        <i class="fas fa-file-code text-orange-500 mr-2"></i>
                        <span class="font-medium">Code Generation</span>
                      </div>
                      <p class="text-gray-600 text-sm">Generate code snippets from natural language descriptions</p>
                    </div>
                    <div class="bg-white p-4 rounded-lg border">
                      <div class="flex items-center mb-2">
                        <i class="fas fa-question-circle text-blue-500 mr-2"></i>
                        <span class="font-medium">Code Explanation</span>
                      </div>
                      <p class="text-gray-600 text-sm">Explain existing code functionality and structure</p>
                    </div>
                    <div class="bg-white p-4 rounded-lg border">
                      <div class="flex items-center mb-2">
                        <i class="fas fa-automated text-green-500 mr-2"></i>
                        <span class="font-medium">Boilerplate Automation</span>
                      </div>
                      <p class="text-gray-600 text-sm">Automate generation of common boilerplate code</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div id="research-tool">
              <h3 class="font-serif text-2xl font-semibold mb-6 text-gray-800">The Research Tool Component</h3>

              <div class="mb-8">
                <h4 class="font-semibold text-lg mb-4">RAG Pipeline Architecture</h4>
                <div class="architecture-diagram">
                  <div class="mermaid-container">
                    <div class="mermaid-controls">
                      <button class="mermaid-control-btn zoom-in" title="放大">
                        <i class="fas fa-search-plus"></i>
                      </button>
                      <button class="mermaid-control-btn zoom-out" title="缩小">
                        <i class="fas fa-search-minus"></i>
                      </button>
                      <button class="mermaid-control-btn reset-zoom" title="重置">
                        <i class="fas fa-expand-arrows-alt"></i>
                      </button>
                      <button class="mermaid-control-btn fullscreen" title="全屏查看">
                        <i class="fas fa-expand"></i>
                      </button>
                    </div>
                    <div class="mermaid">
                      graph LR
                      A["Document Upload"] --> B["Document Ingestion"]
                      B --> C["Chunking"]
                      C --> D["Vector Embedding"]
                      D --> E["Vector Storage"]
                      E --> F["Query Processing"]
                      F --> G["Similarity Search"]
                      G --> H["Context Augmentation"]
                      H --> I["LLM Response Generation"]
                      I --> J["Response to User"]

                      style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
                      style B fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
                      style C fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
                      style D fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
                      style E fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
                      style F fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
                      style G fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
                      style H fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
                      style I fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
                      style J fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
                    </div>
                  </div>
                </div>
              </div>

              <div class="grid md:grid-cols-3 gap-6">
                <div class="bg-white p-6 rounded-lg border">
                  <h5 class="font-semibold mb-3">Document Processing</h5>
                  <ul class="text-gray-600 text-sm space-y-2">
                    <li>• File upload and parsing</li>
                    <li>• Text chunking for context</li>
                    <li>• Metadata extraction</li>
                  </ul>
                </div>

                <div class="bg-white p-6 rounded-lg border">
                  <h5 class="font-semibold mb-3">Vector Embeddings</h5>
                  <ul class="text-gray-600 text-sm space-y-2">
                    <li>• Sentence Transformers</li>
                    <li>• Semantic representation</li>
                    <li>• Dimensionality optimization</li>
                  </ul>
                </div>

                <div class="bg-white p-6 rounded-lg border">
                  <h5 class="font-semibold mb-3">Retrieval & Generation</h5>
                  <ul class="text-gray-600 text-sm space-y-2">
                    <li>• Similarity search (Faiss)</li>
                    <li>• Context augmentation</li>
                    <li>• Prompt engineering</li>
                  </ul>
                </div>
              </div>

              <div class="highlight-box mt-6">
                <h4 class="font-semibold mb-2">LangChain Integration</h4>
                <p>The RAG pipeline uses LangChain for orchestration, providing tools and abstractions for building sophisticated document interaction systems with local LLMs.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Ollama Integration -->
      <section id="ollama-integration" class="py-16 bg-white">
        <div class="container mx-auto px-6">
          <div class="section-card">
            <h2 class="font-serif text-3xl font-bold mb-8 text-gray-800">Local LLM Integration with Ollama</h2>

            <p class="text-lg leading-relaxed mb-8">
              Ollama serves as the core engine for local LLM deployment and management, functioning analogously to Docker for AI models. It simplifies the process of running open-source LLMs on local hardware, abstracting away complexities of model configuration and resource optimization <a href="https://www.cohorte.co/blog/using-ollama-with-python-step-by-step-guide" class="citation-link" target="_blank">[207]</a>.
            </p>

            <div class="grid md:grid-cols-2 gap-8 mb-8">
              <div>
                <h3 class="font-serif text-xl font-semibold mb-4">Model Management Engine</h3>
                <div class="space-y-4">
                  <div class="bg-gray-50 p-4 rounded-lg">
                    <h4 class="font-medium mb-2">Docker Container Setup</h4>
                    <p class="text-gray-600 text-sm">Ollama runs in a dedicated container with volume mounting for persistent model storage at
                      <code>/root/.ollama</code>
                    </p>
                  </div>
                  <div class="bg-gray-50 p-4 rounded-lg">
                    <h4 class="font-medium mb-2">Model Selection</h4>
                    <p class="text-gray-600 text-sm">Support for TinyLlama (lightweight) and GPT-OSS:20b (powerful) with
                      <code>ollama pull</code> commands
                    </p>
                  </div>
                  <div class="bg-gray-50 p-4 rounded-lg">
                    <h4 class="font-medium mb-2">REST API Access</h4>
                    <p class="text-gray-600 text-sm">Exposes API at
                      <code>http://ollama:11434</code> for inter-service communication
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h3 class="font-serif text-xl font-semibold mb-4">Python Integration</h3>
                <div class="bg-gray-900 text-green-400 p-4 rounded-lg mb-4">
                  <pre class="text-sm"># Install Ollama Python library
pip install ollama

# Basic usage example
import ollama

# Generate text response
response = ollama.generate(
    model='tinyllama',
    prompt='Hello, how are you?'
)

# Chat with context
messages = [
    {'role': 'user', 'content': 'Hello!'},
    {'role': 'assistant', 'content': 'Hi there!'},
    {'role': 'user', 'content': 'How are you?'}
]

response = ollama.chat(
    model='tinyllama',
    messages=messages,
    stream=True  # For real-time responses
)</pre>
                </div>

                <div class="space-y-3">
                  <div class="flex items-center">
                    <i class="fas fa-check-circle text-green-500 mr-2"></i>
                    <span class="text-sm">Streaming responses for real-time interaction</span>
                  </div>
                  <div class="flex items-center">
                    <i class="fas fa-check-circle text-green-500 mr-2"></i>
                    <span class="text-sm">Function calling for agentic behavior <a href="https://ollama.com/blog/functions-as-tools" class="citation-link" target="_blank">[208]</a></span>
                  </div>
                  <div class="flex items-center">
                    <i class="fas fa-check-circle text-green-500 mr-2"></i>
                    <span class="text-sm">Multi-modal support for advanced models</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="highlight-box">
              <h4 class="font-semibold mb-2">Advanced Feature: Function Calling</h4>
              <p class="mb-3">Function calling enables sophisticated AI agents by allowing LLMs to interact with external tools and APIs:</p>
              <ul class="space-y-1 text-sm">
                <li>• Code execution in sandboxed environments</li>
                <li>• Vector database queries for research tool</li>
                <li>• External API integrations for extended capabilities</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <!-- Data Storage -->
      <section id="data-storage" class="py-16 bg-gray-50">
        <div class="container mx-auto px-6">
          <div class="section-card">
            <h2 class="font-serif text-3xl font-bold mb-8 text-gray-800">Data Storage and Management</h2>

            <div class="grid md:grid-cols-2 gap-8">
              <div>
                <h3 class="font-serif text-xl font-semibold mb-6">User Data Storage</h3>
                <div class="space-y-4">
                  <div class="bg-white p-6 rounded-lg border">
                    <div class="flex items-center mb-3">
                      <i class="fas fa-database text-blue-500 text-xl mr-3"></i>
                      <h4 class="font-semibold">Relational Database</h4>
                    </div>
                    <p class="text-gray-600 mb-3">SQLite or PostgreSQL for structured data storage:</p>
                    <ul class="text-sm text-gray-600 space-y-1">
                      <li>• Chat histories and conversation context</li>
                      <li>• User sessions and preferences</li>
                      <li>• Application configuration</li>
                    </ul>
                  </div>

                  <div class="bg-white p-6 rounded-lg border">
                    <div class="flex items-center mb-3">
                      <i class="fas fa-hdd text-green-500 text-xl mr-3"></i>
                      <h4 class="font-semibold">Persistent Volumes</h4>
                    </div>
                    <p class="text-gray-600 text-sm">Docker volumes ensure data persistence across container restarts and updates.</p>
                  </div>
                </div>
              </div>

              <div>
                <h3 class="font-serif text-xl font-semibold mb-6">Research Knowledge Base</h3>
                <div class="space-y-4">
                  <div class="bg-white p-6 rounded-lg border">
                    <div class="flex items-center mb-3">
                      <i class="fas fa-vector-square text-purple-500 text-xl mr-3"></i>
                      <h4 class="font-semibold">Vector Database</h4>
                    </div>
                    <p class="text-gray-600 mb-3">Faiss or ChromaDB for efficient vector operations:</p>
                    <ul class="text-sm text-gray-600 space-y-1">
                      <li>• Document embeddings storage</li>
                      <li>• Similarity search indices</li>
                      <li>• Metadata management</li>
                    </ul>
                  </div>

                  <div class="bg-white p-6 rounded-lg border">
                    <div class="flex items-center mb-3">
                      <i class="fas fa-file-alt text-orange-500 text-xl mr-3"></i>
                      <h4 class="font-semibold">Document Storage</h4>
                    </div>
                    <p class="text-gray-600 text-sm">File system storage for original documents with metadata indexing.</p>
                  </div>
                </div>
              </div>
            </div>

            <div class="mt-8 p-6 bg-blue-50 rounded-lg">
              <h4 class="font-semibold text-lg mb-3 text-blue-800">Data Flow Architecture</h4>
              <div class="architecture-diagram">
                <div class="mermaid-container">
                  <div class="mermaid-controls">
                    <button class="mermaid-control-btn zoom-in" title="放大">
                      <i class="fas fa-search-plus"></i>
                    </button>
                    <button class="mermaid-control-btn zoom-out" title="缩小">
                      <i class="fas fa-search-minus"></i>
                    </button>
                    <button class="mermaid-control-btn reset-zoom" title="重置">
                      <i class="fas fa-expand-arrows-alt"></i>
                    </button>
                    <button class="mermaid-control-btn fullscreen" title="全屏查看">
                      <i class="fas fa-expand"></i>
                    </button>
                  </div>
                  <div class="mermaid">
                    graph TD
                    A["User Input"] --> B["API Gateway"]
                    B --> C["Service Processing"]
                    C --> D{"Data Type"}

                    D -->|"Chat Data"| E["Relational DB"]
                    D -->|"Code Data"| E
                    D -->|"Document Data"| F["Vector DB"]
                    D -->|"Embeddings"| F

                    E --> G["User Interface"]
                    F --> H["RAG Pipeline"]
                    H --> G

                    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
                    style B fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
                    style C fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
                    style D fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
                    style E fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
                    style F fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
                    style G fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
                    style H fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#000
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Implementation -->
      <section id="implementation" class="py-16 bg-white">
        <div class="container mx-auto px-6">
          <div class="section-card">
            <h2 class="font-serif text-3xl font-bold mb-8 text-gray-800">Implementation Roadmap</h2>

            <div class="grid md:grid-cols-3 gap-8 mb-8">
              <div class="bg-green-50 p-6 rounded-lg">
                <div class="flex items-center mb-4">
                  <i class="fas fa-rocket text-green-600 text-2xl mr-3"></i>
                  <h3 class="font-semibold text-lg">Phase 1: Foundation</h3>
                </div>
                <ul class="space-y-2 text-sm text-green-700">
                  <li>• Docker Compose setup</li>
                  <li>• Ollama service integration</li>
                  <li>• Basic UI framework</li>
                  <li>• API Gateway implementation</li>
                  <li>• Chatbot service development</li>
                </ul>
              </div>

              <div class="bg-blue-50 p-6 rounded-lg">
                <div class="flex items-center mb-4">
                  <i class="fas fa-code text-blue-600 text-2xl mr-3"></i>
                  <h3 class="font-semibold text-lg">Phase 2: Enhancement</h3>
                </div>
                <ul class="space-y-2 text-sm text-blue-700">
                  <li>• Code Assistant development</li>
                  <li>• Vector database setup</li>
                  <li>• RAG pipeline implementation</li>
                  <li>• Research tool integration</li>
                  <li>• User data persistence</li>
                </ul>
              </div>

              <div class="bg-purple-50 p-6 rounded-lg">
                <div class="flex items-center mb-4">
                  <i class="fas fa-cogs text-purple-600 text-2xl mr-3"></i>
                  <h3 class="font-semibold text-lg">Phase 3: Optimization</h3>
                </div>
                <ul class="space-y-2 text-sm text-purple-700">
                  <li>• Performance tuning</li>
                  <li>• Advanced function calling</li>
                  <li>• Model switching logic</li>
                  <li>• UI/UX refinement</li>
                  <li>• Documentation & testing</li>
                </ul>
              </div>
            </div>

            <div class="highlight-box">
              <h3 class="font-semibold text-lg mb-4">Success Metrics</h3>
              <div class="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 class="font-medium mb-2">Technical Metrics</h4>
                  <ul class="space-y-1 text-sm">
                    <li>• Response latency < 500ms for chat interactions</li>
                    <li>• Document processing time < 1s per page</li>
                    <li>• Memory usage optimized for 8GB RAM systems</li>
                    <li>• Startup time < 30s for all services</li>
                  </ul>
                </div>
                <div>
                  <h4 class="font-medium mb-2">User Experience Metrics</h4>
                  <ul class="space-y-1 text-sm">
                    <li>• Unified interface for all three use cases</li>
                    <li>• Seamless switching between functions</li>
                    <li>• Intuitive document management</li>
                    <li>• Context-aware conversation flow</li>
                  </ul>
                </div>
              </div>
            </div>

            <div class="mt-8 p-6 bg-gray-50 rounded-lg">
              <h3 class="font-semibold text-lg mb-4">Technical Considerations</h3>
              <div class="grid md:grid-cols-2 gap-6 text-sm">
                <div>
                  <h4 class="font-medium mb-2">Hardware Requirements</h4>
                  <p class="text-gray-600 mb-2"><strong>Minimum:</strong> 8GB RAM, 20GB disk space, modern CPU</p>
                  <p class="text-gray-600"><strong>Recommended:</strong> 16GB+ RAM, GPU acceleration, SSD storage</p>
                </div>
                <div>
                  <h4 class="font-medium mb-2">Model Selection Strategy</h4>
                  <p class="text-gray-600 mb-2"><strong>Lightweight:</strong> TinyLlama for general tasks (2GB RAM)</p>
                  <p class="text-gray-600"><strong>Powerful:</strong> GPT-OSS:20b for complex tasks (16GB+ RAM)</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>

    <script>
        // Initialize Mermaid with custom configuration
        mermaid.initialize({ 
            startOnLoad: true, 
            theme: 'default',
            themeVariables: {
                primaryColor: '#e0f2fe',
                primaryTextColor: '#000',
                primaryBorderColor: '#0277bd',
                lineColor: '#374151',
                secondaryColor: '#f3e5f5',
                tertiaryColor: '#e8f5e8',
                background: '#ffffff',
                mainBkg: '#ffffff',
                secondaryBkg: '#f8fafc',
                tertiaryBkg: '#f1f5f9'
            }
        });

        // Initialize Mermaid Controls for zoom and pan
        function initializeMermaidControls() {
            const containers = document.querySelectorAll('.mermaid-container');

            containers.forEach(container => {
            const mermaidElement = container.querySelector('.mermaid');
            let scale = 1;
            let isDragging = false;
            let startX, startY, translateX = 0, translateY = 0;

            // 触摸相关状态
            let isTouch = false;
            let touchStartTime = 0;
            let initialDistance = 0;
            let initialScale = 1;
            let isPinching = false;

            // Zoom controls
            const zoomInBtn = container.querySelector('.zoom-in');
            const zoomOutBtn = container.querySelector('.zoom-out');
            const resetBtn = container.querySelector('.reset-zoom');
            const fullscreenBtn = container.querySelector('.fullscreen');

            function updateTransform() {
                mermaidElement.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;

                if (scale > 1) {
                container.classList.add('zoomed');
                } else {
                container.classList.remove('zoomed');
                }

                mermaidElement.style.cursor = isDragging ? 'grabbing' : 'grab';
            }

            if (zoomInBtn) {
                zoomInBtn.addEventListener('click', () => {
                scale = Math.min(scale * 1.25, 4);
                updateTransform();
                });
            }

            if (zoomOutBtn) {
                zoomOutBtn.addEventListener('click', () => {
                scale = Math.max(scale / 1.25, 0.3);
                if (scale <= 1) {
                    translateX = 0;
                    translateY = 0;
                }
                updateTransform();
                });
            }

            if (resetBtn) {
                resetBtn.addEventListener('click', () => {
                scale = 1;
                translateX = 0;
                translateY = 0;
                updateTransform();
                });
            }

            if (fullscreenBtn) {
                fullscreenBtn.addEventListener('click', () => {
                if (container.requestFullscreen) {
                    container.requestFullscreen();
                } else if (container.webkitRequestFullscreen) {
                    container.webkitRequestFullscreen();
                } else if (container.msRequestFullscreen) {
                    container.msRequestFullscreen();
                }
                });
            }

            // Mouse Events
            mermaidElement.addEventListener('mousedown', (e) => {
                if (isTouch) return; // 如果是触摸设备，忽略鼠标事件

                isDragging = true;
                startX = e.clientX - translateX;
                startY = e.clientY - translateY;
                mermaidElement.style.cursor = 'grabbing';
                updateTransform();
                e.preventDefault();
            });

            document.addEventListener('mousemove', (e) => {
                if (isDragging && !isTouch) {
                translateX = e.clientX - startX;
                translateY = e.clientY - startY;
                updateTransform();
                }
            });

            document.addEventListener('mouseup', () => {
                if (isDragging && !isTouch) {
                isDragging = false;
                mermaidElement.style.cursor = 'grab';
                updateTransform();
                }
            });

            document.addEventListener('mouseleave', () => {
                if (isDragging && !isTouch) {
                isDragging = false;
                mermaidElement.style.cursor = 'grab';
                updateTransform();
                }
            });

            // 获取两点之间的距离
            function getTouchDistance(touch1, touch2) {
                return Math.hypot(
                touch2.clientX - touch1.clientX,
                touch2.clientY - touch1.clientY
                );
            }

            // Touch Events - 触摸事件处理
            mermaidElement.addEventListener('touchstart', (e) => {
                isTouch = true;
                touchStartTime = Date.now();

                if (e.touches.length === 1) {
                // 单指拖动
                isPinching = false;
                isDragging = true;

                const touch = e.touches[0];
                startX = touch.clientX - translateX;
                startY = touch.clientY - translateY;

                } else if (e.touches.length === 2) {
                // 双指缩放
                isPinching = true;
                isDragging = false;

                const touch1 = e.touches[0];
                const touch2 = e.touches[1];
                initialDistance = getTouchDistance(touch1, touch2);
                initialScale = scale;
                }

                e.preventDefault();
            }, { passive: false });

            mermaidElement.addEventListener('touchmove', (e) => {
                if (e.touches.length === 1 && isDragging && !isPinching) {
                // 单指拖动
                const touch = e.touches[0];
                translateX = touch.clientX - startX;
                translateY = touch.clientY - startY;
                updateTransform();

                } else if (e.touches.length === 2 && isPinching) {
                // 双指缩放
                const touch1 = e.touches[0];
                const touch2 = e.touches[1];
                const currentDistance = getTouchDistance(touch1, touch2);

                if (initialDistance > 0) {
                    const newScale = Math.min(Math.max(
                    initialScale * (currentDistance / initialDistance),
                    0.3
                    ), 4);
                    scale = newScale;
                    updateTransform();
                }
                }

                e.preventDefault();
            }, { passive: false });

            mermaidElement.addEventListener('touchend', (e) => {
                // 重置状态
                if (e.touches.length === 0) {
                isDragging = false;
                isPinching = false;
                initialDistance = 0;

                // 延迟重置isTouch，避免鼠标事件立即触发
                setTimeout(() => {
                    isTouch = false;
                }, 100);
                } else if (e.touches.length === 1 && isPinching) {
                // 从双指变为单指，切换为拖动模式
                isPinching = false;
                isDragging = true;

                const touch = e.touches[0];
                startX = touch.clientX - translateX;
                startY = touch.clientY - translateY;
                }

                updateTransform();
            });

            mermaidElement.addEventListener('touchcancel', (e) => {
                isDragging = false;
                isPinching = false;
                initialDistance = 0;

                setTimeout(() => {
                isTouch = false;
                }, 100);

                updateTransform();
            });

            // Enhanced wheel zoom with better center point handling
            container.addEventListener('wheel', (e) => {
                e.preventDefault();
                const rect = container.getBoundingClientRect();
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;

                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                const newScale = Math.min(Math.max(scale * delta, 0.3), 4);

                // Adjust translation to zoom towards center
                if (newScale !== scale) {
                const scaleDiff = newScale / scale;
                translateX = translateX * scaleDiff;
                translateY = translateY * scaleDiff;
                scale = newScale;

                if (scale <= 1) {
                    translateX = 0;
                    translateY = 0;
                }

                updateTransform();
                }
            });

            // Initialize display
            updateTransform();
            });
        }

        // Initialize Mermaid and controls when page loads
        document.addEventListener('DOMContentLoaded', function() {
            mermaid.initialize({ 
                startOnLoad: true, 
                theme: 'default',
                themeVariables: {
                    primaryColor: '#e0f2fe',
                    primaryTextColor: '#000',
                    primaryBorderColor: '#0277bd',
                    lineColor: '#374151',
                    secondaryColor: '#f3e5f5',
                    tertiaryColor: '#e8f5e8',
                    background: '#ffffff',
                    mainBkg: '#ffffff',
                    secondaryBkg: '#f8fafc',
                    tertiaryBkg: '#f1f5f9'
                }
            });
            
            // Wait for mermaid to render, then initialize controls
            setTimeout(initializeMermaidControls, 1000);
        });

        // Smooth scrolling for TOC links
        document.querySelectorAll('.toc-link').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update active link
                    document.querySelectorAll('.toc-link').forEach(l => l.classList.remove('active'));
                    this.classList.add('active');
                }
            });
        });

        // Update active TOC link on scroll
        window.addEventListener('scroll', function() {
            const sections = document.querySelectorAll('section[id]');
            const tocLinks = document.querySelectorAll('.toc-link');
            
            let current = '';
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.clientHeight;
                if (window.pageYOffset >= sectionTop - 100) {
                    current = section.getAttribute('id');
                }
            });
            
            tocLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + current) {
                    link.classList.add('active');
                }
            });
        });

        // Mobile TOC toggle (for responsive design)
        const tocToggle = document.createElement('button');
        tocToggle.innerHTML = '<i class="fas fa-bars"></i>';
        tocToggle.className = 'fixed top-4 left-4 z-50 bg-white p-3 rounded-lg shadow-lg lg:hidden';
        tocToggle.onclick = function(e) {
            e.stopPropagation();
            document.querySelector('.toc-fixed').classList.toggle('open');
        };
        document.body.appendChild(tocToggle);

        // Close TOC when clicking outside
        document.addEventListener('click', function(e) {
            const toc = document.querySelector('.toc-fixed');
            if (toc.classList.contains('open') && !toc.contains(e.target) && e.target !== tocToggle) {
                toc.classList.remove('open');
            }
        });

        // Add copy functionality to code blocks
        document.querySelectorAll('.bg-gray-900 pre').forEach(block => {
            const copyBtn = document.createElement('button');
            copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
            copyBtn.className = 'absolute top-2 right-2 bg-gray-700 text-gray-300 p-2 rounded hover:bg-gray-600';
            copyBtn.onclick = function() {
                navigator.clipboard.writeText(block.textContent);
                copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            };
            
            block.style.position = 'relative';
            block.appendChild(copyBtn);
        });
    </script>
  </body>

</html>



++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Functional MVP for a Local LLM System: Chatbot, Code Assistant, and Research Tool

## 1. System Architecture Overview

The proposed system is a containerized, multi-functional AI assistant designed to operate entirely on local hardware, addressing privacy concerns and leveraging the capabilities of open-source Large Language Models (LLMs). The architecture is built on a microservices model, ensuring modularity and scalability, with Docker Compose orchestrating the various components. This design supports three primary use cases: a general-purpose chatbot, a specialized code assistant, and a research tool powered by a Retrieval-Augmented Generation (RAG) pipeline. The system is optimized for limited hardware, utilizing lightweight models like TinyLlama and, if resources permit, more powerful models like GPT-OSS:20b, all managed through the Ollama framework. The core principles guiding this architecture are modularity, containerization, and a local-first, privacy-centric approach, ensuring a robust, maintainable, and secure system that can be deployed and run on a personal computer or a small server.

### 1.1 Core Principles for a Lightweight MVP

The development of this Minimum Viable Product (MVP) is guided by a set of core principles designed to ensure the system is lightweight, maintainable, and suitable for deployment on resource-constrained hardware. These principles are not merely theoretical but are reflected in the choice of technologies and the architectural design. The emphasis is on creating a system that is easy to set up, manage, and extend, while prioritizing user privacy and data security. The following subsections detail the key principles that form the foundation of this MVP.

#### 1.1.1 Modularity and Microservices

The architecture is fundamentally based on the principles of modularity and microservices, a design choice that offers significant advantages in terms of flexibility, scalability, and maintainability. Instead of building a monolithic application, the system is decomposed into a collection of smaller, independent services, each responsible for a specific function. This approach is well-documented in various successful AI application deployments . For instance, a typical architecture might include a dedicated service for the language model API, another for the database, a third for the main application logic, and a fourth for the user interface . This separation of concerns allows for independent development, deployment, and scaling of each service. For example, if the research tool requires more processing power for its RAG pipeline, its service can be scaled independently without affecting the chatbot or code assistant services. This modularity also simplifies the process of updating or replacing individual components. If a more efficient vector database becomes available, the research tool's service can be updated to use it without requiring changes to the rest of the system. This design is particularly beneficial for an MVP, as it allows for incremental development and testing of each component, reducing the complexity of the initial build and facilitating future enhancements.

#### 1.1.2 Containerization with Docker and Docker Compose

Containerization is a cornerstone of the system's deployment strategy, with Docker and Docker Compose being the primary tools for packaging, distributing, and running the application. This approach ensures that the system is portable, reproducible, and easy to manage across different environments. Each microservice is encapsulated within its own Docker container, which includes all the necessary dependencies, libraries, and configuration files. This eliminates the "it works on my machine" problem, as the containerized application will run consistently regardless of the underlying host operating system. Docker Compose is used to define and orchestrate the multi-container application, allowing for the entire system to be started, stopped, and managed with a single command. This is particularly advantageous for a local deployment, as it simplifies the setup process significantly. A `docker-compose.yml` file can specify all the services, their dependencies, network configurations, and volume mounts, providing a clear and concise blueprint for the entire system . For example, the compose file can define services for the Ollama LLM engine, the vector database, the backend API, and the frontend UI, and specify how they communicate with each other over a private Docker network. This containerized approach not only simplifies deployment but also enhances security by isolating the services from each other and the host system.

#### 1.1.3 Local-First and Privacy-Centric Design

A key design principle of this MVP is its local-first and privacy-centric nature. The entire system is designed to run on local hardware, without relying on external cloud services or APIs for its core functionality. This is a significant departure from many modern AI applications and offers several important benefits. Firstly, it ensures complete data privacy and security. All user data, including chat histories, code, and research documents, remains on the user's machine and is not sent to any third-party servers. This is a critical feature for users who handle sensitive or confidential information. Secondly, it provides greater control and customization. Users can choose which LLMs to run, modify the system's behavior, and extend its capabilities without being constrained by the limitations of a cloud-based service. This is particularly relevant for developers and researchers who want to experiment with different models and techniques. The use of local LLMs, managed through Ollama, is central to this design. Models like TinyLlama and GPT-OSS:20b can be run entirely offline, providing powerful AI capabilities without the need for an internet connection . This local-first approach also has the benefit of reducing latency, as there is no network overhead for API calls to remote servers. The system is designed to be self-contained, with all necessary components, including the LLM, vector database, and application logic, running within the local Docker environment.

### 1.2 High-Level Component Interaction

The system's architecture is designed around a clear and logical flow of data and control between its various components. The user interacts with a single, unified interface, which in turn communicates with a set of specialized backend services. These services are responsible for handling the different use cases—chat, code assistance, and research—and they all rely on the Ollama framework as the central engine for running the local LLMs. This section provides a high-level overview of how these components interact to deliver the system's functionality.

#### 1.2.1 User Interface (UI) as the Central Interaction Point

The User Interface (UI) serves as the primary point of interaction for the user, providing a unified and intuitive way to access all the system's features. Instead of having separate applications for the chatbot, code assistant, and research tool, the system will feature a single, integrated web-based UI. This approach simplifies the user experience and provides a consistent look and feel across all use cases. The UI will be built using modern web technologies like React or Vue.js, and will communicate with the backend services through a RESTful API. The UI will be designed to be responsive, ensuring a seamless experience on both desktop and mobile devices. It will feature a chat-like interface, where users can type their queries and receive responses in real-time. The UI will also include specific input fields and controls for the different use cases. For example, for the code assistant, there might be a dedicated code editor and a button to trigger code generation. For the research tool, there will be an interface for uploading documents and asking questions about them. The UI will be responsible for capturing user input, sending it to the appropriate backend service, and displaying the responses in a clear and organized manner. This centralized UI design is a key aspect of the system's usability, making it easy for users to switch between different tasks without having to learn multiple interfaces.

#### 1.2.2 API Gateway for Routing Requests

To manage the communication between the UI and the various backend services, the system will employ an API Gateway. This component acts as a single entry point for all client requests, routing them to the appropriate backend service based on the nature of the request. For example, a request related to code generation would be routed to the Code Assistant Service, while a request to query a document would be routed to the Research Tool Service. This routing logic can be based on the URL path, request headers, or other parameters. The API Gateway provides several benefits. It simplifies the client-side code, as the UI only needs to know the address of the API Gateway, rather than the addresses of all the individual services. It also provides a central point for implementing cross-cutting concerns, such as authentication, authorization, logging, and rate limiting. For example, the API Gateway can be configured to require an API key for all requests, ensuring that only authorized users can access the system. It can also log all incoming requests and outgoing responses, providing valuable insights into the system's usage and performance. In a Docker-based deployment, the API Gateway can be implemented using a reverse proxy like Nginx, which is well-suited for this role and can be easily configured to route traffic to the different services . This centralized routing and management of requests is a key aspect of the system's architecture, ensuring a clean and efficient communication flow.

#### 1.2.3 Specialized Backend Services for Each Use Case

The core logic of the system is divided into a set of specialized backend services, each responsible for handling a specific use case. This microservices-based approach allows for a clean separation of concerns and makes the system more modular and maintainable. The three main backend services are the Chatbot Service, the Code Assistant Service, and the Research Tool Service. Each of these services will be implemented as a separate Docker container, and will expose a RESTful API for communication.

*   **Chatbot Service**: This service will handle all general-purpose conversational queries. It will receive user messages from the API Gateway, format them appropriately, and send them to the Ollama LLM for processing. It will then stream the LLM's response back to the UI in real-time. This service will also be responsible for managing the conversation history, ensuring that the LLM has the necessary context to provide coherent and relevant responses.

*   **Code Assistant Service**: This service will be specialized for code-related tasks. It will be able to generate code snippets, explain existing code, and help with debugging. It may use a different, code-specific LLM, such as CodeLlama, which is fine-tuned for programming tasks. The service will receive code-related queries from the UI, process them, and return the generated code or explanations. It may also integrate with other tools, such as a code execution environment, to provide more advanced functionality.

*   **Research Tool Service**: This service will implement the RAG pipeline, allowing users to interact with their own documents. It will handle the ingestion of documents, the generation of vector embeddings, and the storage of these embeddings in a vector database. When a user asks a question, the service will retrieve the most relevant document chunks from the vector database and use them to augment the prompt sent to the LLM, resulting in more accurate and context-aware responses.

This division of labor among specialized services is a key strength of the architecture, as it allows each service to be optimized for its specific task and scaled independently.

#### 1.2.4 Ollama as the Central LLM Engine

At the heart of the system is Ollama, which serves as the central engine for running the local LLMs. Ollama is a powerful and user-friendly framework that simplifies the process of downloading, managing, and running a wide variety of open-source LLMs. It provides a simple command-line interface and a RESTful API for interacting with the models. The backend services will communicate with the Ollama service to generate text, and Ollama will handle all the low-level details of model inference. This includes loading the model into memory, tokenizing the input, running the inference process, and streaming the output back to the client. The use of Ollama provides several advantages. It abstracts away the complexity of working with different LLM formats and inference engines, providing a unified interface for all models. It also handles the process of downloading and managing the model files, which can be quite large. The backend services can simply specify the name of the model they want to use (e.g., `tinyllama` or `gpt-oss:20b`), and Ollama will take care of the rest. This makes it easy to switch between different models or to use different models for different tasks. For example, the Code Assistant Service could use a code-specific model, while the Chatbot Service uses a general-purpose model. The Ollama service will be run as a separate Docker container, and will be accessible to the other services via the internal Docker network. This centralized management of LLMs is a key aspect of the system's design, providing a flexible and powerful foundation for all its AI capabilities.

## 2. Containerized Deployment Strategy

The deployment of this multi-functional AI assistant is designed to be as simple and reproducible as possible, leveraging the power of Docker and Docker Compose. This containerized approach ensures that the entire system, with all its dependencies, can be packaged and run on any machine that supports Docker, without the need for complex manual configuration. The strategy involves defining each component of the system as a separate service in a `docker-compose.yml` file, which orchestrates the creation, networking, and management of the containers. This section details the key aspects of this deployment strategy, including the orchestration process, service definitions, and data persistence.

### 2.1 Docker Compose Orchestration

Docker Compose is the primary tool for orchestrating the deployment of the system's various microservices. It allows for the definition of a multi-container application in a single YAML file, which can then be started, stopped, and managed with simple commands. This approach is ideal for a local deployment, as it simplifies the process of setting up and running the entire system. The `docker-compose.yml` file will define all the services that make up the application, including the web UI, the API gateway, the various backend services, the Ollama LLM engine, and the vector database. It will also specify the dependencies between these services, ensuring that they are started in the correct order. For example, the backend services will depend on the Ollama service, and the research tool service will depend on the vector database. Docker Compose will also manage the creation of a private network for the containers, allowing them to communicate with each other securely and efficiently. This orchestration approach provides a clear and concise blueprint for the entire system, making it easy to understand, manage, and replicate.

#### 2.1.1 Defining Services for Each Component

The `docker-compose.yml` file is the heart of the Docker Compose orchestration, where each component of the system is defined as a separate service. This includes the web UI, the API gateway, the specialized backend services (chatbot, code assistant, research tool), the Ollama engine, and the vector database. For each service, we can specify a variety of configuration options, such as the Docker image to use, the build context, the ports to expose, the environment variables to set, and the volumes to mount. This allows us to tailor the configuration of each service to its specific needs, ensuring that it has the necessary resources and dependencies to function correctly. By defining each component as a separate service, we can also take advantage of Docker Compose's dependency management features, ensuring that services are started in the correct order and that they can communicate with each other as needed.

#### 2.1.2 Managing Inter-Container Communication

A key aspect of the Docker Compose orchestration is the management of inter-container communication. By default, Docker Compose creates a private network for the application, and all the services are connected to this network. This allows the containers to communicate with each other using their service names as hostnames, without the need to expose their ports to the host machine. For example, the backend API service can connect to the Ollama service using the hostname `ollama` and the port `11434`, without needing to know the IP address of the Ollama container. This simplifies the configuration of the services and enhances the security of the system, as the internal communication is isolated from the outside world. The API Gateway, which will be implemented using a reverse proxy like Nginx, will be the only service that exposes its ports to the host machine. It will be configured to route incoming requests to the appropriate backend service based on the URL path. This centralized management of communication ensures a clean and efficient flow of data between the services, and makes the system more robust and secure.

#### 2.1.3 Volume Mounting for Data Persistence

To ensure that data is not lost when the containers are stopped or restarted, the deployment strategy will make extensive use of Docker volumes. Volumes are a mechanism for persisting data generated by and used by Docker containers. The `docker-compose.yml` file will define named volumes for each service that needs to persist data. For example, a volume will be created for the Ollama service to store the downloaded LLM models, so that they don't need to be downloaded every time the container is started. Another volume will be created for the vector database to store the document embeddings and metadata. A volume will also be created for the backend services to store user data, such as chat histories and session information. By mounting these volumes to the appropriate directories within the containers, the data will be persisted on the host machine, even if the containers are removed. This ensures that the system can be stopped and started without losing any important data, and makes it easier to back up and restore the system's state. The use of volumes is a critical aspect of the deployment strategy, ensuring the reliability and durability of the system.

### 2.2 Service Definitions

The `docker-compose.yml` file will provide a detailed definition for each of the services that make up the system. These definitions will specify the Docker image to use, the container name, the ports to expose, the environment variables to set, and the volumes to mount. This section provides an overview of the service definitions for the key components of the system.

#### 2.2.1 Web UI Service (React/Next.js)

The Web UI service will be responsible for providing the user-facing interface of the application. It will be built using a modern JavaScript framework like React or Next.js, and will be served by a lightweight web server like Nginx. The service will be defined in the `docker-compose.yml` file to use a custom-built Docker image, which will be based on a Node.js image for the build process and an Nginx image for serving the static files. The service will expose port 80 (or another port of the user's choice) to the host machine, allowing users to access the application through their web browser. The service will also be configured to communicate with the API Gateway service to send user requests and receive responses. For development purposes, the source code directory of the UI application can be mounted as a volume in the container, allowing for hot-reloading and faster development cycles.

#### 2.2.2 API Gateway Service (FastAPI/Node.js)

The API Gateway service will act as the single entry point for all client requests, routing them to the appropriate backend service. It will be implemented using a high-performance web framework like FastAPI (for Python) or Express (for Node.js), and will be run in a separate Docker container. The service will be defined in the `docker-compose.yml` file to use a custom-built Docker image, and will be configured to listen on a specific port (e.g., 8000). The service will be responsible for implementing the routing logic, which will be based on the URL path of the incoming requests. For example, requests to `/api/chat` will be routed to the Chatbot Service, while requests to `/api/code` will be routed to the Code Assistant Service. The API Gateway will also be responsible for implementing cross-cutting concerns, such as authentication, logging, and rate limiting.

#### 2.2.3 Chatbot Service

The Chatbot Service will be a specialized backend service responsible for handling all general-purpose conversational queries. It will be implemented using a Python framework like FastAPI, and will be run in a separate Docker container. The service will be defined in the `docker-compose.yml` file to use a custom-built Docker image, and will be configured to communicate with the Ollama service to generate responses. The service will receive user messages from the API Gateway, format them into a prompt for the LLM, and send them to the Ollama API. It will then stream the LLM's response back to the API Gateway, which will in turn stream it to the UI. The service will also be responsible for managing the conversation history, which will be stored in a database or a persistent cache.

#### 2.2.4 Code Assistant Service

The Code Assistant Service will be another specialized backend service, focused on providing code-related assistance. It will be implemented using a Python framework like FastAPI, and will be run in a separate Docker container. The service will be defined in the `docker-compose.yml` file to use a custom-built Docker image, and will be configured to communicate with the Ollama service, potentially using a different, code-specific LLM. The service will receive code-related queries from the API Gateway, process them, and return the generated code or explanations. It may also integrate with other tools, such as a code execution environment or a static analysis tool, to provide more advanced functionality.

#### 2.2.5 Research Tool Service

The Research Tool Service will be the backend service responsible for implementing the RAG pipeline. It will be implemented using a Python framework like FastAPI, and will be run in a separate Docker container. The service will be defined in the `docker-compose.yml` file to use a custom-built Docker image, and will be configured to communicate with both the Ollama service and the vector database. The service will handle the ingestion of documents, the generation of vector embeddings, and the storage of these embeddings in the vector database. When a user asks a question, the service will retrieve the most relevant document chunks from the vector database and use them to augment the prompt sent to the LLM.

#### 2.2.6 Ollama Service

The Ollama service will be the central engine for running the local LLMs. It will be run in a separate Docker container, using the official Ollama Docker image. The service will be defined in the `docker-compose.yml` file to expose port 11434, which is the default port for the Ollama API. The service will also be configured to mount a volume to persist the downloaded model files. The backend services will communicate with the Ollama service via its RESTful API, sending prompts and receiving generated text. The Ollama service will handle all the low-level details of model inference, including loading the model into memory, tokenizing the input, and running the inference process.

#### 2.2.7 Vector Database Service (Faiss/ChromaDB)

The Vector Database service will be responsible for storing and retrieving the vector embeddings used by the Research Tool. It will be run in a separate Docker container, using a lightweight and efficient vector database like Faiss or ChromaDB. The service will be defined in the `docker-compose.yml` file to use the appropriate Docker image for the chosen database, and will be configured to expose a port for communication. The service will also be configured to mount a volume to persist the vector data. The Research Tool service will communicate with the Vector Database service to store the embeddings of the ingested documents and to perform similarity searches to retrieve relevant chunks for a given query.

## 3. Core Component: The Chatbot

The chatbot is a central feature of the MVP, providing a general-purpose conversational interface for users. It is designed to be a simple yet effective tool for natural language interaction, leveraging the power of local LLMs to generate coherent and context-aware responses. The architecture of the chatbot is built around a clear separation of concerns, with a dedicated user interface, a backend API for processing requests, and a direct integration with the Ollama engine for model inference. This modular design ensures that the chatbot is easy to develop, maintain, and extend.

### 3.1 Architecture and Implementation

The implementation of the chatbot is divided into three main parts: the user interface, the backend API, and the integration with Ollama. The user interface provides the conversational interface for the user, the backend API handles the logic for processing chat requests, and the integration with Ollama is responsible for generating the responses. This separation of concerns allows for a clean and organized codebase, making it easier to develop and maintain the chatbot.

#### 3.1.1 User Interface for Conversational Interaction

The user interface for the chatbot is designed to be simple and intuitive, providing a familiar chat-like experience for the user. It will be built using a modern JavaScript framework like React or Vue.js, and will feature a message history, an input field for typing messages, and a send button. The UI will be responsible for displaying the conversation history, capturing user input, and sending it to the backend API. It will also be responsible for displaying the responses from the backend API in a clear and organized manner. The UI will be designed to be responsive, ensuring a seamless experience on both desktop and mobile devices.

#### 3.1.2 Backend API for Handling Chat Requests

The backend API for the chatbot is responsible for handling the logic for processing chat requests. It will be built using a lightweight web framework like FastAPI or Node.js, and will expose a RESTful API for communication with the user interface. The API will receive user messages from the UI, format them into a prompt for the LLM, and send them to the Ollama service for processing. It will then receive the response from the Ollama service and return it to the UI. The API will also be responsible for maintaining the conversation history, which is essential for providing context-aware responses.

#### 3.1.3 Integration with Ollama's Chat API

The integration with Ollama's chat API is the core of the chatbot's functionality. The backend API will use the official Ollama Python library to communicate with the Ollama service. The library provides a simple and intuitive interface for interacting with the Ollama API, allowing the backend API to send prompts to the LLM and receive generated responses. The library also supports streaming responses, which is crucial for creating a responsive user interface where the model's output appears in real-time, token by token.

### 3.2 Leveraging Local LLMs

The chatbot leverages the power of local LLMs to generate coherent and context-aware responses. The use of local LLMs ensures that all user data remains on the user's machine, providing a high level of privacy and security. The chatbot is designed to be flexible, allowing for the use of different models depending on the user's needs and hardware capabilities.

#### 3.2.1 Using General-Purpose Models (e.g., TinyLlama)

For general-purpose conversational tasks, the chatbot will use a lightweight and efficient model like TinyLlama. TinyLlama is a small but powerful model that is well-suited for running on limited hardware. It is capable of generating coherent and context-aware responses, making it an ideal choice for the chatbot. The use of a general-purpose model like TinyLlama ensures that the chatbot can handle a wide range of conversational topics, from casual chit-chat to more complex queries.

#### 3.2.2 Model Selection and Management via Ollama

The chatbot's architecture is designed to be flexible, allowing for the easy selection and management of different models. This is achieved through the use of Ollama, which provides a simple and intuitive interface for managing local LLMs. The backend API can be configured to use different models by simply changing the model name in the API call. This allows the user to choose the model that best suits their needs, whether it's a lightweight model like TinyLlama for general-purpose chat or a more powerful model like GPT-OSS:20b for more complex tasks.

#### 3.2.3 Maintaining Conversation History and Context

To provide coherent and context-aware responses, the chatbot maintains a history of the conversation. This is achieved by storing the conversation history in a lightweight database like SQLite or by using a simple in-memory data structure. The conversation history is then passed to the LLM as part of the prompt, allowing the model to understand the context of the conversation and generate a relevant response. This is a crucial feature for creating a natural and engaging conversational experience.

## 4. Core Component: The Code Assistant

The code assistant is a specialized tool designed to help users with their coding tasks. It leverages the power of code-specific LLMs to provide accurate and relevant assistance, including code generation, explanation, and debugging. The architecture of the code assistant is similar to that of the chatbot, with a dedicated user interface, a backend API, and an integration with the Ollama engine. However, it is tailored to the specific needs of developers, with a focus on code-related queries and tasks.

### 4.1 Architecture and Implementation

The implementation of the code assistant is divided into three main parts: the user interface, the backend API, and the integration with code-specific LLMs. The user interface provides a dedicated area for code-related queries, the backend API handles the logic for processing code requests, and the integration with code-specific LLMs is responsible for generating the responses.

#### 4.1.1 User Interface for Code-Related Queries

The user interface for the code assistant is designed to be simple and intuitive, providing a dedicated area for code-related queries. It will feature a code editor for writing and editing code, an input field for typing queries, and a button to trigger code generation or explanation. The UI will be responsible for displaying the generated code or explanations in a clear and organized manner, with syntax highlighting to improve readability.

#### 4.1.2 Backend API for Processing Code Requests

The backend API for the code assistant is responsible for handling the logic for processing code requests. It will be built using a lightweight web framework like FastAPI or Node.js, and will expose a RESTful API for communication with the user interface. The API will receive code-related queries from the UI, format them into a prompt for the LLM, and send them to the Ollama service for processing. It will then receive the response from the Ollama service and return it to the UI.

#### 4.1.3 Integration with Code-Specialized LLMs

The integration with code-specialized LLMs is the core of the code assistant's functionality. The backend API will use the official Ollama Python library to communicate with the Ollama service. The library provides a simple and intuitive interface for interacting with the Ollama API, allowing the backend API to send prompts to the LLM and receive generated responses. The API will be configured to use a code-specific LLM, such as CodeLlama, to provide more accurate and relevant responses.

### 4.2 Leveraging Code-Specific LLMs

The code assistant leverages the power of code-specific LLMs to provide accurate and relevant assistance. These models are fine-tuned on a large corpus of code, making them well-suited for code-related tasks. The use of code-specific LLMs ensures that the code assistant can generate high-quality code and provide insightful explanations.

#### 4.2.1 Using Models like CodeLlama or Mistral-Instruct

For code-related tasks, the code assistant will use a code-specific LLM like CodeLlama or Mistral-Instruct. These models are fine-tuned on a large corpus of code, making them well-suited for tasks like code generation, explanation, and debugging. The use of a code-specific model ensures that the code assistant can provide accurate and relevant assistance, even for complex coding tasks.

#### 4.2.2 Generating Code Snippets and Explanations

The code assistant is capable of generating code snippets in a variety of programming languages. The user can simply type a description of the code they want to generate, and the code assistant will generate the corresponding code snippet. The code assistant can also explain existing code, providing a detailed breakdown of what the code does and how it works. This is a valuable feature for developers who are trying to understand a new codebase or debug a complex piece of code.

#### 4.2.3 Automating Boilerplate Code Generation

The code assistant can also be used to automate the generation of boilerplate code. This is a common task for developers, and it can be time-consuming and tedious. The code assistant can generate boilerplate code for a variety of tasks, such as creating a new class, setting up a database connection, or implementing a RESTful API. This can save developers a significant amount of time and effort, allowing them to focus on more complex and creative tasks.

## 5. Core Component: The Research Tool

The research tool is a powerful feature of the MVP, allowing users to interact with a collection of documents and extract information from them. It uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant information from the documents and then uses an LLM to generate a summary or answer. The architecture of the research tool is more complex than that of the chatbot and code assistant, as it involves a vector database for storing and retrieving document embeddings.

### 5.1 Architecture and Implementation

The implementation of the research tool is divided into three main parts: the user interface, the backend API, and the RAG pipeline. The user interface provides a way for users to upload documents and ask questions about them, the backend API handles the logic for managing research queries, and the RAG pipeline is responsible for retrieving relevant information and generating responses.

#### 5.1.1 User Interface for Document Interaction

The user interface for the research tool is designed to be simple and intuitive, providing a way for users to upload documents and ask questions about them. It will feature a file upload area for uploading documents, an input field for typing questions, and a button to trigger the search. The UI will be responsible for displaying the retrieved information and the generated summary or answer in a clear and organized manner.

#### 5.1.2 Backend API for Managing Research Queries

The backend API for the research tool is responsible for handling the logic for managing research queries. It will be built using a lightweight web framework like FastAPI or Node.js, and will expose a RESTful API for communication with the user interface. The API will receive research queries from the UI, process them using the RAG pipeline, and return the generated summary or answer to the UI.

#### 5.1.3 Retrieval-Augmented Generation (RAG) Pipeline

The RAG pipeline is the core of the research tool's functionality. It is responsible for retrieving relevant information from the documents and then using an LLM to generate a summary or answer. The pipeline consists of several steps, including document ingestion, vector embedding, vector storage and retrieval, and integration with an LLM for response generation.

### 5.2 Building the RAG Pipeline

The RAG pipeline is a complex system that involves several components working together to provide accurate and relevant responses. The following sections provide a detailed overview of the steps involved in building the RAG pipeline.

#### 5.2.1 Document Ingestion and Indexing

The first step in the RAG pipeline is to ingest the documents and index them for retrieval. This involves loading the documents from a file or a directory, splitting them into smaller chunks, and then creating a vector representation of each chunk. The vector representation is a numerical representation of the text that captures its semantic meaning.

#### 5.2.2 Vector Embeddings with Sentence Transformers

To create the vector representations of the text, the RAG pipeline uses a sentence transformer model. A sentence transformer is a type of neural network that is trained to map sentences to a high-dimensional vector space, where semantically similar sentences are close to each other. The RAG pipeline will use a pre-trained sentence transformer model to generate the vector embeddings of the document chunks.

#### 5.2.3 Vector Storage and Retrieval with Faiss

Once the vector embeddings have been generated, they need to be stored in a vector database for efficient retrieval. The RAG pipeline will use a lightweight and efficient vector database like Faiss or ChromaDB to store the vector embeddings. When a user asks a question, the RAG pipeline will generate a vector embedding of the question and then use the vector database to retrieve the most similar document chunks.

#### 5.2.4 Integration with LangChain for Orchestration

To orchestrate the various components of the RAG pipeline, the research tool will use a framework like LangChain. LangChain is a powerful framework for building applications with LLMs, and it provides a set of tools and abstractions for building RAG pipelines. LangChain will be used to manage the document ingestion, vector embedding, vector storage and retrieval, and integration with the LLM for response generation.

## 6. Local LLM Integration with Ollama

The successful implementation of a local, multi-functional AI system hinges on a robust and efficient method for managing and interacting with Large Language Models (LLMs). For this MVP, Ollama has been selected as the core engine for local LLM deployment and management. Ollama is an open-source tool designed to simplify the process of running open-source LLMs on local hardware, abstracting away the complexities of model configuration, dependency management, and resource optimization . It functions analogously to Docker for AI models, allowing developers to "pull" pre-packaged models and run them with a simple command-line interface (CLI) or a programmatic REST API. This approach is particularly well-suited for the project's constraints, which include limited hardware capabilities and a strict requirement for local, private operation. By leveraging Ollama, the system can efficiently run models like TinyLlama and, if resources permit, larger models such as GPT-OSS:20b, without relying on external cloud services, thereby ensuring data privacy and minimizing operational costs . The following sections detail the architecture for integrating Ollama into the containerized Docker environment and the methods for programmatic interaction from other services.

### 6.1 Ollama as the Model Management Engine

Ollama serves as the foundational layer for all LLM-related operations within the system architecture. Its primary role is to act as a self-contained service that manages the entire lifecycle of local language models, from initial download and setup to serving inference requests. This centralized management is critical for maintaining a clean separation of concerns, where other application components, such as the chatbot or research tool, do not need to handle the intricate details of model execution. Instead, they communicate with Ollama through a standardized API, requesting text generation or chat completions. This design not only simplifies the development of downstream services but also enhances the system's modularity and scalability. For instance, if a new, more powerful model becomes available, it can be integrated into the system simply by pulling it via Ollama's CLI and updating the model name in the API calls of the respective services, without requiring any changes to their core logic. The Ollama engine is optimized for performance, utilizing libraries like `llama.cpp` under the hood to ensure efficient execution on both CPU and GPU hardware, which is a crucial feature for achieving acceptable performance on resource-constrained local machines .

#### 6.1.1 Running Ollama in a Docker Container

To align with the project's containerized deployment strategy, Ollama itself will be encapsulated within a dedicated Docker container. This approach ensures that the Ollama service and its dependencies are isolated from the host system and other application services, promoting consistency and reproducibility across different development and deployment environments. The `docker-compose.yml` file will define an `ollama` service, specifying the official Ollama Docker image (e.g., `ollama/ollama`). This container will expose the necessary port (by default, `11434`) to allow other containers in the same Docker network to communicate with the Ollama API. A crucial aspect of this setup is the management of model data. Since LLMs can be several gigabytes in size, it is inefficient and impractical to include them in the Docker image itself. Instead, a Docker volume will be mounted to a directory within the container where Ollama stores its models (e.g., `/root/.ollama`). This volume ensures that downloaded models are persisted on the host machine's filesystem, preventing the need to re-download them every time the Ollama container is restarted or recreated. This strategy significantly speeds up the startup time of the system after the initial model pull and provides a mechanism for backing up or sharing the model cache.

#### 6.1.2 Pulling and Managing Models (TinyLlama, GPT-OSS:20b)

Ollama simplifies the process of acquiring and managing LLMs through its intuitive command-line interface. The system will be designed to work with a predefined set of models suitable for the different use cases. For the chatbot and general-purpose tasks, a lightweight yet capable model like **TinyLlama** is an ideal starting point, given the hardware constraints. For more complex tasks, such as in-depth research or sophisticated code generation, the system can be configured to use a larger model like **GPT-OSS:20b**, provided the local hardware has sufficient RAM and computational power. The process of making these models available to the system is straightforward. Upon the first startup, or as part of an initialization script, the system can execute commands like `ollama pull tinyllama` and `ollama pull gpt-oss:20b`. These commands connect to the Ollama model library, download the specified model weights and configurations, and store them in the mounted volume. The `docker-compose.yml` can be configured to run an initialization container that executes these pull commands, ensuring that the required models are available before the main application services start. This automated setup process makes the system easy to deploy and configure for different hardware capabilities, allowing users to select the models that best fit their needs and resources.

#### 6.1.3 Exposing the Ollama API for Service Communication

Once the Ollama service is running within its Docker container, it exposes a REST API that serves as the primary interface for all other services to interact with the managed LLMs. This API provides endpoints for various operations, including generating text, engaging in chat conversations, and listing available models. By default, the API is accessible at `http://localhost:11434`. In a Docker Compose environment, other services can reach the Ollama container using its service name as the hostname (e.g., `http://ollama:11434`). This service discovery mechanism is a key feature of Docker Compose, simplifying inter-service communication. The API is designed to be simple and intuitive. For example, a `POST` request to the `/api/generate` endpoint with a JSON payload containing the model name and a prompt will return the model's generated text. Similarly, the `/api/chat` endpoint is used for conversational interactions, accepting a list of messages and returning the model's response. This standardized, HTTP-based API allows services written in different programming languages to easily integrate with the LLM engine. The Python-based services in this MVP, for instance, will use the official Ollama Python library, which is a thin wrapper around this REST API, providing a more convenient and Pythonic interface for making requests and handling responses .

### 6.2 Interacting with Ollama via Python

The backend services for the chatbot, code assistant, and research tool will be primarily developed in Python, leveraging its rich ecosystem of libraries for AI and web development. To facilitate seamless communication with the Ollama engine, the official `ollama` Python library will be used. This library provides a clean, high-level interface for interacting with the Ollama REST API, abstracting away the low-level details of making HTTP requests and parsing JSON responses. It allows developers to integrate powerful LLM capabilities into their applications with just a few lines of code, making it an ideal choice for rapidly developing the MVP components . The library supports all the features of the Ollama API, including text generation, chat, streaming responses, and even multi-modal inputs for models that support them, such as LLaVA . This comprehensive feature set ensures that the Python-based services can fully leverage the capabilities of the underlying LLMs managed by Ollama. The following subsections explore the specific methods for integrating Ollama into the Python-based services, from basic API calls to more advanced agentic behaviors.

#### 6.2.1 Using the Official Ollama Python Library

The `ollama` Python library is the recommended and most straightforward way to integrate local LLMs into Python applications. Installation is simple, requiring only a `pip install ollama` command . Once installed, it provides a set of intuitive functions that mirror the functionality of the Ollama CLI and REST API. For instance, the `ollama.generate()` function can be used for single-turn text completion tasks. A developer simply needs to provide the model name and a prompt string, and the function will return the model's response. For more interactive, conversational use cases, such as the chatbot, the `ollama.chat()` function is used. This function takes a model name and a list of messages, where each message is a dictionary with a `role` (e.g., "user", "assistant", "system") and `content`. This allows the application to maintain a conversation history and provide context to the model, which is essential for coherent and context-aware interactions . The library also supports streaming responses by setting the `stream=True` parameter, which is crucial for creating a responsive user interface where the model's output appears in real-time, token by token, rather than being returned all at once after a potentially long delay. This feature significantly improves the perceived performance and user experience of the chatbot and other interactive tools.

A key feature of the Ollama Python library is its support for function calling, which enables the creation of more sophisticated AI agents. This allows an LLM to determine when to call a specific Python function to retrieve information or perform an action, extending its capabilities beyond its internal knowledge. For example, a code assistant could be given a function to execute Python code, or a research tool could be given a function to search a local document database. The library handles the process of making the function's signature and documentation available to the model and then executing the function with the arguments provided by the model in its response . This powerful feature is central to building the advanced use cases required for this MVP, transforming the LLM from a simple text generator into an active agent that can interact with its environment and perform complex tasks.

#### 6.2.2 Making Requests to the Ollama REST API

While the official Python library is the most convenient method, it is also possible to interact with Ollama by making direct HTTP requests to its REST API. This approach offers more flexibility and can be useful in scenarios where the Python library is not available or when integrating with services written in other languages. The API is well-documented and follows standard REST conventions. For example, to generate text, a `POST` request is sent to `http://<ollama-host>:11434/api/generate` with a JSON body containing the `model` and `prompt`. The response is a stream of JSON objects, each representing a token of the generated text. For chat interactions, the endpoint is `/api/chat`, and the request body includes the `model` and a `messages` array. This direct API access can be implemented using any HTTP client library, such as Python's `requests` library. This method provides a lower-level control over the interaction, allowing developers to handle the raw response stream and implement custom logic for processing the model's output. It also serves as the underlying mechanism for the official Python and JavaScript libraries, ensuring that any client that can make HTTP requests can leverage the power of Ollama's local LLMs .

#### 6.2.3 Implementing Function Calling for Advanced Agentic Behavior

The function calling capability of the Ollama Python library is a cornerstone for building the advanced features of the code assistant and research tool. This feature allows the LLM to act as a reasoning engine that can decide to use external tools to accomplish a user's request. The implementation involves several steps. First, the developer defines a standard Python function that performs a specific task. This function should have a clear name, type hints for its parameters, and a descriptive docstring, as this information is used by the library to generate a JSON schema that helps the LLM understand the function's purpose and how to use it . For example, a function to add two numbers might look like this:

```python
def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers and return the result.
    """
    return a + b
```

Next, this function is passed to the `ollama.chat()` call via the `tools` parameter. When the user asks a question like "What is 10 + 10?", the LLM can analyze the request and decide to call the `add_two_numbers` function instead of trying to compute the answer itself. The Ollama library then parses the model's response, extracts the function name and arguments, and executes the corresponding Python function. The result of this function call can then be fed back into the conversation, allowing the LLM to formulate a final answer for the user . This mechanism can be extended to more complex scenarios. For the code assistant, a function could be defined to execute a block of code in a sandboxed environment and return the output. For the research tool, a function could query a vector database to retrieve relevant documents based on a search query. This ability to chain together LLM reasoning with external tool execution is what enables the creation of truly powerful and autonomous AI agents, all running locally and privately within the Dockerized environment.

## 7. Data Storage and Management

A robust data storage and management strategy is essential for the MVP to function correctly and provide a seamless user experience. The system needs to store user data, such as chat histories and session information, as well as the knowledge base for the research tool, which consists of vector embeddings of the ingested documents. The architecture is designed to use lightweight and efficient storage solutions that are well-suited for a local deployment environment.

### 7.1 Persistent Storage for User Data

To ensure that user data is not lost when the containers are stopped or restarted, the system will use persistent storage. This will be achieved through the use of Docker volumes, which allow data to be stored on the host machine's filesystem and mounted into the containers. The system will use a lightweight database to store user data, such as chat histories and session information.

#### 7.1.1 Storing Chat Histories and User Sessions

The chatbot and other interactive features of the system will store user data, such as chat histories and session information, to provide a personalized and context-aware experience. This data will be stored in a lightweight database, such as SQLite or PostgreSQL. The database will be run in a separate Docker container, and a volume will be mounted to persist the data on the host machine. This ensures that the user's data is not lost when the containers are restarted, and it also allows for the data to be backed up and restored.

#### 7.1.2 Using a Lightweight Database (e.g., SQLite, PostgreSQL)

For storing user data, the system will use a lightweight database like SQLite or PostgreSQL. SQLite is a serverless database that is easy to set up and use, making it a good choice for a simple MVP. PostgreSQL is a more powerful and feature-rich database that is well-suited for more complex applications. The choice of database will depend on the specific requirements of the system, but both are good options for a local deployment.

### 7.2 Knowledge Base for the Research Tool

The research tool requires a knowledge base to store the vector embeddings of the ingested documents. This knowledge base will be used to retrieve relevant information in response to user queries. The system will use a vector database to store the vector embeddings, as it is a specialized database that is optimized for storing and querying high-dimensional vectors.

#### 7.2.1 Storing Vector Embeddings in a Vector Database

The research tool will use a vector database to store the vector embeddings of the ingested documents. A vector database is a specialized database that is optimized for storing and querying high-dimensional vectors. The system will use a lightweight and efficient vector database like Faiss or ChromaDB. The vector database will be run in a separate Docker container, and a volume will be mounted to persist the data on the host machine.

#### 7.2.2 Managing Document Collections and Metadata

In addition to storing the vector embeddings, the research tool will also need to manage the document collections and their metadata. This includes information such as the document's title, author, and date of creation. This metadata will be stored in a separate database, such as a relational database or a document database. The metadata will be used to provide additional context for the retrieved information, and it will also be used to filter the search results.
