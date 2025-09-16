# SutazAI Application Overview

**Last Updated:** 2025-09-03  
**Version:** 1.0.0  
**Maintainer:** Development Team  

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Vision](#project-vision)
3. [System Architecture](#system-architecture)
4. [Key Features](#key-features)
5. [Technology Stack](#technology-stack)
6. [Service Components](#service-components)
7. [Quick Links](#quick-links)
8. [Contact Information](#contact-information)

## Executive Summary

SutazAI is a hybrid microservices platform featuring event-driven multi-agent orchestration for intelligent automation and AI-powered workflows. The system combines modern web technologies with advanced AI capabilities through a distributed architecture.

### Core Purpose

- **Intelligent Automation**: AI-driven task orchestration and execution
- **Multi-Agent System**: Coordinated AI agents for specialized tasks
- **Scalable Architecture**: Microservices design for horizontal scaling
- **Event-Driven**: Real-time processing with message queue integration

## Project Vision

### Mission Statement

To provide a robust, scalable platform for AI-powered automation that enables organizations to leverage multiple specialized AI agents working in concert to solve complex problems.

### Primary Goals

1. **Modularity**: Component-based architecture for easy extension
2. **Scalability**: Support from single-user to enterprise deployments
3. **Intelligence**: Leverage cutting-edge AI models and techniques
4. **Reliability**: Production-ready with comprehensive monitoring
5. **Security**: Enterprise-grade security and compliance

## System Architecture

### High-Level Design

```
User Interface (Streamlit:11000)
        ↓ HTTP/WebSocket
API Gateway (Kong:10008-10009)
        ↓ REST/gRPC
Backend Service (FastAPI:10200)
        ↓ Service Manager
┌─────────────────────────────────┐
│  Databases & Message Queues     │
│  • PostgreSQL (10000)            │
│  • Redis (10001)                 │
│  • Neo4j (10002-10003)           │
│  • RabbitMQ (10004-10005)        │
│  • Vector DBs (10100-10103)      │
└─────────────────────────────────┘
        ↓ Orchestration
MCP Bridge (11100)
        ↓ Routing
AI Agents (11401-11801)
```

### Design Principles

- **Separation of Concerns**: Clear boundaries between components
- **Event-Driven Architecture**: Asynchronous communication
- **Microservices Pattern**: Independent, scalable services
- **Container-First**: Docker-based deployment
- **API-First Design**: RESTful and gRPC interfaces

## Key Features

### Current Capabilities

1. **Multi-Agent Orchestration**
   - 18+ specialized MCP servers
   - Intelligent task routing
   - Parallel execution support
   - Session memory persistence

2. **Data Management**
   - Relational data (PostgreSQL)
   - Graph relationships (Neo4j)
   - Vector embeddings (ChromaDB, Qdrant, FAISS)
   - Caching layer (Redis)
   - Message queuing (RabbitMQ)

3. **API Gateway**
   - Rate limiting and throttling
   - Authentication and authorization
   - Request routing and load balancing
   - API versioning support

4. **Monitoring & Observability**
   - Real-time metrics collection
   - Distributed tracing
   - Centralized logging
   - Health check endpoints

5. **Security Features**
   - JWT authentication
   - Role-based access control
   - Encrypted communications
   - Audit logging

## Technology Stack

### Backend Technologies

- **Language**: Python 3.12
- **Framework**: FastAPI
- **ORM**: SQLAlchemy
- **Async**: asyncio, aiohttp
- **Testing**: pytest, pytest-cov

### Frontend Technologies

- **Framework**: Streamlit
- **Language**: Python
- **UI Components**: Custom Streamlit components

### Infrastructure

- **Containerization**: Docker, Docker Compose
- **Orchestration**: Docker Swarm (future: Kubernetes)
- **API Gateway**: Kong
- **Message Queue**: RabbitMQ
- **Caching**: Redis

### Databases

- **Relational**: PostgreSQL 16
- **Graph**: Neo4j Community
- **Vector**: ChromaDB, Qdrant, FAISS
- **Cache**: Redis 7

### AI/ML Technologies

- **MCP Protocol**: Model Context Protocol
- **LLM Integration**: Multiple model support
- **Vector Search**: Semantic similarity
- **Agent Framework**: Custom orchestration

## Service Components

### Core Services

| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| PostgreSQL | 10000 | Primary database | ✅ Active |
| Redis | 10001 | Cache & pub/sub | ✅ Active |
| Neo4j | 10002-10003 | Graph database | ✅ Active |
| RabbitMQ | 10004-10005 | Message queue | ✅ Active |
| Kong Gateway | 10008-10009 | API gateway | ✅ Active |
| Backend API | 10200 | Main API service | ✅ Active |
| Streamlit UI | 11000 | User interface | ✅ Active |
| MCP Bridge | 11100 | Agent orchestration | ✅ Active |

### Vector Databases

| Service | Port | Purpose |
|---------|------|---------|  
| ChromaDB | 10100 | Document embeddings |
| Qdrant | 10101-10102 | High-performance vectors |
| FAISS | 10103 | Fast similarity search |

### MCP Agents

18 specialized agents including:
- filesystem, memory, github
- claude-flow, ruv-swarm
- context7, playwright
- sequential-thinking
- And more...

## Quick Links

### Documentation

- [Setup Guide](./setup/local_dev.md)
- [Architecture Details](./architecture/system_design.md)
- [API Reference](./api/endpoints/)
- [Development Guide](./development/coding_standards.md)
- [Operations Manual](./operations/deployment/procedures.md)

### External Resources

- [Docker Documentation](https://docs.docker.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [MCP Protocol Spec](https://github.com/anthropics/mcp)

## Contact Information

### Development Team

- **Technical Lead**: [Contact via team channels]
- **DevOps**: [Infrastructure team]
- **Security**: [Security team]

### Communication Channels

- **Slack**: #sutazai-dev
- **Email**: dev-team@sutazai.local
- **Issue Tracker**: GitHub Issues

### Support

- **Documentation**: This /docs directory
- **FAQ**: [User FAQ](./user_guides/faq.md)
- **Troubleshooting**: [Troubleshooting Guide](./setup/troubleshooting.md)

---

*This document provides a high-level overview of the SutazAI application. For detailed information on specific topics, please refer to the linked documentation sections.*