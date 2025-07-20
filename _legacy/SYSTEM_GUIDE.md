# SutazAI System Guide

This document provides a detailed overview of the SutazAI system architecture, components, and functionality.

## 1. System Architecture

The SutazAI system is a microservices-based architecture orchestrated by Docker Compose. The core components are:

- **Frontend:** A Streamlit-based web UI for interacting with the system.
- **Backend:** A FastAPI-based backend that serves the API and orchestrates the AI agents.
- **Ollama:** The service for running and managing the large language models.
- **Vector Databases:** ChromaDB and Qdrant for vector storage and retrieval.
- **Databases:** PostgreSQL for relational data and Redis for caching.
- **Agents:** A suite of specialized AI agents for various tasks.
- **Monitoring:** Prometheus and Grafana for system monitoring and observability.

For a detailed breakdown of the architecture, please refer to the `docker-compose.yml` file.

## 2. Core Components

### 2.1. Frontend

The frontend is a Streamlit application that provides a user-friendly interface for interacting with the SutazAI system. It communicates with the backend via a REST API.

### 2.2. Backend

The backend is a FastAPI application that serves as the central hub of the system. It is responsible for:

- **API:** Exposing a REST API for the frontend and other clients.
- **Agent Orchestration:** Managing the lifecycle and task execution of the AI agents.
- **Model Management:** Interacting with the Ollama service to manage the AI models.
- **Database & Cache:** Interacting with the databases and cache.

### 2.3. AI Agents

The SutazAI system includes a variety of specialized AI agents for different tasks. These agents are managed by the multi-agent orchestrator in the backend.

## 3. System Management

The `manage.sh` script is the single entry point for all system operations:

- `./manage.sh start`: Start all services.
- `./manage.sh stop`: Stop all services.
- `./manage.sh restart`: Restart all services.
- `./manage.sh status`: Show the status of all services.
- `./manage.sh logs`: Tail the logs for all services.
- `./manage.sh logs <service_name>`: Tail the logs for a specific service.
- `./manage.sh build`: Build or rebuild all services.
- `./manage.sh pull`: Pull the latest Docker images.
- `./manage.sh prune`: Remove old containers and volumes.

## 4. Monitoring

The system includes a monitoring stack based on Prometheus and Grafana. You can access the Grafana dashboard at [http://localhost:3000](http://localhost:3000).
