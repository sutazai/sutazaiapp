# SutazAI System Architecture

This document provides a high-level overview of the SutazAI system architecture.

## Core Components

The SutazAI system is composed of the following core components:

*   **Frontend:** A Streamlit-based web interface for interacting with the system.
*   **Backend:** A FastAPI-based backend that serves the API and orchestrates the AI agents.
*   **Database:** A PostgreSQL database for storing persistent data.
*   **Cache:** A Redis cache for storing temporary data.
*   **Vector Stores:** ChromaDB and Qdrant for storing and retrieving vector embeddings.
*   **Model Serving:** Ollama for serving local language models.
*   **AI Agents:** A collection of specialized AI agents for performing various tasks.

## Service Communication

The services communicate with each other using a combination of REST APIs and a message queue. The frontend communicates with the backend via a REST API. The backend communicates with the other services via a combination of REST APIs and a message queue.

## Data Flow

The data flows through the system as follows:

1.  A user interacts with the frontend to submit a request.
2.  The frontend sends the request to the backend.
3.  The backend processes the request and sends it to the appropriate AI agent.
4.  The AI agent processes the request and returns a response to the backend.
5.  The backend sends the response to the frontend.
6.  The frontend displays the response to the user.