# SutazAI System Architecture

*This document outlines the planned architecture of the SutazAI system, based on the initial blueprint.*

## 1. Overview

SutazAI is designed as a modular, offline AGI/ASI system with four primary layers:
1.  **UI Layer (Streamlit):** User interaction (Chat, Reports, Code Debug, API Console).
2.  **Backend Layer (FastAPI):** API endpoints, service orchestration (Chat, Document Analysis, Code Tools, etc.), authentication.
3.  **Agent Layer:** Orchestration of various AI agents (LangChain, AutoGen, AutoGPT, LocalAGI, etc.) and specialized tools (TabbyML, Semgrep, Skyvern, GPT-Engineer, Aider).
4.  **Model & Memory Layer:** Core LLMs (via Ollama - e.g., Llama 2, DeepSeek-Coder), Vector Store (ChromaDB + FAISS) for long-term memory and RAG.

(Diagram from blueprint would go here ideally)

## 2. Core Components

*   **`sutazai_agi.core`**: Configuration loading, logging, ethical verifier base.
*   **`sutazai_agi.models`**: `LLMInterface` for interacting with Ollama models (generation, embeddings).
*   **`sutazai_agi.memory`**: `VectorStoreInterface` for ChromaDB interactions (add, query, delete).
*   **`sutazai_agi.agents`**: `AgentManager` to load, dispatch, and manage agent execution. `tool_library` defines available tools. `impl/` contains specific agent framework integrations (starting with LangChain).
*   **`sutazai_agi.backend`**: FastAPI application (`main.py`), API endpoints (`api/endpoints.py`), service logic (`services/chat_service.py`, etc.).
*   **`sutazai_agi.ui`**: Streamlit application (`SutazAI_UI.py`), individual pages (`pages/`).
*   **`sutazai_agi.security`**: (Placeholder) Sandboxing mechanisms, security utilities.
*   **`sutazai_agi.testing`**: (Placeholder) Unit and integration tests.

## 3. Data Flow (Example: Chat Request)

1.  User inputs query in Streamlit UI (`01_Chatbot.py`).
2.  UI sends request to FastAPI backend (`/api/v1/chat`).
3.  FastAPI endpoint (`endpoints.py`) receives request, validates input.
4.  Endpoint calls `ChatService` (`chat_service.py`).
5.  `ChatService` determines target agent (e.g., "LangChain Chat Agent") and calls `AgentManager` (`agent_manager.py`).
6.  `AgentManager` retrieves agent config, loads necessary tools (from `tool_library.py`), wraps them with ethical checks (using `ethical_verifier.py`), and dispatches to the specific agent implementation (e.g., `langchain_agent.py`).
7.  LangChain agent uses `LLMInterface` (`llm_interface.py`) to generate text via Ollama and may use tools (like search via `VectorStoreInterface` (`vector_store.py`)).
8.  Ethical verifier checks tool actions before execution and final output after generation.
9.  Result propagates back through Manager -> Service -> Endpoint -> UI.

## 4. Configuration

*   `config/settings.yaml`: Global settings (models, paths, logging, API keys/secrets).
*   `config/agents.yaml`: Definitions of available agents, their types, models, tools, and specific framework configurations.

## 5. Roadmap Alignment (Initial Focus: Q1)

This structure supports the Q1 goals:
*   Local LLM setup (`models/llm_interface.py`).
*   Vector Memory (`memory/vector_store.py`).
*   Basic Agent Orchestrator (`agents/agent_manager.py`).
*   Ethical Verifier Base (`core/ethical_verifier.py`).
*   Minimal FastAPI & UI (`backend/`, `ui/`).
*   Initial LangChain agent integration (`agents/impl/langchain_agent.py`).

*(This document will be updated as the project progresses through Q2-Q4 milestones.)* 