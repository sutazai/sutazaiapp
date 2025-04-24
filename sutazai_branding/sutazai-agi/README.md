# SutazAI AGI/ASI System

+## Overview

+This project implements the SutazAI AGI/ASI system as defined in the comprehensive technical blueprint. It aims to create a fully autonomous, offline AI system running on a dedicated server (Dell PowerEdge R720).

**Key Features (Planned):**

*   Offline operation (Air-gapped)
*   Modular architecture (Models, Agents, Backend, UI)
*   Integration of multiple large language models
*   Natural language understanding and generation
*   Advanced reasoning and decision making capabilities
*   Seamless integration with various APIs and services
*   High-performance parallel processing for quick responses

+## Running the System

1.  **Start Ollama Server:**
    ```bash
    # Ensure Ollama service is running in the background
    # (Usually starts automatically after installation, or use systemd/launchd)
    # Alternatively, run manually if needed:
    # ollama serve
    # Or use the provided script:
    # bash scripts/start_ollama.sh
    ```

2.  **Start the FastAPI Backend:**
    ```bash
    bash scripts/start_backend.sh
    # Or manually:
    # uvicorn sutazai_agi.backend.main:app --host 0.0.0.0 --port 8000 --reload # Use --reload for development
    ```

3.  **Start the Streamlit UI:**
    ```bash
    bash scripts/start_ui.sh
    # Or manually:
    # streamlit run sutazai_agi/ui/SutazAI_UI.py
    ```

4.  **Access the UI:**
    Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

+## Project Structure

```python
+/
+├── sutazai_agi/          # Main Python package
+│   ├── core/             # Core utilities, config, constants
+│   ├── models/           # Model loading, interfaces
+│   ├── memory/           # Vector DB setup, retrieval
+│   ├── agents/           # Agent framework integration, tools
+│   ├── backend/          # FastAPI application
+│   ├── ui/               # Streamlit UI components
+│   ├── security/         # Sandboxing, security utils
+│   └── testing/          # Tests
+├── config/               # Configuration files (settings.yaml, agents.yaml)
+├── scripts/              # Startup, installation, maintenance scripts
+├── data/                 # Local datasets, knowledge base, model weights
+├── docs/                 # Project documentation
+├── notebooks/            # Jupyter notebooks
+├── .gitignore
+├── requirements.txt
+└── README.md
+```
+
+## Contributing

Please refer to CONTRIBUTING.md (to be created).
+
+## License

Specify your license here (e.g., MIT, Apache 2.0). 
