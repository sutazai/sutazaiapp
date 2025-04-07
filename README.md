# SutazAI AGI/ASI System
+
+## Overview
+
+This project implements the SutazAI AGI/ASI system as defined in the comprehensive technical blueprint. It aims to create a fully autonomous, offline AI system running on a dedicated server (Dell PowerEdge R720).
+
+**Key Features (Planned):**
+*   Offline operation (Air-gapped)
+*   Modular architecture (Models, Agents, Backend, UI)
+*   Integration of multiple open-source LLMs (DeepSeek, Llama) via Ollama
+*   Vector-based memory (ChromaDB + FAISS)
+*   Multiple autonomous agent frameworks (LangChain, AutoGen, AutoGPT, etc.)
+*   Specialized tools (Code generation, editing, analysis, browser emulation)
+*   FastAPI backend for services
+*   Streamlit web UI for interaction
+*   Ethical constraints and sandboxed execution
+*   Path towards self-improvement
+
+## Setup
+
+1.  **Clone the repository:**
+    ```bash
+    git clone <repository_url>
+    cd sutazai-agi # Or the correct directory name
+    ```
+
+2.  **Install System Dependencies:**
+    *   Ensure Python 3.10+ is installed.
+    *   Install Ollama: Follow instructions at [https://ollama.com/](https://ollama.com/)
+    *   Install necessary build tools (e.g., `build-essential`, `cmake` for some dependencies).
+    *   (Optional but Recommended) Set up a Python virtual environment:
+        ```bash
+        python -m venv venv
+        source venv/bin/activate # On Linux/macOS
+        # venv\Scripts\activate # On Windows
+        ```
+
+3.  **Install Python Dependencies:**
+    ```bash
+    pip install -r requirements.txt
+    # Or run the script:
+    # bash scripts/install_dependencies.sh
+    ```
+
+4.  **Download Models via Ollama:**
+    Pull the required models specified in `config/settings.yaml`. For example:
+    ```bash
+    ollama pull llama2
+    ollama pull deepseek-coder:33b
+    # Add other models as needed
+    ```
+
+5.  **Configure Settings:**
+    *   Review and adjust `config/settings.yaml` and `config/agents.yaml` for your environment (e.g., model names, paths).
+
+6.  **Prepare Data Directory:**
+    *   Create necessary subdirectories within `data/` if needed (e.g., for ChromaDB persistence).
+
+## Running the System
+
+1.  **Start Ollama Server:**
+    ```bash
+    # Ensure Ollama service is running in the background
+    # (Usually starts automatically after installation, or use systemd/launchd)
+    # Alternatively, run manually if needed:
+    # ollama serve
+    # Or use the provided script:
+    # bash scripts/start_ollama.sh
+    ```
+
+2.  **Start the FastAPI Backend:**
+    ```bash
+    bash scripts/start_backend.sh
+    # Or manually:
+    # uvicorn sutazai_agi.backend.main:app --host 0.0.0.0 --port 8000 --reload # Use --reload for development
+    ```
+
+3.  **Start the Streamlit UI:**
+    ```bash
+    bash scripts/start_ui.sh
+    # Or manually:
+    # streamlit run sutazai_agi/ui/SutazAI_UI.py
+    ```
+
+4.  **Access the UI:**
+    Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
+
+## Project Structure
+
+```
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
+
+Please refer to CONTRIBUTING.md (to be created).
+
+## License
+
+Specify your license here (e.g., MIT, Apache 2.0). 