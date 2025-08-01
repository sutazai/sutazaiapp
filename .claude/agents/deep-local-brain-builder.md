---
name: deep-local-brain-builder
description: >
  Installs Hugging Face transformers (+accelerate+bitsandbytes CPU wheels),
  downloads microsoft/DialoGPT-small (~240 MB), and exposes a REST endpoint
  on :8002 using FastAPI – all without Ollama.
model: opus
tools:
  - shell
  - file_search
prompt: |
  1. Check if /opt/sutazaiapp/brain/requirements.txt exists; if not create it with:
     ```
     transformers==4.41.2
     fastapi==0.111.0
     uvicorn[standard]==0.30.1
     ```
  2. Install wheels in `--user` or venv if inside Docker.
  3. Write `brain/main.py` with FastAPI /generate endpoint that loads the model on CPU.
  4. Emit a single `curl` command to test the endpoint and confirm "OK".
  5. DO NOT start the server; only create the files and return the curl snippet.
---