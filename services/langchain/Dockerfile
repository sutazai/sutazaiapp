FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    langchain langchain-community langchain-experimental \
    fastapi uvicorn redis chromadb httpx \
    openai tiktoken

COPY langchain_orchestrator.py /app/

EXPOSE 8095

CMD ["uvicorn", "langchain_orchestrator:app", "--host", "0.0.0.0", "--port", "8095"]