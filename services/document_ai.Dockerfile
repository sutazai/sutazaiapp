FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Add health check and resource limits
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl --fail http://localhost:8082/health || exit 1

# Resource constraints
ENV OMP_NUM_THREADS=2
ENV OCR_THREAD_LIMIT=4

CMD ["uvicorn", "document_ai:app", "--host", "0.0.0.0", "--port", "8082"]