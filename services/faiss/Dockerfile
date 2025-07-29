FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir faiss-cpu fastapi uvicorn numpy

COPY faiss_service.py /app/

EXPOSE 8100

CMD ["uvicorn", "faiss_service:app", "--host", "0.0.0.0", "--port", "8100"]