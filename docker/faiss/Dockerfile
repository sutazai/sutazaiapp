FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install faiss-cpu numpy scipy flask

COPY faiss_service.py .

EXPOSE 8000
CMD ["python", "faiss_service.py"]
