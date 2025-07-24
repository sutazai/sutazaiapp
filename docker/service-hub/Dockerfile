FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    redis \
    asyncio \
    pydantic \
    websockets

# Copy service implementation
COPY service_hub.py /app/

EXPOSE 8080

CMD ["uvicorn", "service_hub:app", "--host", "0.0.0.0", "--port", "8080"]