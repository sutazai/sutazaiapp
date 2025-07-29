FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn httpx redis asyncio \
    pydantic sqlalchemy asyncpg aiofiles

COPY orchestrator.py /app/

EXPOSE 8200

CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8200"]