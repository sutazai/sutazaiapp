# Production-ready Dockerfile for jarvis-task-controller (FastAPI)

FROM python:3.11-slim-bullseye AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Builder: install dependencies in a venv
FROM base AS builder
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# Final: copy app and venv, run as non-root
FROM base AS final
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . /app
RUN useradd -r -u 10002 -g root appuser \
    && chown -R appuser:root /app
USER appuser

# Expect src/main.py to start FastAPI with Uvicorn
CMD ["python", "src/main.py"]

