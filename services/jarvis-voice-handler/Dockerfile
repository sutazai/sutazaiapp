# Production-ready Dockerfile for jarvis-voice-handler
#
# Rationale:
# - Use a small, official, version-pinned Python base for reproducibility
# - Multi-stage pattern keeps final image slim and secure
# - Installs system deps required for audio (PyAudio/PortAudio)
# - Avoids copying tests/docs and local env into the image

FROM python:3.11-slim-bullseye AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime system dependencies for audio I/O
# - libportaudio2 is required by PyAudio at runtime
# - libasound2 provides ALSA backend commonly used in containers
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Builder stage: dependencies
# -------------------------
FROM base AS builder

# Build dependencies for compiling PyAudio wheels if necessary
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        portaudio19-dev \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# -------------------------
# Final stage: application
# -------------------------
FROM base AS final

# Copy virtualenv from builder and set PATH
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy service source code
COPY . /app

# Security best practice: run as non-root
RUN useradd -r -u 10001 -g root appuser \
    && chown -R appuser:root /app
USER appuser

# Explanatory CMD: service must provide src/main.py
# This keeps container contract explicit while avoiding speculative app code here
CMD ["python", "src/main.py"]

