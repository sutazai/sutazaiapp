FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    jax[cpu] \
    flax \
    optax \
    fastapi \
    uvicorn

COPY web_interface.py .

EXPOSE 8080


# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["python", "web_interface.py"]
