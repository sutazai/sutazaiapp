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

CMD ["python", "web_interface.py"]
