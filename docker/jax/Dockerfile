FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install JAX and ecosystem
RUN pip install --no-cache-dir \
    jax[cuda12_pip] \
    jaxlib \
    flax \
    optax \
    distrax \
    chex \
    fastapi \
    uvicorn \
    numpy

# Create JAX API server
RUN echo 'import jax\nimport jax.numpy as jnp\nimport uvicorn\nfrom fastapi import FastAPI\nimport numpy as np\n\napp = FastAPI()\n\n# Check JAX backend\nprint(f"JAX backend: {jax.default_backend()}")\nprint(f"JAX devices: {jax.devices()}")\n\n@app.get("/health")\ndef health():\n    return {\n        "status": "healthy",\n        "jax_version": jax.__version__,\n        "backend": jax.default_backend(),\n        "devices": [str(d) for d in jax.devices()]\n    }\n\n@app.post("/compute")\ndef compute(data: dict):\n    try:\n        operation = data.get("operation", "square")\n        input_data = jnp.array(data.get("input", [1.0, 2.0, 3.0]))\n        \n        if operation == "square":\n            result = jnp.square(input_data)\n        elif operation == "sin":\n            result = jnp.sin(input_data)\n        elif operation == "matmul":\n            matrix = jnp.array(data.get("matrix", [[1, 2], [3, 4]]))\n            result = jnp.matmul(matrix, input_data[:len(matrix)])\n        else:\n            result = input_data\n        \n        return {"result": result.tolist()}\n    except Exception as e:\n        return {"error": str(e)}\n\n@app.post("/gradient")\ndef gradient(data: dict):\n    try:\n        def loss_fn(x):\n            return jnp.sum(jnp.square(x))\n        \n        x = jnp.array(data.get("input", [1.0, 2.0, 3.0]))\n        grad_fn = jax.grad(loss_fn)\n        gradient = grad_fn(x)\n        \n        return {\n            "gradient": gradient.tolist(),\n            "loss": float(loss_fn(x))\n        }\n    except Exception as e:\n        return {"error": str(e)}\n\nif __name__ == "__main__":\n    uvicorn.run(app, host="0.0.0.0", port=8087)' > jax_server.py

# Create necessary directories
RUN mkdir -p /data /logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORM_NAME=cpu

# Expose port
EXPOSE 8087

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8087/health || exit 1

# Run JAX server
CMD ["python", "jax_server.py"]