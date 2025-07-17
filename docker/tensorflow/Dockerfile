FROM tensorflow/tensorflow:2.14.0-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install additional TensorFlow ecosystem
RUN pip install --no-cache-dir \
    tensorflow-hub \
    tensorflow-datasets \
    tensorflow-probability \
    keras \
    fastapi \
    uvicorn \
    transformers

# Create TensorFlow API server
RUN echo 'import tensorflow as tf\nimport uvicorn\nfrom fastapi import FastAPI\nimport numpy as np\n\napp = FastAPI()\n\n# Check GPU availability\nphysical_devices = tf.config.experimental.list_physical_devices("GPU")\nprint(f"GPUs available: {len(physical_devices)}")\n\n@app.get("/health")\ndef health():\n    return {\n        "status": "healthy",\n        "tensorflow_version": tf.__version__,\n        "gpu_available": len(tf.config.experimental.list_physical_devices("GPU")) > 0,\n        "eager_execution": tf.executing_eagerly()\n    }\n\n@app.post("/predict")\ndef predict(data: dict):\n    try:\n        # Simple TensorFlow computation\n        x = tf.constant(data.get("input", [1.0, 2.0, 3.0]))\n        y = tf.square(x)\n        result = y.numpy().tolist()\n        \n        return {"result": result}\n    except Exception as e:\n        return {"error": str(e)}\n\n@app.post("/train")\ndef train(data: dict):\n    try:\n        # Simple linear model training\n        X = tf.constant(data.get("X", [[1.0], [2.0], [3.0]]))\n        y = tf.constant(data.get("y", [2.0, 4.0, 6.0]))\n        \n        model = tf.keras.Sequential([\n            tf.keras.layers.Dense(1, input_shape=(1,))\n        ])\n        \n        model.compile(optimizer="adam", loss="mse")\n        history = model.fit(X, y, epochs=10, verbose=0)\n        \n        return {\n            "final_loss": float(history.history["loss"][-1]),\n            "weights": model.get_weights()[0].tolist(),\n            "bias": float(model.get_weights()[1][0])\n        }\n    except Exception as e:\n        return {"error": str(e)}\n\nif __name__ == "__main__":\n    uvicorn.run(app, host="0.0.0.0", port=8086)' > tensorflow_server.py

# Create necessary directories
RUN mkdir -p /data /logs /models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port
EXPOSE 8086

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8086/health || exit 1

# Run TensorFlow server
CMD ["python", "tensorflow_server.py"]