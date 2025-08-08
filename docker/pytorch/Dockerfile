FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn

EXPOSE 8888


# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
