# TabbyML CPU-Only Smart Build - Research-based solution
FROM tabbyml/tabby:latest

# SOLUTION: Instead of replacing llama-server, use tabby-cpu exclusively
# This avoids the CUDA dependency issues entirely

# Set environment for CPU-only operation
ENV TABBY_MODEL_CACHE_ROOT=/data
ENV RUST_LOG=error
ENV TABBY_DISABLE_USAGE_COLLECTION=true
ENV TABBY_DEVICE=cpu
ENV CUDA_VISIBLE_DEVICES=""

# Create a smart wrapper that forces CPU mode
RUN echo '#!/bin/bash\n/opt/tabby/bin/tabby-cpu "$@"' > /opt/tabby/bin/tabby-smart-cpu && \
    chmod +x /opt/tabby/bin/tabby-smart-cpu

# Pre-download models using CPU binary to reduce startup time
RUN /opt/tabby/bin/tabby-cpu download --model StarCoder-1B || true
RUN /opt/tabby/bin/tabby-cpu download --model Qwen2-1.5B-Instruct || true

# Use smart CPU wrapper
ENTRYPOINT ["/opt/tabby/bin/tabby-smart-cpu"]

# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["serve", "--model", "StarCoder-1B", "--chat-model", "Qwen2-1.5B-Instruct"]