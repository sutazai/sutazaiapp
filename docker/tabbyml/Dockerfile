FROM nvidia/cuda:11.8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/* && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://get.docker.com | sh

WORKDIR /app
RUN curl -fsSL https://github.com/TabbyML/tabby/releases/latest/download/tabby_x86_64-manylinux2014 -o tabby
RUN chmod +x tabby

EXPOSE 8080

# Run as non-root user
USER nobody

# Run as non-root user
USER nobody
CMD ["./tabby", "serve", "--model", "CodeLlama-7B", "--host", "0.0.0.0", "--port", "8080"]
