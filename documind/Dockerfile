FROM node:20-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/DocumindHQ/documind.git .
RUN npm ci --legacy-peer-deps
CMD ["sleep","infinity"]
