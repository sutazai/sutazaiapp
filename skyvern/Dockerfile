FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir playwright \
 && python -m playwright install --with-deps chromium

RUN git clone https://github.com/Skyvern-AI/skyvern.git . \
 && pip install --no-cache-dir -e .

CMD ["sleep","infinity"]
