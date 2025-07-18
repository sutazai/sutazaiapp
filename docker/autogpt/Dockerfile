FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Clone AutoGPT and install
RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git /app/autogpt
WORKDIR /app/autogpt/classic

RUN pip install -e .

EXPOSE 8000

CMD ["python", "-m", "autogpt", "--continuous", "--ai-settings-file", "/app/ai_settings.yaml"]
