FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Poetry
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry

# Clone PrivateGPT
RUN git clone https://github.com/imartinez/privateGPT.git .

# Configure Poetry to not create virtual environment (since we're in container)
ENV POETRY_VENV_IN_PROJECT=false
ENV POETRY_NO_INTERACTION=1

# Install dependencies using Poetry
RUN poetry install --no-dev

# Copy configuration if exists
COPY settings.yaml settings.yaml

EXPOSE 8001

CMD ["poetry", "run", "python", "-m", "scripts.setup", "--profile", "local"]
