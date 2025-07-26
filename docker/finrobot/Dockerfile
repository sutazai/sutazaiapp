FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone FinRobot repository
RUN git clone https://github.com/AI4Finance-Foundation/FinRobot.git . || mkdir -p /app

# Install financial analysis dependencies
RUN pip install --no-cache-dir \
    yfinance \
    pandas \
    numpy \
    scikit-learn \
    ta \
    fastapi \
    uvicorn \
    httpx \
    plotly \
    matplotlib

# Create FinRobot service
COPY finrobot_service.py /app/main.py

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]