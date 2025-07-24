FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone FinRobot
RUN git clone https://github.com/AI4Finance-Foundation/FinRobot.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask requests yfinance pandas numpy

VOLUME ["/app/data"]
EXPOSE 8080

# Create web interface
COPY web_interface.py .

CMD ["python", "web_interface.py"]
