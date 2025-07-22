FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Chrome
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    gnupg \
    unzip \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Clone Browser-Use
RUN git clone https://github.com/browser-use/browser-use.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask requests selenium

# Create configuration
ENV CHROME_PATH=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

VOLUME ["/app/data"]
EXPOSE 8080

# Create web interface
COPY web_interface.py .

CMD ["python", "web_interface.py"]
