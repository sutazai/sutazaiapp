FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Clone Documind
RUN git clone https://github.com/DocumindHQ/documind.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask requests pypdf2 python-docx

VOLUME ["/app/data"]
EXPOSE 8080

# Create web interface
COPY web_interface.py .

CMD ["python", "web_interface.py"]
