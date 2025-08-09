FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY chainlit_app.py .

EXPOSE 8000


# Run as non-root user
USER nobody

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app && chown -R appuser:appuser /app
USER appuser

CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]