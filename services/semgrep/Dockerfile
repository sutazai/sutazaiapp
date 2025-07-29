FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir semgrep fastapi uvicorn

COPY semgrep_service.py /app/

EXPOSE 8087

CMD ["uvicorn", "semgrep_service:app", "--host", "0.0.0.0", "--port", "8087"]