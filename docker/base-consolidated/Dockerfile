FROM sutazai/python-agent-base:latest
# Service-specific configurations for base-consolidated
COPY requirements.txt* ./
RUN pip install -r requirements.txt || true
COPY . .
