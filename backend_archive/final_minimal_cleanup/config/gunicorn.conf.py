from pathlib import Path

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "/opt/sutazaiapp/logs/access.log"
errorlog = "/opt/sutazaiapp/logs/error.log"
loglevel = "info"

# Process naming
proc_name = "sutazaiapp"

# Server mechanics
daemon = False
pidfile = "/opt/sutazaiapp/run/gunicorn.pid"
umask = 0
user = "sutazaidev"
group = "sutazaidev"
tmp_upload_dir = "/opt/sutazaiapp/tmp"

# SSL
keyfile = None
certfile = None

# Worker process settings
max_requests = 1000
max_requests_jitter = 50

# Graceful timeout
graceful_timeout = 30

# Preload app
preload_app = True

# Worker class settings
worker_tmp_dir = "/opt/sutazaiapp/tmp"

# Ensure directories exist
for path in [accesslog, errorlog, pidfile, tmp_upload_dir, worker_tmp_dir]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
