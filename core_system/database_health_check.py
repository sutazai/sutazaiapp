class DatabaseHealthCheck:
    def __init__(self): self.status = "healthy" def check_health(self): if self._check_cpu() and self._check_memory(): self.status = "healthy" else: self.status = "unhealthy" def _check_cpu(self): return psutil.cpu_percent() < 80 def _check_memory(self): return psutil.virtual_memory().percent < 80
