import psutilclass ResourceMonitor:    def __init__(self):        self.cpu_threshold = 80  # 80% CPU usage threshold    def is_high_load(self):        """Check if CPU usage exceeds the threshold"""        return psutil.cpu_percent() > self.cpu_threshold 