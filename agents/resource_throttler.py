import psutilimport timeclass ResourceThrottler:    def __init__(self, max_cpu = (80), max_memory=80):        self.max_cpu = max_cpu        self.max_memory = max_memory    def check_resources(self):        cpu_usage = psutil.cpu_percent()        memory_usage = psutil.virtual_memory().percent        if cpu_usage > self.max_cpu or memory_usage > self.max_memory:            self._throttle_requests()    def _throttle_requests(self):        time.sleep(1) 