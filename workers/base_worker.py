import timeimport threadingfrom abc import ABC, abstractmethodclass SutazAiWorker(ABC):    def __init__(self, name, interval=60):        self.name = name        self.interval = interval        self._running = False        self._thread = None    @abstractmethod    def execute(self):        pass    def start(self):        if self._running:            return                self._running = True        self._thread = threading.Thread(target=self._run)        self._thread.start()    def stop(self):        self._running = False        if self._thread:            self._thread.join()    def _run(self):        while self._running:            try:                self.execute()            except Exception as e:                print(f"Error in worker {self.name}: {str(e)}")            time.sleep(self.interval)    def __del__(self):        self.stop() 