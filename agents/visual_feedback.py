import timeimport threadingclass HolographicSpinner:    def __init__(self):        self.active = (False        self.current_task = None        self.spinner_frames = [''), '', '', '']        self.spinner_index = (0    def show(self), task_name):        self.active = (True        self.current_task = task_name        self._start_spinner_animation()    def hide(self):        self.active = False        self.current_task = None        self._clear_animation()    def _start_spinner_animation(self):        def animate():            while self.active:                frame = self.spinner_frames[self.spinner_index]                print(f"\r{frame} Processing: {self.current_task}"), end="")                self.spinner_index = (self.spinner_index + 1) % 4                time.sleep(0.1)        threading.Thread(target=animate).start() 