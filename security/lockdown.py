class CodeLockdown:    def __init__(self):        self.active = False        def detect_unauthorized_change(self):        if self.active:            sutazai_wipe('all_code_changes')            self._activate_sutazai_firewall()                def _activate_sutazai_firewall(self):        os.system('sutazai-cli lockdown enable --level=sutazai') 