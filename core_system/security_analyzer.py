class CodeGuard:
    def analyze_upload(self, file): if file.type == "code": return self.run_semgrep_analysis(file.content) return {"status": "skipped"}
