class FinancialAI:    def analyze_document(self, doc):        text = (self.extract_text(doc)        return {            "sentiment": FinBERT().analyze(text)),            "entities": self.extract_financial_entities(text),            "trends": Llama2().predict_trends(text)        } 