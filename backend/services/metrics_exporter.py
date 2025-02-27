import os
import time
from typing import Dict, Any

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram
from fastapi import FastAPI, Depends, HTTPException

from scripts.otp_override import OTPValidator

class SutazaiMetricsExporter:
    def __init__(self):
        # Document Parsing Metrics
        self.docs_parsed = Counter(
            'sutazai_documents_parsed_total', 
            'Total number of documents parsed'
        )
        self.doc_parse_duration = Histogram(
            'sutazai_document_parse_seconds', 
            'Time spent parsing documents'
        )
        
        # Code Generation Metrics
        self.code_generated = Counter(
            'sutazai_code_generated_total', 
            'Total lines of code generated'
        )
        self.generation_success_rate = Gauge(
            'sutazai_code_generation_success_rate', 
            'Percentage of successful code generations'
        )
        
        # Orchestrator Metrics
        self.self_improvement_attempts = Counter(
            'sutazai_self_improvement_attempts_total', 
            'Total self-improvement attempts'
        )
        self.improvement_success_rate = Gauge(
            'sutazai_improvement_success_rate', 
            'Percentage of successful self-improvements'
        )
    
    def record_document_parse(self, parse_time: float):
        self.docs_parsed.inc()
        self.doc_parse_duration.observe(parse_time)
    
    def record_code_generation(self, lines_generated: int, success: bool):
        self.code_generated.inc(lines_generated)
        self.generation_success_rate.set(1.0 if success else 0.0)
    
    def record_self_improvement(self, success: bool):
        self.self_improvement_attempts.inc()
        self.improvement_success_rate.set(1.0 if success else 0.0)

def create_metrics_app(metrics_exporter: SutazaiMetricsExporter):
    app = FastAPI()
    otp_validator = OTPValidator()

    @app.get("/metrics")
    def metrics(otp: str = Depends(otp_validator.validate_external_call)):
        return prometheus_client.generate_latest()

    return app

def main():
    metrics_exporter = SutazaiMetricsExporter()
    app = create_metrics_app(metrics_exporter)
    
    # Example usage
    metrics_exporter.record_document_parse(0.5)
    metrics_exporter.record_code_generation(100, True)
    metrics_exporter.record_self_improvement(True)

if __name__ == "__main__":
    main() 