# Reliability & Performance

- SLOs: Availability 99.9%, p95 latency < 2s, TTFT < 500ms
- SLIs: request_success_rate, latency_p95, error_rate, queue_depth
- Backpressure: bounded queues, shed load at gateway
- Rate Limits: per IP and per token; burst and sustained
- Retries: exponential backoff, idempotent only
- Capacity: 1000 concurrent users (assumption), 10 concurrent agents

Footnotes: [`REAL_FEATURES_AND_USERSTORIES.md` 300–312]