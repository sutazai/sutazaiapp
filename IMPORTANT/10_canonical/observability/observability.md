# Observability

- Logs: structured JSON; correlation IDs; levels per component
- Metrics: Prometheus endpoints `/metrics`; golden signals
- Traces: optional OpenTelemetry; spans across BE->Ollama->DB
- Dashboards: Grafana—API, Agents, DB, Ollama
- Alerts: AlertManager with on-call routing; runbooks per alert
