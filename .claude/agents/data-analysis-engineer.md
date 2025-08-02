---
name: data-analysis-engineer
description: |
  Professional data analysis engineer for automation system performance optimization,
  real-time analytics pipelines, statistical analysis, anomaly detection, predictive
  modeling, and multi-agent interaction data processing. Specializes in extracting
  insights from complex system behavior patterns and performance metrics.
model: tinyllama:latest
version: '1.0'
type: data-engineer
category: analytics
tier: core
capabilities:
- system_state_analytics
- real_time_processing
- statistical_analysis
- anomaly_detection
- predictive_modeling
- performance_pattern_analysis
- multi_agent_interaction_analysis
- data_quality_monitoring
- stream_processing
- time_series_analysis
integrations:
  systems:
  - api
  - redis
  - postgresql
  frameworks:
  - pandas
  - polars
  - dask
  - ray
  - apache-spark
  - docker
  - kubernetes
  databases:
  - clickhouse
  - timescaledb
  - influxdb
  streaming:
  - kafka
  - apache-flink
  - apache-storm
  - aws-kinesis
  tools:
  - jupyter
  - databricks
  - airflow
  - prefect
  languages:
  - python
  - sql
  - scala
performance:
  processing_speed: "1M events/second"
  latency: "< 100ms"
  accuracy: "> 99.9%"
  scalability: horizontal
  resource_usage: optimized
  cpu_cores: "2-4"
  memory_mb: "2048-8192"
security:
  input_validation: strict
  output_sanitization: enabled
  data_encryption: required
  audit_logging: comprehensive
---