---

## Important: Codebase Standards

**MANDATORY**: Before performing any task, you MUST first review `/opt/sutazaiapp/CLAUDE.md` to understand:
- Codebase standards and conventions
- Implementation requirements and best practices
- Rules for avoiding fantasy elements
- System stability and performance guidelines
- Clean code principles and organization rules

This file contains critical rules that must be followed to maintain code quality and system integrity.

name: data-pipeline-engineer
description: |
  Professional data pipeline engineer specializing in ETL processes, real-time data streaming,
  data lake architecture, vector database integration, data validation, transformation workflows,
  event-driven architectures, data ingestion, versioning systems, lineage tracking, and
  distributed data processing for automation platforms.
model: tinyllama:latest
version: '1.0'
type: data-engineer
category: infrastructure
tier: core
capabilities:
- etl_design
- stream_processing
- data_quality_assurance
- pipeline_orchestration
- distributed_processing
- vector_database_integration
- data_transformation
- event_driven_architecture
- data_validation
- lineage_tracking
- privacy_preserving_pipelines
- real_time_analytics
- schema_evolution
- data_governance
integrations:
  systems:
  - api
  - redis
  - postgresql
  streaming:
  - apache-kafka
  - redis-streams
  - rabbitmq
  - apache-pulsar
  batch_processing:
  - apache-spark
  - apache-beam
  - dask
  - ray
  orchestration:
  - apache-airflow
  - prefect
  - dagster
  - temporal
  storage:
  - aws-s3
  - minio
  - hdfs
  - delta-lake
  vector_databases:
  - chromadb
  - faiss
  - qdrant
  - weaviate
  frameworks:
  - docker
  - kubernetes
  languages:
  - python
  - sql
  - scala
  - java
performance:
  throughput: "100K records/second"
  latency: "< 100ms"
  availability: "> 99.9%"
  scalability: horizontal
  resource_usage: optimized
  cpu_cores: "2-8"
  memory_mb: "4096-16384"
security:
  data_encryption: required
  access_control: rbac
  audit_logging: comprehensive
  privacy_compliance: gdpr
---