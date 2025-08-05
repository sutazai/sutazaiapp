# SutazAI Data Governance Framework

## Overview

The SutazAI Data Governance Framework is a comprehensive, enterprise-grade system that manages the complete lifecycle of data across the distributed AI architecture. It provides automated data classification, compliance monitoring, quality assessment, lineage tracking, and lifecycle management for all 69 AI agents and their data flows.

## Key Features

### 🔍 **Automated Data Classification**
- **Multi-dimensional Classification**: Automatically classifies data as Public, Internal, Confidential, or Restricted
- **Regulatory Scope Detection**: Identifies data subject to GDPR, CCPA, HIPAA, SOX regulations
- **Pattern Recognition**: Uses advanced regex patterns and metadata analysis for accurate classification
- **Confidence Scoring**: Provides confidence levels for all classification decisions

### 📊 **Comprehensive Data Lifecycle Management** 
- **Automated Retention Policies**: Configurable retention rules based on data classification and regulations
- **Tiered Storage**: Hot/warm/cold storage tier management with automatic transitions
- **Legal Hold Support**: Prevents deletion of data under legal hold with full audit trail
- **Automated Archival**: Background processes for data archival and cleanup

### ⚖️ **Multi-Regulation Compliance**
- **GDPR Compliance**: Right to deletion, data minimization, consent management
- **CCPA Compliance**: Consumer rights, data disclosure, deletion capabilities
- **HIPAA Compliance**: PHI protection, access controls, audit requirements
- **SOX Compliance**: Financial data retention, immutable audit trails

### 🔍 **Advanced Data Quality Monitoring**
- **Multi-Dimensional Quality**: Completeness, accuracy, consistency, timeliness, validity, uniqueness
- **Real-time Assessment**: Continuous quality monitoring with automated alerts
- **Quality Rules Engine**: Configurable rules for different data types and domains
- **Issue Tracking**: Comprehensive quality issue management with resolution workflows

### 🔗 **Complete Data Lineage Tracking**
- **End-to-End Lineage**: Full data flow tracking from source to consumption
- **Impact Analysis**: Understand downstream effects of data changes
- **Neo4j Integration**: Graph database for complex lineage relationships
- **Time-Travel Queries**: Historical lineage reconstruction

### 📚 **Intelligent Data Catalog**
- **Auto-Discovery**: Automatic detection and cataloging of data assets
- **Rich Metadata**: Comprehensive asset metadata with business context
- **Powerful Search**: Full-text search with faceted filtering
- **Asset Recommendations**: Personalized data asset recommendations

### 🔄 **Comprehensive Data Versioning**
- **Version Control**: Git-like versioning for data assets
- **Branching & Merging**: Support for parallel data development workflows  
- **Time Travel**: Query data as it existed at any point in time
- **Change Tracking**: Detailed diff capabilities between versions

### 📊 **Governance Dashboards**
- **Executive Overview**: High-level governance metrics and KPIs
- **Compliance Dashboard**: Regulation-specific compliance status
- **Quality Dashboard**: Data quality trends and issues
- **Operational Dashboard**: System health and performance metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Governance Framework                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Data Classifier │  │ Lifecycle Mgr   │  │ Compliance Mgr  │  │
│  │                 │  │                 │  │                 │  │
│  │ • Pattern Match │  │ • Retention     │  │ • GDPR/CCPA     │  │
│  │ • ML Models     │  │ • Archival      │  │ • HIPAA/SOX     │  │
│  │ • Rule Engine   │  │ • Deletion      │  │ • Violations    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Quality Monitor │  │ Lineage Tracker │  │  Data Catalog   │  │
│  │                 │  │                 │  │                 │  │
│  │ • Quality Rules │  │ • Neo4j Graph   │  │ • Discovery     │  │
│  │ • Assessments   │  │ • Impact Anal.  │  │ • Search        │  │
│  │ • Issue Mgmt    │  │ • Time Travel   │  │ • Metadata      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Data Versioning │  │  Audit Logger   │  │  Governance API │  │
│  │                 │  │                 │  │                 │  │
│  │ • Git-like Ver. │  │ • Tamper-proof  │  │ • REST APIs     │  │
│  │ • Branching     │  │ • Compliance    │  │ • Dashboards    │  │
│  │ • Time Travel   │  │ • Security      │  │ • Integration   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Initialize the Governance Framework

```python
from data_governance import DataGovernanceFramework

# Configure the framework
config = {
    'lifecycle': {'batch_size': 100, 'execution_interval_minutes': 60},
    'audit': {'retention_days': 2555, 'batch_size': 1000},
    'compliance': {'enabled_regulations': ['gdpr', 'ccpa', 'hipaa']},
    'quality': {'default_quality_threshold': 0.8}
}

# Initialize framework
framework = DataGovernanceFramework(config)
await framework.initialize()
```

### 2. Process Data Through Governance Pipeline

```python
# Process new data
results = await framework.process_new_data(
    data_id="user_data_001",
    content='{"email": "user@example.com", "age": 25}',
    metadata={
        "source": "user_registration",
        "table_name": "users",
        "created_at": "2024-01-01T00:00:00Z"
    }
)

print(f"Classification: {results['classification']['level']}")
print(f"Quality Score: {results['quality_score']}")
print(f"Compliance Status: {results['compliance_status']}")
```

### 3. Search Data Catalog

```python
from data_governance import SearchQuery, DataClassification

# Search for confidential data
query = SearchQuery(
    text="customer email",
    classifications=[DataClassification.CONFIDENTIAL],
    limit=50
)

results = await framework.data_catalog.search_assets(query)
print(f"Found {results.total_count} assets")
```

### 4. Track Data Lineage

```python
# Get upstream lineage
upstream = await framework.lineage_tracker.get_upstream_lineage("user_data_001")
print(f"Found {len(upstream)} upstream dependencies")

# Analyze impact of changes
impact = await framework.lineage_tracker.analyze_data_impact(
    "user_data_001", "schema_change"
)
print(f"Risk Level: {impact['risk_level']}")
```

## API Endpoints

### Governance Dashboard
- `GET /api/v1/governance/dashboard/overview` - Complete governance overview
- `GET /api/v1/governance/dashboard/compliance` - Compliance dashboard
- `GET /api/v1/governance/dashboard/quality` - Data quality dashboard

### Data Processing
- `POST /api/v1/governance/process-data` - Process data through governance pipeline
- `GET /api/v1/governance/data/{data_id}/classification` - Get classification results

### Audit & Compliance
- `GET /api/v1/governance/audit/events` - Query audit events
- `GET /api/v1/governance/compliance/reports/{regulation}` - Generate compliance reports
- `POST /api/v1/governance/compliance/violations/{id}/resolve` - Resolve violations

### Data Catalog
- `GET /api/v1/governance/catalog/search` - Search data catalog
- `GET /api/v1/governance/catalog/assets/{id}` - Get asset details

### Data Lineage
- `GET /api/v1/governance/lineage/{id}/upstream` - Get upstream lineage
- `GET /api/v1/governance/lineage/{id}/downstream` - Get downstream lineage
- `GET /api/v1/governance/lineage/{id}/impact` - Get impact analysis

### Data Quality
- `POST /api/v1/governance/quality/assess` - Assess data quality
- `GET /api/v1/governance/quality/issues` - Get quality issues

### Data Versioning
- `POST /api/v1/governance/versioning/{id}/versions` - Create new version
- `GET /api/v1/governance/versioning/{id}/history` - Get version history

## Configuration

### Environment Variables

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai_password
POSTGRES_DB=sutazai

# Neo4j Configuration (for lineage)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Storage Configuration
DATA_STORAGE_PATH=/data
AUDIT_LOG_PATH=/data/audit
GOVERNANCE_CONFIG_PATH=/config/governance

# Compliance Configuration
GDPR_ENABLED=true
CCPA_ENABLED=true
HIPAA_ENABLED=false
SOX_ENABLED=false

# Quality Thresholds
DEFAULT_QUALITY_THRESHOLD=0.8
CRITICAL_QUALITY_THRESHOLD=0.5
```

### Framework Configuration

```python
config = {
    # Lifecycle Management
    'lifecycle': {
        'batch_size': 100,
        'execution_interval_minutes': 60,
        'require_approval_for_deletion': True,
        'max_deletions_per_batch': 10,
        'storage': {
            'hot_path': '/data/hot',
            'warm_path': '/data/warm', 
            'cold_path': '/data/cold',
            'archive_path': '/data/archive'
        }
    },
    
    # Audit Logging
    'audit': {
        'audit_log_path': '/data/audit',
        'retention_days': 2555,  # 7 years
        'batch_size': 1000,
        'flush_interval_seconds': 60,
        'enable_encryption': True,
        'enable_compression': True
    },
    
    # Compliance Management
    'compliance': {
        'enabled_regulations': ['gdpr', 'ccpa', 'hipaa', 'sox'],
        'batch_size': 100,
        'audit_interval_hours': 24
    },
    
    # Data Lineage
    'lineage': {
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'password',
            'database': 'lineage'
        },
        'max_lineage_depth': 20,
        'batch_size': 1000,
        'cache_ttl_hours': 24
    },
    
    # Quality Monitoring
    'quality': {
        'batch_size': 1000,
        'assessment_interval_hours': 6,
        'max_sample_failures': 10,
        'default_quality_threshold': 0.8,
        'critical_quality_threshold': 0.5
    },
    
    # Data Catalog
    'catalog': {
        'auto_discovery_enabled': True,
        'quality_threshold': 0.7,
        'max_sample_records': 5
    },
    
    # Data Versioning
    'versioning': {
        'default_strategy': 'hybrid',
        'max_versions_per_asset': 100,
        'auto_cleanup_enabled': True,
        'retention_days': 365,
        'storage': {
            'compression_enabled': True,
            'encryption_enabled': False
        }
    }
}
```

## Integration with SutazAI

### AI Agent Data Integration

```python
# Track AI agent communication data
await framework.process_new_data(
    data_id=f"agent_{agent_id}_{timestamp}",
    content=agent_message,
    metadata={
        "source": "ai_agent",
        "agent_id": agent_id,
        "data_type": "communication",
        "classification_hint": "internal"
    }
)
```

### Model Lifecycle Tracking

```python
# Track AI model versions
model_id = await framework.lineage_tracker.track_ai_model(
    model_name="sentiment_analyzer",
    model_version="v2.1.0", 
    training_data_ids=["training_set_001", "validation_set_001"]
)

# Version model artifacts
version = await framework.versioning.create_version(
    asset_id=model_id,
    data_content=model_metadata,
    change_type="update",
    description="Updated sentiment model with new training data"
)
```

### Knowledge Graph Integration

```python
# Register knowledge graph entities
node_id = await framework.lineage_tracker.track_database_table(
    database="sutazai_knowledge",
    schema="entities", 
    table="concepts",
    source_system="neo4j"
)

# Track data flows
await framework.lineage_tracker.record_data_flow(
    source_id="raw_documents_001",
    target_id=node_id,
    event_type="data_transformation",
    process_name="entity_extraction",
    transformation_logic="NLP entity extraction and knowledge graph population"
)
```

## Monitoring and Alerting

### Quality Alerts
- Automatic alerts when data quality scores drop below thresholds
- Email/Slack notifications for critical quality issues
- Quality trend monitoring and reporting

### Compliance Alerts
- Real-time violation detection and notification
- Regulatory deadline tracking and reminders
- Automated compliance report generation

### System Health Monitoring
- Background process health checks
- Storage usage monitoring and alerts
- Performance metrics and optimization recommendations

## Best Practices

### 1. Data Classification
- Always provide context metadata for better classification accuracy
- Review and validate automated classifications for sensitive data
- Use custom classification rules for domain-specific data types

### 2. Lifecycle Management
- Set appropriate retention policies based on business and regulatory requirements
- Regularly review and update lifecycle policies
- Use legal holds judiciously and remove when no longer needed

### 3. Quality Management  
- Define quality rules specific to your data domains
- Implement data quality checks at ingestion points
- Address quality issues promptly to prevent downstream impact

### 4. Compliance
- Regularly audit compliance status across all regulations
- Implement privacy by design principles
- Maintain comprehensive audit trails for all data operations

### 5. Lineage Tracking
- Document all data transformations and processing steps
- Use consistent naming conventions for better lineage visualization
- Regularly validate lineage accuracy through impact analysis

## Security Considerations

### Data Protection
- All sensitive data is encrypted at rest and in transit
- Access controls based on data classification levels
- Audit logging for all data access and modifications

### Compliance Security
- Tamper-proof audit logs with integrity verification
- Role-based access to governance functions
- Secure storage of compliance artifacts and reports

### API Security
- OAuth2/JWT authentication for all API endpoints
- Rate limiting and request validation
- HTTPS enforcement for all communications

## Performance Optimization

### Scalability
- Horizontal scaling support for all components
- Asynchronous processing for heavy governance operations
- Caching strategies for frequently accessed metadata

### Storage Optimization
- Tiered storage with automatic data lifecycle transitions
- Compression and deduplication for audit logs
- Efficient indexing for fast search and retrieval

### Processing Optimization
- Batch processing for bulk operations
- Parallel processing where applicable
- Background tasks for non-critical operations

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check component memory usage
GET /api/v1/governance/admin/statistics

# Adjust batch sizes in configuration
config['quality']['batch_size'] = 500  # Reduce from 1000
```

#### Slow Lineage Queries
```bash
# Check Neo4j performance
GET /api/v1/governance/lineage/statistics

# Optimize Neo4j indexes
CREATE INDEX ON :LineageNode(id)
CREATE INDEX ON :LineageNode(type)
```

#### Quality Assessment Delays
```bash
# Check quality monitor statistics
GET /api/v1/governance/quality/statistics

# Adjust assessment interval
config['quality']['assessment_interval_hours'] = 12  # Increase from 6
```

### Monitoring Commands

```bash
# Check overall system health
curl -X GET "http://localhost:8000/api/v1/governance/health"

# Get comprehensive statistics
curl -X GET "http://localhost:8000/api/v1/governance/admin/statistics"

# Monitor audit events
curl -X GET "http://localhost:8000/api/v1/governance/audit/events?limit=100"

# Check compliance status
curl -X GET "http://localhost:8000/api/v1/governance/dashboard/compliance"
```

## Contributing

### Development Setup

1. **Clone and Install**
   ```bash
   git clone <repository>
   cd backend/data_governance
   pip install -r requirements.txt
   ```

2. **Setup Development Database**
   ```bash
   docker-compose up -d postgres neo4j redis
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v --cov=data_governance
   ```

4. **Code Quality**
   ```bash
   black data_governance/
   isort data_governance/
   flake8 data_governance/
   mypy data_governance/
   ```

### Adding New Features

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new functionality
3. Update documentation and API endpoints
4. Ensure backward compatibility
5. Add configuration options where appropriate

## License

This data governance framework is part of the SutazAI system and follows the same licensing terms as the main project.

## Support

For technical support and questions:
- Review the troubleshooting section above
- Check the API documentation for endpoint details
- Monitor system logs for error messages
- Use the admin statistics endpoints for system health insights

---

*This framework provides enterprise-grade data governance for the SutazAI distributed AI system, ensuring compliance, quality, and proper lifecycle management for all data assets across 69 AI agents and supporting infrastructure.*