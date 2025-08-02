# SutazAI Knowledge Graph Builder

## Overview

The Knowledge Graph Builder is a comprehensive system for constructing, managing, and querying knowledge representations of the SutazAI codebase hygiene standards and rules. It creates sophisticated semantic networks that enable advanced reasoning, compliance validation, and automated enforcement of coding standards.

## Features

### Core Capabilities
- **Entity Extraction**: Automatically extracts entities, relationships, and attributes from documentation, code, and configurations
- **Semantic Network Construction**: Builds comprehensive knowledge graphs representing standards, rules, agents, and their relationships
- **Graph-Based Reasoning**: Implements multi-hop reasoning algorithms for complex compliance validation
- **Compliance Validation**: Real-time validation of codebase against established hygiene standards
- **Automated Reporting**: Generates comprehensive compliance reports and recommendations

### Knowledge Domains Covered
1. **Professional Standards** (No fantasy elements, DRY principles, etc.)
2. **Technical Standards** (Docker, testing, security requirements)
3. **Process Standards** (Deployment, monitoring, CI/CD)
4. **Team Standards** (Documentation, code reviews, collaboration)
5. **AI/ML Specific Standards** (Model versioning, experiment tracking)

### Enforcement Hierarchy
- **🚨 BLOCKING**: Critical violations that prevent deployment
- **⚠️ WARNING**: Issues requiring review and approval
- **📋 GUIDANCE**: Best practices and recommendations

## Architecture

### Knowledge Graph Schema

```
Standards
├── Professional Standards
│   ├── No Fantasy Elements (BLOCKING)
│   ├── Existing Functionality Preservation (BLOCKING)
│   ├── Professional Project Treatment (WARNING)
│   └── Clear Documentation (GUIDANCE)
├── Technical Standards
│   ├── Docker Excellence (WARNING)
│   ├── Code Management (WARNING)
│   ├── Security Requirements (BLOCKING)
│   └── Testing Standards (WARNING)
├── Process Standards
│   ├── Universal Deployment (BLOCKING)
│   ├── Self-Healing Architecture (WARNING)
│   ├── Chaos Engineering (GUIDANCE)
│   └── System Health Monitoring (WARNING)
└── Team Standards
    ├── Version Control (WARNING)
    ├── Cleanup Procedures (WARNING)
    └── Agent Collaboration (GUIDANCE)
```

### Entity Types
- **Standards**: Core hygiene rules and requirements
- **Agents**: AI agents in the SutazAI system
- **Processes**: Workflows and procedures
- **Tools**: Development and deployment tools
- **Metrics**: Compliance and performance measurements
- **Violations**: Non-compliance instances

### Relationship Types
- **enforces**: Standard enforces a requirement
- **depends_on**: Entity depends on another
- **validates**: Process validates compliance
- **implements**: Agent implements a standard
- **monitors**: System monitors compliance
- **triggers**: Event triggers action
- **requires**: Prerequisite relationship
- **blocks**: Blocking relationship
- **warns**: Warning relationship

## Implementation

The knowledge graph is implemented using:
- **Neo4j**: Graph database backend
- **Python**: Core processing logic
- **NetworkX**: Graph algorithms and analysis
- **spaCy**: Natural language processing
- **PyTorch Geometric**: Graph neural networks
- **D3.js**: Interactive visualizations
- **FastAPI**: REST API endpoints

## Usage

### Building Knowledge Graph
```bash
# Build from documentation
python -m knowledge_graph_builder build --source /opt/sutazaiapp/CLAUDE.md

# Build from Docker configs
python -m knowledge_graph_builder build --source /opt/sutazaiapp/docker-compose*.yml

# Build comprehensive graph
python -m knowledge_graph_builder build --comprehensive
```

### Querying Standards
```bash
# Check compliance for specific standard
python -m knowledge_graph_builder query --standard "no_fantasy_elements"

# Validate agent compliance
python -m knowledge_graph_builder validate --agent "knowledge-graph-builder"

# Generate compliance report
python -m knowledge_graph_builder report --format json
```

### Visualization
```bash
# Generate interactive visualization
python -m knowledge_graph_builder visualize --output web

# Export graph formats
python -m knowledge_graph_builder export --format graphml
```

## Integration

The Knowledge Graph Builder integrates with:
- **All SutazAI Agents**: Provides knowledge representation
- **Health Check System**: Compliance validation
- **Docker Infrastructure**: Container standards enforcement
- **CI/CD Pipelines**: Automated compliance checking
- **Monitoring Systems**: Real-time standards compliance

## Configuration

See `knowledge_graph_config.yaml` for detailed configuration options including:
- Entity extraction parameters
- Reasoning engine settings
- Compliance thresholds
- Visualization preferences
- API endpoints and authentication

## Development

### Prerequisites
- Python 3.11+
- Neo4j 5.x
- Docker and Docker Compose
- Node.js (for visualization components)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start Neo4j database
docker-compose up neo4j -d

# Initialize knowledge graph
python -m knowledge_graph_builder init
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run compliance validation tests
python -m knowledge_graph_builder test
```

## API Reference

### REST Endpoints
- `GET /api/standards` - List all standards
- `GET /api/agents` - List all agents
- `POST /api/validate` - Validate compliance
- `GET /api/reports` - Generate reports
- `POST /api/query` - Execute graph queries
- `GET /api/visualize` - Get visualization data

### Graph Queries
```cypher
// Find all blocking standards
MATCH (s:Standard {enforcement_level: "BLOCKING"}) RETURN s

// Find agents not complying with standards
MATCH (a:Agent)-[:VIOLATES]->(s:Standard) RETURN a, s

// Find dependency chains
MATCH (a:Agent)-[:DEPENDS_ON*]->(dep) RETURN a, dep
```

## Monitoring and Alerts

The system provides:
- Real-time compliance monitoring
- Automated violation detection
- Integration with existing alerting systems
- Performance metrics and dashboards
- Trend analysis and reporting

## Contributing

Please follow the established hygiene standards when contributing:
1. Read `/opt/sutazaiapp/CLAUDE.md` thoroughly
2. Ensure no fantasy elements are introduced
3. Maintain existing functionality
4. Add comprehensive tests
5. Update documentation
6. Validate compliance before submission

## License

This project is part of the SutazAI automation system and follows the same licensing terms.