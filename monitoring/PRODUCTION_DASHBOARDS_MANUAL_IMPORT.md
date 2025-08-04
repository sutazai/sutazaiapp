# SutazAI Production Monitoring Dashboards - Manual Import Guide

## Overview

This guide provides instructions for manually importing the 8 production-grade monitoring dashboards into Grafana at http://localhost:10050.

## Authentication

- **URL**: http://localhost:10050
- **Username**: admin
- **Password**: admin (or check the environment variable GF_SECURITY_ADMIN_PASSWORD)

## Dashboard Files Location

All dashboard JSON files are located in: `/opt/sutazaiapp/monitoring/grafana/dashboards/`

## Manual Import Process

### Step 1: Access Grafana Web Interface
1. Open your web browser
2. Navigate to http://localhost:10050
3. Login with admin credentials

### Step 2: Create Folders
Create the following folders in Grafana (Settings > Folders):
- Executive
- Operations  
- Developer
- Security
- Business
- Cost
- UX
- Capacity

### Step 3: Import Dashboards

Import each dashboard JSON file into its respective folder:

#### 1. Executive Dashboard
- **File**: `/opt/sutazaiapp/monitoring/grafana/dashboards/executive/executive-overview.json`
- **Folder**: Executive
- **Audience**: C-Suite, Leadership Team
- **Refresh**: 30 seconds
- **Focus**: High-level KPIs, system availability, business metrics

#### 2. Operations Dashboard (NOC)
- **File**: `/opt/sutazaiapp/monitoring/grafana/dashboards/operations/operations-overview.json`
- **Folder**: Operations
- **Audience**: Network Operations Center, DevOps Team
- **Refresh**: 10 seconds
- **Focus**: System health, infrastructure monitoring, real-time alerts

#### 3. Developer Dashboard
- **File**: `/opt/sutazaiapp/monitoring/grafana/dashboards/developer/developer-overview.json`
- **Folder**: Developer
- **Audience**: Development Team, Software Engineers
- **Refresh**: 5 seconds
- **Focus**: Application performance, debugging metrics, code-level insights

#### 4. Security Dashboard (SOC)
- **File**: `/opt/sutazaiapp/monitoring/grafana/dashboards/security/security-overview.json`
- **Folder**: Security
- **Audience**: Security Operations Center, InfoSec Team
- **Refresh**: 5 seconds
- **Focus**: Security events, threat detection, access monitoring

#### 5. Business Metrics Dashboard
- **File**: `/opt/sutazaiapp/monitoring/grafana/dashboards/business/business-metrics.json`
- **Folder**: Business
- **Audience**: Product Managers, Business Analysts
- **Refresh**: 30 seconds
- **Focus**: Business KPIs, user engagement, service quality

#### 6. Cost Optimization Dashboard
- **File**: `/opt/sutazaiapp/monitoring/grafana/dashboards/cost/cost-optimization.json`
- **Folder**: Cost
- **Audience**: Finance Team, Resource Managers
- **Refresh**: 30 seconds
- **Focus**: Resource utilization, cost efficiency, optimization opportunities

#### 7. User Experience Dashboard
- **File**: `/opt/sutazaiapp/monitoring/grafana/dashboards/ux/user-experience.json`
- **Folder**: UX
- **Audience**: UX Team, Customer Success
- **Refresh**: 5 seconds
- **Focus**: User-facing performance, quality metrics

#### 8. Capacity Planning Dashboard
- **File**: `/opt/sutazaiapp/monitoring/grafana/dashboards/capacity/capacity-planning.json`
- **Folder**: Capacity
- **Audience**: Infrastructure Team, Architects
- **Refresh**: 1 minute
- **Focus**: Resource forecasting, scaling requirements, growth planning

## Import Instructions

For each dashboard:
1. In Grafana, go to "+" (Create) > Import
2. Click "Upload JSON file" or copy-paste the JSON content
3. Select the appropriate folder from the dropdown
4. Click "Import"
5. Verify the dashboard loads correctly with data

## Alert Rules

The production alert rules have been created in:
`/opt/sutazaiapp/monitoring/prometheus/production_alerts.yml`

To activate these alerts:
1. Copy the file to your Prometheus rules directory
2. Restart Prometheus to load the new rules
3. Verify alerts are loaded in Prometheus web interface

## Dashboard Features

### Auto-Refresh Settings
All dashboards have auto-refresh enabled with audience-appropriate intervals:
- **Real-time monitoring** (Security, Developer, Operations): 5-10 seconds
- **Business metrics** (Executive, Business, Cost): 30 seconds  
- **Planning** (Capacity): 1 minute

### Cross-Dashboard Navigation
Each dashboard includes navigation links to related dashboards for seamless workflow transitions.

### Responsive Design
All dashboards are optimized for various screen sizes and display resolutions.

### Alert Integration
Dashboards show active alert counts and statuses from Prometheus AlertManager.

## Key Metrics by Dashboard

### Executive Overview
- System availability percentage
- Active alerts count
- API request rate
- Response time P95
- API success rate
- Active services count
- Service health distribution
- Resource utilization trends

### Operations (NOC)
- Active alerts
- Services online count
- CPU/Memory/Disk usage
- Response time
- System resource utilization
- API traffic (requests/errors)
- Network I/O
- Disk I/O

### Developer Dashboard
- API response time percentiles (P50, P95, P99)
- HTTP response codes distribution
- Container memory/CPU usage
- Database metrics (connections, transactions)
- Redis cache metrics
- API methods distribution
- AI agent status

### Security (SOC)
- Active security alerts
- Failed requests rate
- Authentication failures
- Access violations
- Request rate (DDoS detection)
- Security services status
- Security-related HTTP errors
- Top request sources
- Error code distribution
- Network traffic anomalies

### Business Metrics
- Daily active API calls
- Active AI agents
- Service quality (SLA)
- User experience score
- API usage growth
- AI agent utilization
- Feature usage distribution
- Business SLAs
- Volume metrics
- Data activity

### Cost Optimization
- CPU/Memory/Storage efficiency gauges
- Resource pool size
- Container resource usage
- Memory/CPU cost distribution
- Network cost analysis
- Cost optimization opportunities
- Agent resource allocation

### User Experience
- P50/P95 latency gauges
- API success rate
- System availability
- Response time distribution
- Request volume and errors
- Response code distribution
- Backend performance impact
- Service health impact
- Detailed performance by API method

### Capacity Planning
- Current CPU/Memory/Storage capacity
- Storage forecast
- Resource utilization trends
- Load growth patterns
- Service scaling requirements
- Database capacity planning
- Resource allocation by service
- Network capacity planning
- Storage I/O capacity

## Troubleshooting

### Common Issues
1. **No data showing**: Verify Prometheus is running and scraping targets
2. **Authentication errors**: Check Grafana admin credentials
3. **Dashboard import errors**: Validate JSON syntax
4. **Missing metrics**: Ensure all exporters are running

### Data Sources
Ensure the following data sources are configured in Grafana:
- **Prometheus**: http://prometheus:9090 (or appropriate URL)
- **Loki** (if using): http://loki:3100
- **Jaeger** (if using): http://jaeger:16686

## Maintenance

- Review and update dashboards quarterly
- Monitor alert rule effectiveness
- Update thresholds based on system growth
- Validate data source connections regularly

## Support

For issues or enhancements, contact the Platform Team or create an issue in the SutazAI repository.

---

**Generated**: $(date)
**Version**: 1.0
**Files**: 8 production dashboards + alert rules
**Location**: /opt/sutazaiapp/monitoring/grafana/dashboards/