# ISSUE-0016: Monitoring Dashboards Unconfigured

**Impacted Components:** Grafana, Prometheus, Observability stack
**Context:** Full monitoring stack deployed (Prometheus, Grafana, Loki, AlertManager) but no dashboards configured. Metrics collected but not visualized.

**Options:**
- **A: Import Community Dashboards + Custom SutazAI Metrics** (Recommended)
  - Pros: Quick start with proven dashboards, customizable
  - Cons: May include unnecessary metrics
  
- **B: Build Custom Dashboards from Scratch**
  - Pros: Exactly what we need, no bloat
  - Cons: Time-consuming, reinventing wheels
  
- **C: Use Grafana Cloud Pre-built**
  - Pros: Professional dashboards, managed service
  - Cons: External dependency, potential data privacy concerns

**Recommendation:** A - Import standard dashboards then customize

**Dashboard Requirements:**
1. **System Overview** - Container health, resource usage
2. **API Performance** - Request rates, latencies, errors
3. **Agent Metrics** - Task processing, queue depths, success rates
4. **Database Performance** - Query times, connection pools
5. **Business Metrics** - User activity, feature usage

**Implementation Steps:**
1. Import Docker/container dashboards (ID: 893, 13112)
2. Import PostgreSQL dashboard (ID: 9628)
3. Import FastAPI/Python dashboards (ID: 12900)
4. Create custom SutazAI agent performance dashboard
5. Configure alerts for P0 component failures

**Consequences:** 
- Requires defining SLIs/SLOs for alerting
- Need to instrument application code for custom metrics
- Grafana persistent storage for dashboard definitions
- Team training on dashboard usage

**Dependencies:** None (monitoring stack already running)

**Acceptance Criteria:**
```gherkin
Given Grafana access
When user logs in
Then 5 core dashboards are visible with live data

Given a service failure
When metrics cross threshold
Then alert fires and shows on dashboard

Given custom metrics pushed
When dashboard refreshes
Then SutazAI-specific panels update
```

**Metrics to Track:**
| Metric | Type | Dashboard | Alert Threshold |
|--------|------|-----------|-----------------|
| Container restarts | Counter | System | >3 in 5min |
| API response time | Histogram | API | p95 > 1s |
| Agent task queue | Gauge | Agents | >100 pending |
| DB connections | Gauge | Database | >80% pool |
| Active users | Gauge | Business | N/A |
| Memory usage | Gauge | System | >90% |

**Evidence:** 
[source] http://localhost:10201 (Grafana empty)
[source] /opt/sutazaiapp/docker-compose.yml#L800-L850
[source] /opt/sutazaiapp/IMPORTANT/10_canonical/observability/observability.md#L1-L100