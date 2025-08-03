# Ollama Integration Rollback Procedure

## Overview

This document defines comprehensive rollback procedures for the Ollama Integration deployment. The rollback strategy ensures system stability can be restored within 60 seconds of trigger activation, with full system recovery in under 3.5 minutes.

## Rollback Architecture

### Blue-Green Environment Setup

```
Production Environment (Blue - Current):
┌─────────────────────────────────────────┐
│ 131 Agents (Legacy BaseAgent)          │
│ ├── Stable, tested configuration       │
│ ├── Known performance baselines        │  
│ └── Immediate fallback capability      │
└─────────────────────────────────────────┘

Deployment Environment (Green - Target):
┌─────────────────────────────────────────┐
│ X Agents (BaseAgentV2)                 │
│ ├── Enhanced Ollama integration        │
│ ├── Under active deployment            │
│ └── Subject to rollback if needed      │
└─────────────────────────────────────────┘
```

## Automated Rollback Triggers

### Critical Triggers (Immediate Rollback)
1. **Memory Exhaustion**: System memory usage > 90%
2. **High Error Rate**: Error rate > 5% for any agent type
3. **Ollama Overload**: Connection failures > 10% in 60 seconds
4. **Health Check Failures**: > 2 consecutive failures per agent
5. **Circuit Breaker Cascade**: > 3 circuit breaker trips per minute
6. **System Unresponsiveness**: No heartbeat from > 10% of agents for 2 minutes

### Warning Triggers (Pause and Evaluate)
1. **Elevated Response Time**: > 5x baseline for any agent type
2. **Memory Pressure**: Memory usage > 75% sustained for 5 minutes
3. **Ollama Degradation**: Query success rate < 95%
4. **Task Failure Rate**: > 1% task completion failures
5. **Resource Contention**: CPU usage > 90% sustained for 3 minutes

## Rollback Decision Matrix

| Trigger Type | Severity | Action | Timeline | Approval Required |
|--------------|----------|--------|----------|-------------------|
| Memory > 90% | Critical | Immediate Rollback | < 60s | Automated |
| Error Rate > 5% | Critical | Immediate Rollback | < 60s | Automated |
| Health Failures | Critical | Immediate Rollback | < 60s | Automated |
| Response Time 5x | High | Pause & Evaluate | < 2m | Lead Engineer |
| Memory > 75% | Medium | Monitor & Prepare | < 5m | Team Decision |
| Ollama < 95% | Medium | Monitor & Prepare | < 5m | Team Decision |

## Rollback Procedures

### Phase 1: Immediate Stop (0-30 seconds)

#### Automated Actions
```bash
# Stop current deployment immediately
kubectl patch deployment ollama-integration -p '{"spec":{"paused":true}}'

# Prevent new agent migrations
export DEPLOYMENT_PAUSED=true

# Lock deployment scripts
touch /opt/sutazaiapp/deployment/.rollback-in-progress

# Alert deployment team
curl -X POST "$SLACK_WEBHOOK" -d "{\"text\":\"🚨 ROLLBACK INITIATED: Ollama Integration Deployment\"}"
```

#### Manual Verification
- [ ] Deployment process halted
- [ ] No new agents migrating to BaseAgentV2
- [ ] Current stable agents still operational
- [ ] Monitoring systems active and reporting

### Phase 2: Traffic Redirection (30-60 seconds)

#### Load Balancer Update
```bash
# Redirect all traffic to stable (blue) environment
/opt/sutazaiapp/scripts/rollback-ollama-integration.sh --redirect-traffic

# Update service discovery
consul-template -template="agent-discovery.tpl:agent-discovery.conf:nginx -s reload"

# Verify traffic redirection
curl -f http://load-balancer/health-check
```

#### Container Orchestration
```bash
# Scale down enhanced agents gradually
for agent in $(cat /tmp/deployed-agents.txt); do
    docker-compose -f docker-compose.ollama-optimized.yml scale $agent=0
    sleep 2
done

# Ensure legacy agents handle full load
docker-compose -f docker-compose.yml scale --verify-capacity
```

### Phase 3: Environment Swap (60-120 seconds)

#### Container Replacement
```bash
# Replace enhanced containers with legacy versions
/opt/sutazaiapp/scripts/rollback-ollama-integration.sh --swap-containers

# Restore original configurations
cp /opt/sutazaiapp/backup/agent-configs/* /opt/sutazaiapp/agents/configs/

# Restart services with legacy base
docker-compose -f docker-compose.yml up -d --force-recreate
```

#### Database State Restoration
```bash
# Restore agent registration states
psql -h postgres -d sutazai -c "
    UPDATE agents 
    SET base_version='1.0.0', status='active', updated_at=NOW()
    WHERE base_version='2.0.0';
"

# Clear enhanced agent metrics
redis-cli DEL "metrics:base-agent-v2:*"
```

### Phase 4: Validation (120-210 seconds)

#### System Health Verification
```bash
# Check all legacy agents are responding
/opt/sutazaiapp/scripts/validate-rollback.sh --full-check

# Verify task processing resumed
curl -f http://backend:8000/api/system/health

# Confirm resource utilization normalized
/opt/sutazaiapp/scripts/check-resources.sh --post-rollback
```

#### Performance Baseline Validation
- [ ] Task processing rate >= baseline
- [ ] Response times within normal range
- [ ] Memory usage < 70% of available
- [ ] CPU usage < 80% of available
- [ ] No error spikes in logs
- [ ] All health checks passing

## Rollback Validation Checklist

### Immediate Validation (0-3 minutes)
- [ ] Deployment process completely stopped
- [ ] Traffic successfully redirected to stable environment
- [ ] All legacy agents reporting healthy
- [ ] Enhanced agents gracefully shutdown
- [ ] No data loss or corruption
- [ ] System responding to requests

### Short-term Validation (3-15 minutes)
- [ ] Task queue processing normally
- [ ] No backlog of failed tasks
- [ ] Performance metrics at baseline
- [ ] Resource utilization stable
- [ ] Monitoring systems functional
- [ ] Alert systems operational

### Long-term Validation (15-60 minutes)
- [ ] System stability maintained
- [ ] No recurring issues
- [ ] Task completion rates normal
- [ ] User experience unaffected
- [ ] All integrations functional
- [ ] Backup systems verified

## Post-Rollback Analysis

### Immediate Actions
1. **Root Cause Analysis**: Identify why rollback was triggered
2. **Impact Assessment**: Measure affected systems and users
3. **Data Integrity Check**: Verify no data corruption occurred
4. **Performance Analysis**: Compare pre/post rollback metrics
5. **Communication**: Update stakeholders on situation

### Investigation Protocol
```bash
# Collect deployment logs
tar -czf rollback-logs-$(date +%Y%m%d_%H%M%S).tar.gz \
    /opt/sutazaiapp/logs/deployment_*.log \
    /var/log/docker/*.log \
    /opt/sutazaiapp/monitoring/metrics/

# Analyze failure patterns
/opt/sutazaiapp/scripts/analyze-rollback.sh --incident-report

# Generate rollback report
/opt/sutazaiapp/scripts/generate-rollback-report.sh > rollback-incident-$(date +%Y%m%d_%H%M%S).md
```

## Rollback Prevention Strategies

### Pre-deployment Validation
- Comprehensive testing in staging environment
- Resource capacity planning and validation
- Performance benchmarking and comparison
- Gradual rollout with smaller batches
- Enhanced monitoring during deployment

### Early Warning Systems
- Real-time resource monitoring with alerts
- Automated performance regression detection
- Circuit breaker pattern implementation
- Health check frequency increase during deployment
- Predictive failure analysis

## Recovery Planning

### Retry Strategy
After successful rollback and root cause resolution:

1. **Issue Resolution**: Fix identified problems
2. **Enhanced Testing**: Additional validation in staging
3. **Gradual Retry**: Smaller batch sizes for next attempt
4. **Extended Monitoring**: Longer observation periods
5. **Team Review**: Post-incident retrospective

### Lessons Learned Integration
- Update deployment procedures based on findings
- Enhance monitoring and alerting systems
- Improve rollback trigger sensitivity
- Strengthen pre-deployment validation
- Document new failure patterns and solutions

## Emergency Contacts

### Escalation Path
1. **L1 - Deployment Engineer**: Immediate response required
2. **L2 - Infrastructure Lead**: < 2 minutes response
3. **L3 - System Architect**: < 5 minutes response  
4. **L4 - Engineering Director**: < 10 minutes response

### Communication Channels
- **Primary**: Slack #deployment-alerts
- **Secondary**: PagerDuty escalation
- **Emergency**: Direct phone calls
- **Documentation**: Incident tracking system

## Rollback Automation Scripts

### Script Locations
- `/opt/sutazaiapp/scripts/rollback-ollama-integration.sh` - Main rollback script
- `/opt/sutazaiapp/scripts/validate-rollback.sh` - Validation script
- `/opt/sutazaiapp/scripts/analyze-rollback.sh` - Analysis script
- `/opt/sutazaiapp/scripts/generate-rollback-report.sh` - Reporting script

### Script Execution Order
1. **Immediate**: `rollback-ollama-integration.sh --emergency`
2. **Validation**: `validate-rollback.sh --full-check`
3. **Analysis**: `analyze-rollback.sh --incident-report`
4. **Reporting**: `generate-rollback-report.sh`

## Success Criteria for Rollback Completion

### Technical Metrics
- All legacy agents healthy and responsive
- Task processing rate >= 95% of baseline
- Response times within 110% of baseline
- Resource usage within normal operating range
- No active alerts or alarms
- System stability confirmed for 30+ minutes

### Business Metrics
- Zero user-visible impact
- No task failures or data loss
- Service availability maintained at 99.9%+
- No escalation to external stakeholders
- Recovery time within SLA requirements

---

## Rollback Timeline Summary

| Phase | Duration | Cumulative | Key Activities |
|-------|----------|------------|----------------|
| Immediate Stop | 0-30s | 30s | Halt deployment, alert team |
| Traffic Redirect | 30-60s | 90s | Route to stable environment |
| Container Swap | 60-120s | 210s | Replace with legacy agents |
| Validation | 120-210s | 330s | Verify system health |
| **Total** | **5.5 minutes** | | Complete rollback |

**Target Recovery Time**: < 60 seconds for traffic redirection
**Maximum Recovery Time**: < 3.5 minutes for full system restoration

This rollback procedure ensures rapid, reliable recovery from any deployment issues while maintaining system integrity and minimizing user impact.