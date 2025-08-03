# CI/CD Hygiene Integration Guide

## Overview

This guide provides comprehensive instructions for integrating codebase hygiene enforcement into your CI/CD pipelines. The system supports GitHub Actions, GitLab CI, Jenkins, and can be adapted to other platforms.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Platforms](#supported-platforms)
3. [Configuration](#configuration)
4. [Enforcement Levels](#enforcement-levels)
5. [Agent Integration](#agent-integration)
6. [Quality Gates](#quality-gates)
7. [Monitoring & Reporting](#monitoring--reporting)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Quick Start

### GitHub Actions

1. The workflow is automatically triggered on:
   - Pull requests
   - Pushes to main/develop branches
   - Daily schedule (2 AM UTC)
   - Manual dispatch

2. No additional setup required - the workflow file is already in place at `.github/workflows/hygiene-enforcement.yml`

### GitLab CI

1. Include the hygiene configuration in your main `.gitlab-ci.yml`:
   ```yaml
   include:
     - local: '.gitlab-ci-hygiene.yml'
   ```

2. Configure variables in GitLab project settings:
   - `ENFORCEMENT_LEVEL_MR`: Level for merge requests (default: standard)
   - `ENFORCEMENT_LEVEL_SCHEDULE`: Level for scheduled runs (default: comprehensive)

### Jenkins

1. Create a new Pipeline job
2. Set the Pipeline script from SCM
3. Point to `jenkins/Jenkinsfile.hygiene`
4. Configure parameters as needed

### Universal Runner

For custom CI/CD platforms or local testing:

```bash
# Quick analysis
./scripts/ci-cd/hygiene-runner.sh --rules "1,2,3" --priority critical

# Comprehensive check
./scripts/ci-cd/hygiene-runner.sh --priority high --dry-run

# With custom configuration
./scripts/ci-cd/hygiene-runner.sh --config my-hygiene-config.json
```

## Supported Platforms

### GitHub Actions
- **File**: `.github/workflows/hygiene-enforcement.yml`
- **Features**:
  - Automatic PR comments with hygiene reports
  - Artifact uploads for all reports
  - Quality gate enforcement
  - Scheduled cleanup with automated PRs
  - Integration with GitHub security features

### GitLab CI
- **File**: `.gitlab-ci-hygiene.yml`
- **Features**:
  - Merge request integration
  - JUnit test report format
  - Metrics export for monitoring
  - Docker-in-Docker support
  - Automated merge request creation

### Jenkins
- **File**: `jenkins/Jenkinsfile.hygiene`
- **Features**:
  - Blue Ocean compatible
  - HTML report publishing
  - Email/Slack notifications
  - Build badges
  - Parameterized builds

### Universal Support
- **Script**: `scripts/ci-cd/hygiene-runner.sh`
- **Compatible with**:
  - CircleCI
  - Travis CI
  - Azure DevOps
  - Bamboo
  - TeamCity
  - Local development

## Configuration

### Global Configuration File

Create `.hygiene-config.json` in your project root:

```json
{
  "enforcement_level": "standard",
  "rules": {
    "critical": [1, 2, 3, 13],
    "high": [4, 5, 8, 11, 12],
    "medium": [6, 7, 9, 10],
    "low": [14, 15, 16]
  },
  "thresholds": {
    "critical_violations": 0,
    "high_violations": 5,
    "hygiene_score": 70
  },
  "agents": {
    "garbage-collector": {
      "enabled": true,
      "auto_fix": true
    },
    "script-consolidator": {
      "enabled": true,
      "auto_fix": false
    }
  },
  "notifications": {
    "slack": {
      "webhook": "${SLACK_WEBHOOK}",
      "channel": "#ci-notifications"
    },
    "email": {
      "recipients": ["dev-team@example.com"],
      "on_failure": true,
      "on_success": false
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENFORCEMENT_LEVEL` | Overall enforcement level | `standard` |
| `DRY_RUN` | Run without making changes | `false` |
| `HYGIENE_RULES` | Comma-separated rule numbers | All rules |
| `MAX_PARALLEL_JOBS` | Maximum parallel analysis jobs | `4` |
| `ARTIFACT_RETENTION_DAYS` | How long to keep reports | `30` |

## Enforcement Levels

### Quick
- **Rules**: Critical violations only (1, 2, 3, 13)
- **Agents**: None
- **Use case**: Fast feedback on PRs
- **Duration**: ~5 minutes

### Standard
- **Rules**: Critical + High priority (1-8, 11-13)
- **Agents**: Limited set (garbage-collector, code-auditor)
- **Use case**: Regular PR checks
- **Duration**: ~15 minutes

### Comprehensive
- **Rules**: All 16 rules
- **Agents**: All available agents
- **Use case**: Scheduled runs, release branches
- **Duration**: ~30-45 minutes

## Agent Integration

### Available Agents

1. **garbage-collector**
   - Rules: 13
   - Purpose: Remove junk files and abandoned code
   - Safe for automation: Yes

2. **script-consolidator**
   - Rules: 7, 12
   - Purpose: Consolidate duplicate scripts
   - Safe for automation: Partial (requires review)

3. **docker-optimizer**
   - Rules: 11
   - Purpose: Optimize Docker structure
   - Safe for automation: No (requires manual review)

4. **code-auditor**
   - Rules: 1, 2, 3
   - Purpose: Validate code compliance
   - Safe for automation: Yes (analysis only)

5. **documentation-manager**
   - Rules: 6, 15
   - Purpose: Organize and deduplicate documentation
   - Safe for automation: Yes

### Custom Agent Configuration

```yaml
# In your CI configuration
- name: Run custom agent
  run: |
    python scripts/agents/hygiene-agent-orchestrator.py \
      --agent "my-custom-agent" \
      --rules "1,2,3" \
      --config "agents/configs/my-agent.yaml"
```

## Quality Gates

### Default Quality Gates

1. **Critical Violations**: Must be 0
2. **High Violations**: Warning if > 5
3. **Hygiene Score**: Must be ≥ 70%

### Custom Quality Gates

Configure in your CI pipeline:

```yaml
# GitHub Actions example
- name: Evaluate custom quality gates
  run: |
    SCORE=$(jq -r '.summary.hygiene_score' hygiene-report.json)
    if (( $(echo "$SCORE < 80" | bc -l) )); then
      echo "::error::Hygiene score too low: $SCORE%"
      exit 1
    fi
```

## Monitoring & Reporting

### Report Formats

1. **JSON**: Machine-readable, full details
2. **Markdown**: Human-readable, PR comments
3. **HTML**: Web-viewable, with charts
4. **JUnit XML**: Test result format
5. **Prometheus**: Metrics export

### Metrics Exported

- `hygiene_score`: Overall codebase hygiene score (0-100)
- `critical_violations_total`: Count of critical violations
- `high_violations_total`: Count of high priority violations
- `medium_violations_total`: Count of medium priority violations
- `low_violations_total`: Count of low priority violations
- `enforcement_duration_seconds`: Pipeline execution time
- `agents_executed_total`: Number of agents run

### Dashboard Integration

#### Grafana
```json
{
  "dashboard": {
    "title": "Codebase Hygiene",
    "panels": [
      {
        "title": "Hygiene Score Trend",
        "targets": [
          {
            "expr": "hygiene_score"
          }
        ]
      }
    ]
  }
}
```

#### Slack Notifications
```javascript
{
  "text": "Hygiene Enforcement Results",
  "blocks": [
    {
      "type": "section",
      "fields": [
        {
          "type": "mrkdwn",
          "text": `*Score:* ${score}%`
        },
        {
          "type": "mrkdwn",
          "text": `*Critical:* ${critical}`
        }
      ]
    }
  ]
}
```

## Troubleshooting

### Common Issues

#### 1. Pipeline Timeout
**Problem**: Pipeline exceeds time limit
**Solution**: 
- Reduce enforcement level
- Increase timeout in CI configuration
- Run rules in smaller batches

#### 2. Agent Failures
**Problem**: Specific agent consistently fails
**Solution**:
- Check agent logs in `logs/agent-*.log`
- Verify agent dependencies are installed
- Run agent locally with verbose mode

#### 3. False Positives
**Problem**: Rules flagging valid code
**Solution**:
- Add exceptions to `.hygiene-ignore` file
- Customize rule configuration
- Submit issue for rule improvement

#### 4. Merge Conflicts in Automated PRs
**Problem**: Automated cleanup PRs have conflicts
**Solution**:
- Configure rebase strategy
- Limit scope of automated fixes
- Schedule during low-activity periods

### Debug Mode

Enable verbose logging:

```bash
# GitHub Actions
- run: ./scripts/ci-cd/hygiene-runner.sh --verbose

# GitLab CI
variables:
  HYGIENE_VERBOSE: "true"

# Jenkins
parameters {
  booleanParam(name: 'VERBOSE', defaultValue: false)
}
```

## Best Practices

### 1. Progressive Enforcement
Start with warning-only mode and gradually increase enforcement:

```yaml
Week 1-2: Dry run only, gather baseline
Week 3-4: Enforce critical rules only
Week 5-6: Add high priority rules
Week 7+: Full enforcement
```

### 2. Team Communication
- Announce hygiene enforcement rollout
- Provide training on CLAUDE.md rules
- Share weekly hygiene reports
- Celebrate improvements

### 3. Performance Optimization
- Cache dependencies between runs
- Use parallel job execution
- Skip unchanged files when possible
- Archive old reports regularly

### 4. Integration with Development Workflow
- Add pre-commit hooks for local checking
- Integrate with IDE plugins
- Provide quick-fix scripts
- Document common violations and fixes

### 5. Continuous Improvement
- Review false positives monthly
- Update rule configurations based on feedback
- Track hygiene score trends
- Adjust thresholds based on project maturity

## Advanced Configuration

### Custom Rule Sets

Create rule profiles for different scenarios:

```json
{
  "profiles": {
    "feature-branch": {
      "rules": [1, 2, 3, 13],
      "enforcement": "warning"
    },
    "release-branch": {
      "rules": "all",
      "enforcement": "strict"
    },
    "hotfix": {
      "rules": [1, 2],
      "enforcement": "critical-only"
    }
  }
}
```

### Conditional Enforcement

```yaml
# Only enforce on specific file changes
- name: Check changed files
  run: |
    if git diff --name-only HEAD^ | grep -E '\.(py|js|yml)$'; then
      ./scripts/ci-cd/hygiene-runner.sh --rules "1,2,3"
    fi
```

### Integration with Code Review

```javascript
// Automatically request review from hygiene experts
if (hygieneScore < 80 || criticalViolations > 0) {
  github.pulls.requestReviewers({
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.issue.number,
    team_reviewers: ['hygiene-experts']
  });
}
```

## Appendix

### Rule Reference

| Rule | Description | Priority | Auto-fixable |
|------|-------------|----------|--------------|
| 1 | No fantasy elements | Critical | No |
| 2 | Don't break functionality | Critical | No |
| 3 | Analyze everything | Critical | No |
| 4 | Reuse before creating | High | Partial |
| 5 | Professional project | High | No |
| 6 | Clear documentation | Medium | Yes |
| 7 | Eliminate script chaos | High | Partial |
| 8 | Python script sanity | High | Yes |
| 9 | Version control | Medium | Partial |
| 10 | Functionality-first cleanup | Critical | No |
| 11 | Docker structure | High | Partial |
| 12 | Single deployment script | High | Partial |
| 13 | No garbage | Critical | Yes |
| 14 | Correct AI agent | Medium | No |
| 15 | Documentation dedup | Medium | Yes |
| 16 | Local LLMs via Ollama | Low | No |

### Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in `logs/hygiene-*.log`
3. Submit an issue with the hygiene report attached
4. Contact the DevOps team for CI/CD-specific issues