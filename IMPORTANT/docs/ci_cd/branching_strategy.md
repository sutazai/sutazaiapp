# Branching Strategy and Git Workflow

**Created:** 2025-08-08  
**Version:** 1.0.0  
**Status:** Active

## Overview

This document defines the Git branching strategy, naming conventions, and workflow for the SutazAI project. We follow a modified GitFlow approach optimized for continuous delivery.

## Branch Types and Naming Conventions

### Main Branches (Protected)

#### `main`
- **Purpose:** Production-ready code
- **Protection:** Requires PR, 2 approvals, passing CI/CD
- **Deployment:** Auto-deploys to production
- **Merge From:** `release/*` and `hotfix/*` only

#### `develop`
- **Purpose:** Integration branch for next release
- **Protection:** Requires PR, 1 approval, passing CI/CD
- **Deployment:** Auto-deploys to staging
- **Merge From:** `feature/*`, `bugfix/*`

### Supporting Branches

#### `feature/*`
- **Naming:** `feature/JIRA-123-short-description`
- **Purpose:** New features and enhancements
- **Created From:** `develop`
- **Merge To:** `develop`
- **Lifetime:** Delete after merge
- **Examples:**
  - `feature/SUTA-456-add-oauth-integration`
  - `feature/SUTA-789-optimize-vector-search`

#### `bugfix/*`
- **Naming:** `bugfix/JIRA-123-short-description`
- **Purpose:** Non-critical bug fixes
- **Created From:** `develop`
- **Merge To:** `develop`
- **Lifetime:** Delete after merge
- **Examples:**
  - `bugfix/SUTA-321-fix-memory-leak`
  - `bugfix/SUTA-654-correct-api-response`

#### `release/*`
- **Naming:** `release/v1.2.3`
- **Purpose:** Prepare for production release
- **Created From:** `develop`
- **Merge To:** `main` and back to `develop`
- **Lifetime:** Delete after merge
- **Examples:**
  - `release/v17.1.0`
  - `release/v17.2.0-rc1`

#### `hotfix/*`
- **Naming:** `hotfix/JIRA-123-short-description`
- **Purpose:** Critical production fixes
- **Created From:** `main`
- **Merge To:** `main` and `develop`
- **Lifetime:** Delete after merge
- **Examples:**
  - `hotfix/SUTA-999-fix-security-vulnerability`
  - `hotfix/SUTA-888-restore-database-connection`

### Experimental Branches

#### `experiment/*`
- **Naming:** `experiment/description`
- **Purpose:** Proof of concepts, experiments
- **Created From:** Any branch
- **Merge To:** None (reference only)
- **Lifetime:** Archive after 30 days
- **Examples:**
  - `experiment/quantum-optimization`
  - `experiment/new-ui-framework`

## Commit Message Convention

We follow the Conventional Commits specification:

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or corrections
- `build`: Build system changes
- `ci`: CI/CD configuration changes
- `chore`: Maintenance tasks
- `revert`: Revert previous commit

### Examples
```bash
# Feature
feat(backend): add vector similarity search endpoint

# Bug fix
fix(frontend): correct navigation menu overflow

# Documentation
docs(api): update authentication flow diagram

# Performance
perf(database): optimize query with proper indexing

# Breaking change
feat(api)!: change response format for /agents endpoint

BREAKING CHANGE: Response now returns array instead of object
```

## Pull Request Requirements

### PR Title Format
```
[JIRA-123] Type: Brief description
```

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added/updated
- [ ] CHANGELOG.md updated

## Screenshots (if applicable)

## Related Issues
Closes #123
```

### Review Requirements

#### Standard Changes
- Minimum 1 approval
- All CI/CD checks passing
- No merge conflicts
- Code coverage maintained or improved

#### High-Risk Changes
- Minimum 2 approvals (1 senior engineer)
- Security team review for auth/crypto changes
- Performance team review for database changes
- Architecture team review for major refactoring

## Feature Flags

### Implementation
```python
# backend/app/core/feature_flags.py
FEATURE_FLAGS = {
    "new_vector_search": os.getenv("ENABLE_NEW_VECTOR_SEARCH", "false") == "true",
    "advanced_monitoring": os.getenv("ENABLE_ADVANCED_MONITORING", "false") == "true",
    "experimental_ai": os.getenv("ENABLE_EXPERIMENTAL_AI", "false") == "true",
}

# Usage
if FEATURE_FLAGS["new_vector_search"]:
    return new_vector_search_implementation()
else:
    return legacy_search()
```

### Flag Lifecycle
1. **Development:** Flag created, disabled by default
2. **Testing:** Enabled in staging environment
3. **Rollout:** Gradual production enablement
4. **Stable:** Enabled by default
5. **Cleanup:** Flag and old code removed

## Version Tagging

### Semantic Versioning
We follow SemVer: `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes
- **MINOR:** New features (backwards compatible)
- **PATCH:** Bug fixes

### Tag Format
```bash
# Production release
git tag -a v17.1.0 -m "Release version 17.1.0"

# Pre-release
git tag -a v17.2.0-rc1 -m "Release candidate 1 for 17.2.0"

# Beta
git tag -a v17.3.0-beta.1 -m "Beta 1 for 17.3.0"
```

## Workflow Examples

### Feature Development
```bash
# 1. Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/SUTA-123-add-new-endpoint

# 2. Make changes and commit
git add .
git commit -m "feat(api): add new /analysis endpoint"

# 3. Push and create PR
git push origin feature/SUTA-123-add-new-endpoint
# Create PR via GitHub/GitLab UI

# 4. After approval and merge
git checkout develop
git pull origin develop
git branch -d feature/SUTA-123-add-new-endpoint
```

### Hotfix Deployment
```bash
# 1. Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/SUTA-999-critical-fix

# 2. Make fix and commit
git add .
git commit -m "fix(security): patch SQL injection vulnerability"

# 3. Push and create PR to main
git push origin hotfix/SUTA-999-critical-fix
# Create PR to main

# 4. After merge to main, merge to develop
git checkout develop
git pull origin develop
git merge main
git push origin develop
```

### Release Process
```bash
# 1. Create release branch
git checkout develop
git pull origin develop
git checkout -b release/v17.2.0

# 2. Update version numbers
# Update version in pyproject.toml, package.json, etc.
git commit -m "chore(release): bump version to 17.2.0"

# 3. Fix any release issues
git commit -m "fix(release): correct production configuration"

# 4. Merge to main
git checkout main
git merge --no-ff release/v17.2.0
git tag -a v17.2.0 -m "Release version 17.2.0"
git push origin main --tags

# 5. Merge back to develop
git checkout develop
git merge --no-ff release/v17.2.0
git push origin develop

# 6. Clean up
git branch -d release/v17.2.0
```

## Merge Strategies

### Feature to Develop
- **Strategy:** Squash and merge
- **Reason:** Clean history, single commit per feature

### Release to Main
- **Strategy:** Merge commit (no fast-forward)
- **Reason:** Preserve release history

### Hotfix to Main
- **Strategy:** Merge commit
- **Reason:** Traceable hotfix history

### Main to Develop (sync)
- **Strategy:** Merge commit
- **Reason:** Clear sync points

## Branch Protection Rules

### Main Branch
```yaml
Protection Rules:
  - Require pull request before merging
  - Require approvals: 2
  - Dismiss stale reviews
  - Require review from CODEOWNERS
  - Require status checks:
    - CI/CD Pipeline
    - Security Scan
    - Code Coverage (>80%)
  - Require branches up to date
  - Include administrators
  - Restrict push access to: release-managers
```

### Develop Branch
```yaml
Protection Rules:
  - Require pull request before merging
  - Require approvals: 1
  - Require status checks:
    - CI/CD Pipeline
    - Unit Tests
    - Linting
  - Require branches up to date
```

## Conflict Resolution

### Guidelines
1. **Regular Syncing:** Pull from target branch daily
2. **Small PRs:** Keep changes focused and minimal
3. **Communication:** Coordinate with team on shared files
4. **Testing:** Always test after resolving conflicts

### Resolution Process
```bash
# Update your branch with latest changes
git checkout feature/your-branch
git fetch origin
git rebase origin/develop

# Resolve conflicts
# Edit conflicted files
git add .
git rebase --continue

# Force push if necessary (only on feature branches)
git push origin feature/your-branch --force-with-lease
```

## Git Hooks

### Pre-commit
```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run linting
flake8 backend/ || exit 1
eslint frontend/ || exit 1

# Check for secrets
git secrets --pre_commit_hook || exit 1
```

### Commit-msg
```bash
#!/bin/sh
# .git/hooks/commit-msg

# Validate commit message format
commit_regex='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    exit 1
fi
```

## Rollback Procedures

### Feature Rollback
```bash
# Revert merge commit
git checkout develop
git revert -m 1 <merge-commit-hash>
git push origin develop
```

### Production Rollback
```bash
# Quick rollback to previous version
git checkout main
git revert HEAD
git push origin main

# Or rollback to specific version
git checkout main
git reset --hard v17.1.0
git push origin main --force
```

## Best Practices

### Do's
- ✅ Keep branches short-lived (<1 week)
- ✅ Update branch daily from target
- ✅ Write descriptive commit messages
- ✅ Test before pushing
- ✅ Review your own code first
- ✅ Delete branches after merge
- ✅ Use feature flags for risky changes

### Don'ts
- ❌ Force push to protected branches
- ❌ Commit directly to main/develop
- ❌ Merge without reviews
- ❌ Leave commented code
- ❌ Commit sensitive data
- ❌ Create branches from outdated code
- ❌ Merge with failing tests

## Monitoring and Metrics

### Branch Metrics
- Average branch lifetime
- Number of commits per branch
- Merge conflict frequency
- Review turnaround time

### Quality Metrics
- Code coverage trends
- Build success rate
- Test pass rate
- Security scan results

## Training Resources

### Internal
- Git workflow training videos
- Branching strategy workshop
- Code review best practices guide

### External
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

## Support

For questions or issues:
- **Slack:** #git-help
- **Wiki:** Internal Git Guide
- **Contact:** DevOps Team

---

*This document is version controlled. Submit changes via PR.*