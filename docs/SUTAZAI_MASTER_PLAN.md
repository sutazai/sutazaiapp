# SutazAI Master Plan
(Hyper‑Exhaustive "Maximal Overdrive" Edition)

**Owner**: Florin Cristian Suta  
**Contact**: chrissuta01@gmail.com | +48 517 716 005

## Table of Contents
- [Pre-Phase: Fixing Current Errors](#pre-phase-fixing-current-errors)
- [Phase 1: Codebase Audit & Repository Reorganization](#phase-1-codebase-audit--repository-reorganization)
- [Phase 2: Two‑Server Synchronization & Supreme AI Orchestrator Deployment](#phase-2-twoserver-synchronization--supreme-ai-orchestrator-deployment)
- [Phase 3: Automated Deployment Pipeline & Dependency Resolution](#phase-3-automated-deployment-pipeline--dependency-resolution)
- [Phase 4: Document & Diagram Parsing Integration](#phase-4-document--diagram-parsing-integration)
- [Phase 5: Advanced Code Generation & Autonomous Development](#phase-5-advanced-code-generation--autonomous-development)
- [Phase 6: OTP-Based Online Override & Self‑Improvement Routines](#phase-6-otp-based-online-override--selfimprovement-routines)
- [Phase 7: Final Audit, Documentation & Handoff](#phase-7-final-audit-documentation--handoff)
- [Phase 8: Continuous Maintenance, Observability & Advanced CI/CD](#phase-8-continuous-maintenance-observability--advanced-cicd)
- [Phase 9: Final Handoff, Training & Future Scalability](#phase-9-final-handoff-training--future-scalability)

## Pre-Phase: Fixing Current Errors

Before starting Phase 1, you may encounter:

- `chown: invalid user: 'sutazai_dev:sutazai_dev'`
- `fatal: Need to specify how to reconcile divergent branches.`

### Issue A: Invalid User

**Cause**: The user `sutazai_dev` does not exist.

**Steps to Fix**:
1. Create the Non‑Root User:
   ```bash
   sudo adduser sutazai_dev
   ```
   You'll be prompted for a password and user details.

2. (Optional) Grant Sudo Privileges:
   ```bash
   sudo usermod -aG sudo sutazai_dev
   ```

3. Re‑run chown:
   ```bash
   cd /opt
   sudo chown -R sutazai_dev:sutazai_dev SutazAI
   ```

4. Verify:
   ```bash
   ls -l /opt | grep SutazAI
   ```
   Should list `sutazai_dev sutazai_dev` as owner/group.

### Issue B: Git Branch Divergence

**Cause**: Your local Git branch and the remote origin/master have diverged.

**Ways to Fix**:
1. Merge Strategy (traditional default):
   ```bash
   git config pull.rebase false
   git pull origin master
   ```

2. Fast‑Forward Only:
   ```bash
   git config pull.ff only
   git pull origin master
   ```

3. Rebase:
   ```bash
   git config pull.rebase true
   git pull origin master
   ```

4. One-Time:
   ```bash
   # Merge:
   git pull origin master --no-rebase

   # or Rebase:
   git pull origin master --rebase
   ```

## Phase 1: Codebase Audit & Repository Reorganization

### Objective
- Clone SutazAI onto both servers (Code & Deployment)
- Reorganize directories for clarity and modularity
- Set up Python virtual environment
- Audit the code with automated and manual checks

### Detailed Steps

#### Clone & Pull from GitHub

**Code Server (192.168.100.28)**:
```bash
cd /opt
sudo mkdir SutazAI
sudo chown -R sutazai_dev:sutazai_dev SutazAI
cd SutazAI
git clone https://github.com/sutazai/sutazaiapp.git .
git config pull.rebase false  # Choose your preferred strategy
git pull origin master
```

**Deployment Server (192.168.100.100)**: Same steps as above.

#### Directory Reorganization

**Ideal layout**:
```
/opt/SutazAI/
├── ai_agents/
├── model_management/
├── backend/
├── web_ui/
├── scripts/
├── packages/
├── logs/
├── doc_data/
└── docs/
```

**Permissions**:
```bash
sudo chown -R sutazai_dev:sutazai_dev /opt/SutazAI/
chmod -R 750 /opt/SutazAI/
```

Document in `/docs/DIRECTORY_STRUCTURE.md`.

#### Virtual Environment & Dependencies

```bash
cd /opt/SutazAI
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt || \
    pip install --no-index --find-links=packages/wheels -r requirements.txt
```

Record in `/docs/DEPLOYMENT.md`.

#### Initial Code Audit

**Automated**:
```bash
semgrep --config=auto .
pylint backend/
mypy .
bandit -r .
```

**Manual**: Inspect `scripts/deploy.sh`, `ai_agents/superagi/`, etc.

Summarize in `/docs/audit/Audit_Report_v1.md`.

## Phase 2: Two‑Server Synchronization & Supreme AI Orchestrator Deployment

### Objective
- Automate code sync: Code Server → Deployment Server
- Deploy the Supreme AI Orchestrator (SuperAGI) with Florin Cristian Suta's details

### Detailed Steps

#### SSH Key Setup

On Code Server as sutazai_dev:
```bash
ssh-keygen -t ed25519 -C "sutazai_deploy" -f ~/.ssh/sutazai_deploy -N ""
ssh-copy-id -i ~/.ssh/sutazai_deploy.pub root@192.168.100.100
```

Document in `/docs/REPO_SYNC.md`.

#### Git Hook: Post‑Commit

`/opt/SutazAI/.git/hooks/post-commit`:
```bash
#!/bin/bash
ssh root@192.168.100.100 "cd /opt/SutazAI && ./scripts/trigger_deploy.sh"
```
```bash
chmod +x .git/hooks/post-commit
```

**Fallback**: `scripts/setup_repos.sh`:
```bash
#!/bin/bash
cd /opt/SutazAI
git fetch --all
git reset --hard origin/master
```
```bash
chmod +x scripts/setup_repos.sh
```

#### Supreme AI Orchestrator (SuperAGI) Deployment

`/ai_agents/superagi/config.toml`:
```toml
[owner]
name = "Florin Cristian Suta"
email = "chrissuta01@gmail.com"
phone = "+48517716005"
```

**Start Script**: `scripts/start_superagi.sh`:
```bash
#!/bin/bash
cd /opt/SutazAI/ai_agents/superagi
nohup python3 supreme_agent.py >> /opt/SutazAI/logs/agent.log 2>&1 &
```
```bash
chmod +x scripts/start_superagi.sh
```

Test by running it on Deployment Server, watch `/opt/SutazAI/logs/agent.log`.

#### Basic Logging

JSON logs:
```python
import logging, json
logger = logging.getLogger("superagi")
logger.info(json.dumps({"event": "orchestrator_start", "owner": "Florin Cristian Suta"}))
```

## Phase 3: Automated Deployment Pipeline & Dependency Resolution

### Objective
- Develop a robust deploy.sh with offline fallback, OTP gating, logging, health checks, rollback on error

### Detailed Steps

#### deploy.sh (in scripts/)

**Steps**:
- git pull origin master
- If external fetch needed, OTP check
- Create/activate venv
- Install dependencies
- Verify model files
- Start backend + web UI, run health checks
- Log to `/opt/SutazAI/logs/deploy.log`
- Rollback on errors

**Sample**:
```bash
#!/usr/bin/env bash
set -euo pipefail
DEPLOY_LOG="/opt/SutazAI/logs/deploy.log"
exec > >(tee -a "$DEPLOY_LOG") 2>&1

git pull origin master || echo "Warning: can't pull"

if [[ -n "${OTP_TOKEN:-}" ]]; then
  python3 scripts/otp_override.py validate "$OTP_TOKEN" || { echo "Invalid OTP"; exit 1; }
fi

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip setuptools wheel

pip install -r requirements.txt || \
    pip install --no-index --find-links=packages/wheels -r requirements.txt

# Verify models, etc...

# Start backend & UI, run health checks...
```

**Rollback**:
```bash
trap 'echo "Error line $LINENO"; ./scripts/deploy.sh --rollback <COMMIT>; exit 1' ERR

if [[ "$1" == "--rollback" ]]; then
    git reset --hard "$2"
    ./scripts/deploy.sh
    exit 0
fi
```

**Offline Fallback**: `.whl` files in `packages/wheels/`, or Node modules in `packages/node/`.

Document usage in `/docs/DEPLOYMENT.md`.

## Phase 4: Document & Diagram Parsing Integration

### Objective
- Parse PDFs, DOCXs, and analyze diagrams (Molmo or placeholders)
- Provide REST endpoints + a UI for file uploads

### Detailed Steps

#### Document Parsing (backend/services/doc_processing.py)

```python
import fitz, docx2txt
def parse_pdf(path: str) -> dict: ...
def parse_docx(path: str) -> dict: ...
```

#### Diagram Parsing (backend/services/diagram_parser.py)

```python
def analyze_diagram(path: str) -> dict: ...
```

#### FastAPI Endpoints

```python
@router.post("/doc/parse")
async def doc_parse(file: UploadFile = File(...)): ...

@router.post("/diagram/analyze")
async def diagram_analyze(file: UploadFile = File(...)): ...
```

#### UI

- `DocumentUploader.jsx`, `DiagramUploader.jsx`
- `<input type="file" />` → POST → `/doc/parse` or `/diagram/analyze`

#### Testing

- Large PDFs, corrupted files, etc.

## Phase 5: Advanced Code Generation & Autonomous Development

### Objective
- Integrate local LLMs (GPT4All, DeepSeek‑Coder) to generate code from textual specs
- Add multi‑agent synergy (AutoGPT, Semgrep, TabbyML, etc.)
- Provide UI to enter specs, see generated code

### Detailed Steps

#### Code Generation Module (backend/services/code_generation.py)

```python
def generate_code(spec_text: str, model_choice: str = "gpt4all") -> str: ...
def local_llm_generate(spec_text: str, model: str) -> str: ...
```

#### Multi-Agent Orchestrator

- Orchestrator calls AutoGPT, Semgrep, TabbyML for refinement
- Logs attempts in `/opt/SutazAI/logs/agent.log`
- API Endpoint (`/code/generate`)
- UI (`CodeGeneration.jsx`)
- Testing & Performance

## Phase 6: OTP-Based Online Override & Self‑Improvement Routines

### Objective
- Mandate OTP for external ops (pip installs, remote calls)
- Orchestrator self‑improves by monitoring logs/performance, auto‑redeploying or rolling back on performance anomalies

### Detailed Steps

#### OTP Enforcement (scripts/otp_override.py)

```python
import pyotp
SECRET = "BASE32SECRET123"
def validate_otp(token: str) -> bool: ...
```

#### Self‑Improvement

- Orchestrator monitors logs, doc parse times, code gen success, etc.
- Auto‑adjusts or triggers rollback if thresholds are exceeded

#### Automated Rollback

```bash
git reset --hard <LAST_STABLE_COMMIT>
./scripts/deploy.sh
```

Logs each rollback event.

## Phase 7: Final Audit, Documentation & Handoff

### Objective
- Final exhaustive audit (automated + manual)
- Unify all docs (deployment, security, AI agents, code generation, self‑improvement)
- Create a handoff archive and training materials

### Detailed Steps

#### Final Audit

```bash
semgrep --config=auto .
pylint backend/
mypy .
bandit -r .
```

Summarize in `/docs/audit/Audit_Report_vFinal.md`.

#### Documentation

- `/docs/DEPLOYMENT.md`, `/docs/SECURITY.md`, `/docs/AI_AGENTS.md`, `/docs/CODE_GENERATION.md`, etc.
- Architecture diagrams in `/docs/architecture/`

#### Handoff Archive

```bash
tar -czvf sutazai_handoff.tar.gz /opt/SutazAI --exclude=venv
```

Document in `/docs/Handoff.md`.

#### Training

- Possibly record short screen captures for dev onboarding

## Phase 8: Continuous Maintenance, Observability & Advanced CI/CD

### Objective
- Real-time observability (Prometheus, Grafana), advanced logging & alerting, a CI/CD pipeline for continuous testing/scanning, integrated with orchestrator's self‑improvement logic

### Detailed Steps

#### Observability

- Prometheus + Node Exporter for CPU, memory, doc parse stats
- Grafana dashboards for real-time metrics, set alerts if usage/exceptions spike
- (Optional) Jaeger for distributed tracing if microservices expand

#### Advanced Logging & Alerting

- Structured JSON logs, logrotate for `/opt/SutazAI/logs/*.log`
- Slack/email notifications on repeated rollbacks, OTP errors, or CPU > 90%

#### CI/CD Pipeline

- Jenkins, GitHub Actions, or GitLab CI
- Stages:
  - Static Analysis (Semgrep, Bandit)
  - Unit/Integration Tests (Pytest)
  - Performance Tests (Locust, k6)
  - Security (optional container scans if Docker-based)
  - Deployment (calls deploy.sh if OTP or override is provided)

Document in `/docs/CI_CD.md`.

#### Continuous Self‑Improvement

- Orchestrator references Prometheus data to adapt or rollback automatically

## Phase 9: Final Handoff, Training & Future Scalability

### Objective
- Provide a final archive of the entire system, plus comprehensive training materials
- Outline a roadmap for GPU acceleration, advanced AI modules, multi‑region expansions

### Detailed Steps

#### Final Archive

```bash
git pull origin master
tar -czvf sutazai_final_handoff.tar.gz /opt/SutazAI --exclude=venv
```

Document in `/docs/Handoff.md`.

#### Developer Training & Onboarding

- Workshops or recorded sessions on offline fallback, OTP usage, doc/diagram parsing, code generation, orchestrator logic, rollbacks, logs
- A Q&A doc in `/docs/training/FAQ.md` for common issues

#### Future Scalability

- **GPU Integration**: Steps for installing an NVIDIA GPU in the R720, configuring CUDA drivers, enabling PyTorch or TensorFlow GPU acceleration
- **Additional Agents**: RL modules, knowledge graphs, specialized domain LLMs
- **Multi‑Region Deployments**: Potentially replicate environment across data centers or use container orchestration (Kubernetes) if scaling out

## Conclusion

By adhering to these hyper‑exhaustive instructions, your SutazAI platform becomes secure, OTP‑gated (offline first), self‑improving, and scalable, fully under the ownership of Florin Cristian Suta—ensuring robust AI functionalities, thorough logging, and advanced future‑proofing for GPU and multi‑region expansions. 