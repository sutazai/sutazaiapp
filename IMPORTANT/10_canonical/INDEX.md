# SutazAI Canonical Architecture (ASoT)

Authoritative, reconciled documentation. Diagrams are provided as Mermaid source (.mmd). PNG rendering is tracked separately (see ISSUE-0013) and will be produced via CI using `@mermaid-js/mermaid-cli`.

- Current State
  - Context: `current_state/context.mmd`
  - Containers: `current_state/containers.mmd`
  - Components (Backend): `current_state/components_backend.mmd`
  - Sequences: `current_state/seq_inference.mmd`, `seq_rag_ingest.mmd`, `seq_rag_query.mmd`, `seq_agent_exec.mmd`, `seq_alerting.mmd`
  - Data ERD: `current_state/erd_current.mmd`
  - Deployment: `current_state/deployment_topology.md`
  - Environments: `current_state/environments.md`

- Target State
  - Context: `target_state/context.mmd`
  - Containers: `target_state/containers.mmd`
  - Components: `target_state/components.mmd`
  - Data ERD: `target_state/erd_target.mmd`

- Domain Model & Glossary: `domain_model/domain_glossary.md`
- API & Integration Contracts: `api_contracts/contracts.md`
- Data: `data/data_management.md`
- Security & Privacy: `security/security_privacy.md`
- Reliability & Performance: `reliability/reliability_performance.md`
- Observability: `observability/observability.md`
- Operations: `operations/operations.md`, `operations/ci_cd_multiarch.md`, `operations/image_scanning.md`, `operations/archive_management.md`, `operations/docs_generation.md`
- Standards & Governance: `standards/engineering_standards.md`, `standards/ADR-0001.md`, `standards/ADR-0002.md`, `standards/ADR-0003.md`, `standards/ADR-0004.md`

Sources are cited as footnotes with file paths and line ranges using the format: [source] /opt/sutazaiapp/IMPORTANT/<path>#L<from>-L<to>.

Assumptions are explicitly labeled where runtime verification is pending; such items also have Issue Cards and backlog links.
