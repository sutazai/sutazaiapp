# Security & Privacy

- Threat Model (STRIDE):
  - Spoofing: enforce JWT + service tokens
  - Tampering: signed images; immutability
  - Repudiation: audit logs
  - Information Disclosure: TLS, least privilege
  - Denial of Service: rate limits, backpressure
  - Elevation of Privilege: RBAC, container caps

- AuthN/Z: JWT, RBAC, scopes per endpoint
- Secrets: `.env` only; no hardcoded secrets; rotation quarterly
- Encryption: in transit (TLS in gateway); at rest via disk encryption
- Least Privilege: per-service DB users, minimal grants

Footnotes: [`SUTAZAI_PRD.md` 40–60]
