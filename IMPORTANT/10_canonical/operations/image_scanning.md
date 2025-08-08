# Image Scanning & SBOM Verification

- Vulnerability Scans
  - Use Trivy in CI: `trivy image --severity HIGH,CRITICAL --ignore-unfixed --format table registry/sutazai/backend:${GIT_SHA}`
  - For multi-arch manifests:
    - `trivy image --list-all-packs --vuln-type os,library registry/sutazai/backend:${GIT_SHA}`
    - Trivy pulls per-arch layers under the manifest list

- Filesystem Scans
  - `trivy fs --severity HIGH,CRITICAL --exit-code 1 .`

- SBOM
  - Enable SBOM during buildx builds (`--sbom=true`)
  - Verify SBOM artifacts exist for each tag and platform

- Policy
  - Block release on CRITICAL vulns; HIGH allowed with waiver and issue card
  - Store scan reports under `/reports/security/` with build metadata

- References
  - Trivy multi-arch scanning notes; SBOM best practices
