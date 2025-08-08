# CI/CD: Multi-Architecture Build & Release

Goals: deterministic, secure, reproducible images for linux/amd64 and linux/arm64 with SBOM and vulnerability scans.

- Build toolchain
  - Docker Buildx with separate builder instance
  - Inline cache (`type=registry`) to speed subsequent builds
  - Secrets via `--secret id=...` (no ARG secrets)
  - Platforms: `linux/amd64,linux/arm64`

- Example commands
  - Create builder: `docker buildx create --use --name sutazai-builder` (once)
  - Build & push multi-arch:
    - `docker buildx build --platform linux/amd64,linux/arm64 -t registry/sutazai/backend:${GIT_SHA} -f docker/backend/Dockerfile --cache-to type=registry,ref=registry/sutazai/cache:backend,mode=max --cache-from type=registry,ref=registry/sutazai/cache:backend --secret id=pip_index_url,src=.secrets/pip_index_url.txt --push .`
  - SBOM:
    - `docker buildx build ... --sbom=true ...`

- GitHub Actions (snippet)
  - Setup QEMU + Buildx
  - Login to registry (OIDC or PAT)
  - Reuse cache refs per image
  - Example step:
    - uses: docker/setup-buildx-action@v3
    - uses: docker/build-push-action@v6
      with:
        context: .
        file: docker/backend/Dockerfile
        platforms: linux/amd64,linux/arm64
        push: true
        tags: registry/sutazai/backend:${{ github.sha }}
        cache-from: type=registry,ref=registry/sutazai/cache:backend
        cache-to: type=registry,ref=registry/sutazai/cache:backend,mode=max
        secrets: |
          pip_index_url=${{ secrets.PIP_INDEX_URL }}
        sbom: true

- Release policy
  - Tagging: `:sha` for immutability, `:main` for rolling, `:vX.Y.Z` for releases
  - Provenance: attach SBOM, store `cosign` attestations (future)

- References
  - Buildx deep dive, secure secret handling, and caching strategies
