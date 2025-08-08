# ISSUE-0013 — Diagram PNG Rendering Pipeline Missing
**Impacted components:** Documentation CI, Canonical diagrams consumers
**Context:** Canonical docs contain Mermaid (.mmd) diagrams but no rendered PNGs; quality gate requires both.
**Options:**
- **A:** Add CI step using `@mermaid-js/mermaid-cli` in Node container to render .mmd → .png (recommended) — Pros: reproducible, cross-platform; Cons: adds Node toolchain.
- **B:** Use Dockerized mermaid-cli (`minlag/mermaid-cli`) — Pros: no host Node; Cons: heavier image, cache management.
- **C:** GitHub Action/3rd-party renderer — Pros: turnkey; Cons: external dependency, runner constraints.
**Recommendation:** A — Add mermaid-cli to CI with deterministic version pinning and cache.
**Consequences:** Introduces CI job, requires checking .mmd compile, commits .png alongside .mmd.
**Evidence:** [source] /opt/sutazaiapp/IMPORTANT/10_canonical/INDEX.md#L1-L60; [source] /opt/sutazaiapp/IMPORTANT/10_canonical/current_state/context.mmd#L1-L80
