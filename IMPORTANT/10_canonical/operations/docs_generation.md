# Documentation Generation (Multi-Platform)

- Sources
  - Python backend/services: Sphinx + autodoc; pdoc for lightweight modules
  - TypeScript/JS frontend: TypeDoc + JSDoc
  - OpenAPI: generate via FastAPI schema export; publish under `/docs/api`
  - Diagrams: Mermaid (.mmd) stored in repo; CI renders PNGs

- Multi-platform targets
  - Developer site (HTML): versioned with `gh-pages` or artifact hosting
  - PDF bundles: `pandoc` from canonical `.md`
  - CLI help extraction: `--help` scraped into docs

- CI steps
  - Build symbol graphs (Python stubs via `sphinx-apidoc`, TS via `typedoc`)
  - Render Mermaid to PNG using `@mermaid-js/mermaid-cli`
  - Validate links and anchors
  - Publish artifacts on release tags

- Principles
  - Docs-first: canonical truth in `/IMPORTANT/10_canonical/`
  - No TBD: unresolved items get Issue Cards
  - Every change updates docs in same PR
