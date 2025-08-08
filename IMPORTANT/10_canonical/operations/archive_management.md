# Archive Scanning & Management

- Scope
  - Supported: zip, tar, tgz, tar.gz, tar.bz2, tar.xz, gz, bz2, xz, rar, 7z
  - Parse: md, txt, rst, adoc, docx, pdf, puml, mmd, drawio, csv, xlsx

- Process
  - Automated extraction (read-only) into `00_inventory/_extracted/`
  - Compute checksums and provenance; log to `archives_audit.json`
  - OCR for PDFs (if image-based)
  - Line-by-line coverage matrix (100%) in `doc_review_matrix.csv`

- AI assistance
  - Classify docs, detect duplicates/drift, suggest canonicalization
  - Generate Issue Cards for conflicts

- Deep archive migration
  - Stage cold data to deep storage with manifest and checksums
  - Verify integrity on retrieval; maintain retention policies

- Operations
  - Scheduled scan job; diffs stored in `/reports/archive-scans/`
  - Alerts on new conflicts or drift
