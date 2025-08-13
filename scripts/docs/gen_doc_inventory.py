#!/usr/bin/env python3
"""
Generate Phase 1 documentation inventory for /opt/sutazaiapp/IMPORTANT.

Outputs under IMPORTANT/00_inventory/:
 - inventory.json: file metadata (path, type, size, mtime, sha256)
 - inventory.md: human-readable table summary
 - archives_audit.json: per-archive listing with entry checksums where supported
 - doc_review_matrix.csv: line coverage matrix for text docs (100-line chunks)

Usage:
  python3 scripts/docs/gen_doc_inventory.py \
    --root /opt/sutazaiapp/IMPORTANT \
    --out /opt/sutazaiapp/IMPORTANT/00_inventory

Notes:
 - Respects the codebase rule to derive facts from IMPORTANT as source of truth.
 - No network access; uses stdlib tarfile/zipfile for archive introspection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import tarfile
import zipfile


TEXT_EXTS = {".md", ".txt", ".rst", ".adoc"}
ARCHIVE_EXTS = {
    ".zip",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".tbz2",
    ".tar.bz2",
    ".txz",
    ".tar.xz",
    ".gz",
    ".bz2",
    ".xz",
    ".rar",
    ".7z",
}
NON_TEXT_DOC_EXTS = {".docx", ".pdf", ".puml", ".mmd", ".drawio", ".csv", ".xlsx"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso8601(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def list_zip_entries(archive_path: Path) -> List[Dict[str, object]]:
    entries = []
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for zi in zf.infolist():
                if zi.is_dir():
                    continue
                # Compute SHA256 of file content without extracting to disk
                h = hashlib.sha256()
                with zf.open(zi, "r") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                entries.append(
                    {
                        "name": zi.filename,
                        "size": zi.file_size,
                        "sha256": h.hexdigest(),
                    }
                )
    except zipfile.BadZipFile:
        entries.append({"error": "BadZipFile"})
    return entries


def list_tar_entries(archive_path: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    try:
        with tarfile.open(archive_path, mode="r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                h = hashlib.sha256()
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
                entries.append({"name": m.name, "size": m.size, "sha256": h.hexdigest()})
    except tarfile.TarError as e:
        entries.append({"error": f"TarError: {e}"})
    return entries


def list_single_compressed(archive_path: Path) -> List[Dict[str, object]]:
    # Without knowing original filename, record checksum of decompressed stream if possible.
    # For simplicity and portability, just record the archive checksum here.
    return [{"note": "single-file compressed (no listing)", "sha256_archive": sha256_file(archive_path)}]


def audit_archive(archive_path: Path) -> Dict[str, object]:
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        entries = list_zip_entries(archive_path)
    elif name.endswith((".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz")):
        entries = list_tar_entries(archive_path)
    elif name.endswith((".gz", ".bz2", ".xz")):
        entries = list_single_compressed(archive_path)
    else:
        entries = [
            {
                "note": "unsupported archive type for introspection",
                "sha256_archive": sha256_file(archive_path),
            }
        ]
    return {"entries": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate IMPORTANT docs inventory")
    parser.add_argument("--root", type=str, default="/opt/sutazaiapp/IMPORTANT")
    parser.add_argument(
        "--out", type=str, default="/opt/sutazaiapp/IMPORTANT/00_inventory",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    inventory: List[Dict[str, object]] = []
    archives_audit: Dict[str, object] = {}

    # Walk files
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        # Skip output directory to avoid self-referencing explosion
        if outdir in [d] or str(d).startswith(str(outdir)):
            continue
        for fn in filenames:
            p = d / fn
            try:
                stat = p.stat()
                size = stat.st_size
                mtime = stat.st_mtime
            except FileNotFoundError:
                continue
            ext = "".join(p.suffixes).lower() or p.suffix.lower()
            entry: Dict[str, object] = {
                "path": rel(p, root),
                "abs_path": str(p),
                "type": ext or p.suffix.lower(),
                "size": size,
                "mtime": iso8601(mtime),
                "sha256": sha256_file(p),
            }
            inventory.append(entry)

            # Audit archives
            if any(entry["path"].lower().endswith(ext_) for ext_ in ARCHIVE_EXTS):
                rel_path = rel(p, root)
                archives_audit[rel_path] = audit_archive(p)

    # Write inventory.json
    inv_json = outdir / "inventory.json"
    with inv_json.open("w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    # Write inventory.md
    inv_md = outdir / "inventory.md"
    with inv_md.open("w", encoding="utf-8") as f:
        f.write("# IMPORTANT Inventory\n\n")
        f.write(f"Root: `{root}`\n\n")
        f.write("| Path | Type | Size (bytes) | Modified (UTC) | SHA256 |\n")
        f.write("|------|------|--------------:|----------------|--------|\n")
        for e in sorted(inventory, key=lambda x: str(x["path"])):
            f.write(
                f"| `{e['path']}` | {e['type']} | {e['size']} | {e['mtime']} | `{str(e['sha256'])[:12]}…` |\n"
            )

    # Write archives_audit.json
    arc_json = outdir / "archives_audit.json"
    with arc_json.open("w", encoding="utf-8") as f:
        json.dump(archives_audit, f, indent=2, ensure_ascii=False)

    # Write doc_review_matrix.csv
    drm_csv = outdir / "doc_review_matrix.csv"
    with drm_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path",
            "from_line",
            "to_line",
            "status",
            "notes",
            "source_of_truth_ref",
        ])
        for e in inventory:
            p = Path(e["abs_path"])  # type: ignore[index]
            ext = p.suffix.lower()
            rel_path = rel(p, root)
            if ext in TEXT_EXTS:
                try:
                    with p.open("r", encoding="utf-8", errors="replace") as tf:
                        lines = tf.readlines()
                except Exception:
                    lines = []
                total = len(lines)
                if total == 0:
                    writer.writerow([rel_path, 1, 1, "⚠️unclear", "Empty or unreadable", rel_path])
                else:
                    start = 1
                    while start <= total:
                        end = min(start + 99, total)
                        writer.writerow(
                            [
                                rel_path,
                                start,
                                end,
                                "⚠️unclear",
                                "Initial scan; pending reconciliation",
                                rel_path,
                            ]
                        )
                        start = end + 1
            elif ext in NON_TEXT_DOC_EXTS or any(rel_path.lower().endswith(ae) for ae in ARCHIVE_EXTS):
                writer.writerow(
                    [
                        rel_path,
                        1,
                        -1,
                        "⚠️unclear",
                        "Non-text or binary document; manual review required",
                        rel_path,
                    ]
                )
            else:
                # Non-doc files are out of scope for review matrix
                continue

    # Executive summary (brief) to help phase hand-off
    summary_md = outdir / "EXEC_SUMMARY_PHASE1.md"
    total_files = len(inventory)
    total_archives = len(archives_audit)
    text_docs = 0
    for e in inventory:
        if Path(str(e["abs_path"])) .suffix.lower() in TEXT_EXTS:  # type: ignore[index]
            text_docs += 1
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Phase 1 Executive Summary\n\n")
        f.write(f"Root: `{root}`\n\n")
        f.write(f"- Total files scanned: {total_files}\n")
        f.write(f"- Archives identified: {total_archives}\n")
        f.write(f"- Text documents (eligible for line coverage): {text_docs}\n")
        f.write(f"- Outputs: inventory.json, inventory.md, archives_audit.json, doc_review_matrix.csv\n")

    print(f"Phase 1 outputs written to: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

