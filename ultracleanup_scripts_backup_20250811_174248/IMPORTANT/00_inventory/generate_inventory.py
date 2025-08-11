#!/usr/bin/env python3
import os
import csv
import json
import hashlib
import subprocess
import datetime
import shutil
import gzip
import bz2
import lzma
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

ROOT = Path('/opt/sutazaiapp/IMPORTANT').resolve()
OUT_DIR = ROOT / '00_inventory'
EXTRACT_DIR = OUT_DIR / '_extracted'

ARCHIVE_EXTS = {
    '.zip': 'zip',
    '.tar': 'tar',
    '.tgz': 'tar',
    '.tar.gz': 'tar',
    '.tar.bz2': 'tar',
    '.tar.xz': 'tar',
    '.gz': 'gz',
    '.bz2': 'bz2',
    '.xz': 'xz',
    '.rar': 'rar',
    '.7z': '7z',
}

DOC_EXTS = {'.md', '.txt', '.rst', '.adoc', '.puml', '.mmd', '.drawio', '.csv', '.xlsx', '.pdf', '.docx'}

# Exclude generated dirs from scanning to avoid recursion
EXCLUDE_DIRS = {str((ROOT / p).resolve()) for p in [
    '00_inventory', '01_findings', '02_issues', '10_canonical', '20_plan', '99_appendix'
]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_provenance(path: Path) -> Dict[str, Optional[str]]:
    try:
        rel = str(path.relative_to(Path('/opt/sutazaiapp')))
    except ValueError:
        rel = str(path)
    try:
        out = subprocess.check_output([
            'git', '-C', '/opt/sutazaiapp', 'log', '-1', '--format=%H|%an|%ad', '--', rel
        ], stderr=subprocess.DEVNULL, text=True).strip()
        if out:
            commit, author, date = out.split('|', 2)
            return {'commit': commit, 'author': author, 'date': date}
    except Exception:
        pass
    return {'commit': None, 'author': None, 'date': None}


def file_type(path: Path) -> str:
    ext = path.suffix.lower()
    # Handle double extensions for tar.*
    if path.name.endswith('.tar.gz'):
        return 'tar.gz'
    if path.name.endswith('.tar.bz2'):
        return 'tar.bz2'
    if path.name.endswith('.tar.xz'):
        return 'tar.xz'
    return ext.lstrip('.') or 'unknown'


def list_files() -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune excluded dirs
        if os.path.abspath(dirpath) in EXCLUDE_DIRS:
            dirnames[:] = []
            continue
        for name in filenames:
            files.append(Path(dirpath) / name)
    return files


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    for ext in ARCHIVE_EXTS:
        if name.endswith(ext):
            return True
    return False


def archive_kind(path: Path) -> str:
    name = path.name.lower()
    for ext, kind in ARCHIVE_EXTS.items():
        if name.endswith(ext):
            return kind
    return 'unknown'


def extract_archive(path: Path, dest_root: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    extracted: List[Dict[str, Any]] = []
    dest_dir = dest_root / path.stem
    # Disambiguate for tar double extensions
    if path.name.lower().endswith(('.tar.gz', '.tar.bz2', '.tar.xz')):
        base = path.name
        dest_dir = dest_root / base.rsplit('.', 2)[0]
    dest_dir.mkdir(parents=True, exist_ok=True)
    kind = archive_kind(path)
    try:
        if kind == 'zip':
            with zipfile.ZipFile(path, 'r') as zf:
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    target = dest_dir / zi.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(zi) as src, open(target, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append({
                        'path': str(target.relative_to(ROOT)),
                        'size': target.stat().st_size,
                        'sha256': sha256_file(target)
                    })
        elif kind == 'tar':
            mode = 'r:*'
            with tarfile.open(path, mode) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    tf.extract(member, dest_dir)
                    target = dest_dir / member.name
                    extracted.append({
                        'path': str(target.relative_to(ROOT)),
                        'size': target.stat().st_size,
                        'sha256': sha256_file(target)
                    })
        elif kind in ('gz', 'bz2', 'xz'):
            # Single-file compression. Decompress to dest_dir/base
            base_name = path.name
            for suf in ('.gz', '.bz2', '.xz'):
                if base_name.endswith(suf):
                    base_name = base_name[:-len(suf)]
                    break
            target = dest_dir / base_name
            opener = gzip.open if kind == 'gz' else bz2.open if kind == 'bz2' else lzma.open
            with opener(path, 'rb') as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted.append({
                'path': str(target.relative_to(ROOT)),
                'size': target.stat().st_size,
                'sha256': sha256_file(target)
            })
        else:
    except Exception as e:


def load_text_for_review(path: Path) -> Tuple[List[str], Optional[str]]:
    ext = path.suffix.lower()
    try:
        if ext in {'.md', '.txt', '.rst', '.adoc', '.puml', '.mmd', '.csv'}:
            text = path.read_text(encoding='utf-8', errors='replace')
            return text.splitlines(), None
        # For drawio (XML), treat as text
        if ext == '.drawio':
            text = path.read_text(encoding='utf-8', errors='replace')
            return text.splitlines(), None
        # For binaries we don't parse here
        return [], f'Extraction not implemented for {ext}'
    except Exception as e:
        return [], f'Error reading file: {e}'


def generate_inventory():
    files = list_files()
    inventory: List[Dict[str, Any]] = []
    archives_audit: Dict[str, Any] = {}

    # Duplicate detection helpers
    checksum_to_paths: Dict[str, List[str]] = {}
    name_to_variants: Dict[str, List[Path]] = {}

    for f in files:
        try:
            st = f.stat()
            checksum = sha256_file(f)
        except Exception:
            # Skip unreadable files
            continue
        prov = git_provenance(f)
        rel = str(f.relative_to(ROOT))
        info = {
            'path': rel,
            'abs_path': str(f),
            'file_type': file_type(f),
            'ext': f.suffix.lower(),
            'last_modified': datetime.datetime.fromtimestamp(st.st_mtime).isoformat(),
            'size_bytes': st.st_size,
            'sha256': checksum,
            'git_commit': prov.get('commit'),
            'git_author': prov.get('author'),
            'git_date': prov.get('date'),
        }
        inventory.append(info)
        checksum_to_paths.setdefault(checksum, []).append(rel)
        name_to_variants.setdefault(f.name.lower(), []).append(f)

        if is_archive(f):
            archives_audit[str(f.relative_to(ROOT))] = {
                'extracted_files': extracted,
            }

    # Write inventory.json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    (OUT_DIR / 'archives_audit.json').write_text(json.dumps(archives_audit, indent=2), encoding='utf-8')

    # Build inventory.md with duplicates and drift summary
    lines: List[str] = []
    lines.append(f"# Inventory for {ROOT}\n")
    lines.append(f"Generated: {datetime.datetime.utcnow().isoformat()}Z\n")
    lines.append("\n## Files\n")
    lines.append("| Path | Type | Modified | Size (bytes) | SHA256 | Author | Commit |\n")
    lines.append("|---|---|---:|---:|---|---|---|\n")
    for item in sorted(inventory, key=lambda x: x['path']):
        lines.append(f"| {item['path']} | {item['file_type']} | {item['last_modified']} | {item['size_bytes']} | {item['sha256'][:12]} | {item['git_author'] or ''} | {(item['git_commit'] or '')[:12]} |")

    # Duplicates
    exact_dups = {h: p for h, p in checksum_to_paths.items() if len(p) > 1}
    if exact_dups:
        lines.append("\n## Exact Duplicates\n")
        for h, paths in exact_dups.items():
            lines.append(f"- Hash {h}:\n  - " + "\n  - ".join(paths))

    # Near-duplicates by name
    lines.append("\n## Potential Version Drift (same filenames)\n")
    for name, variants in name_to_variants.items():
        if len(variants) > 1:
            # Summarize sizes and checksums
            lines.append(f"- {name}:\n" + "\n".join([f"  - {str(v.relative_to(ROOT))} ({v.stat().st_size} bytes, {sha256_file(v)[:12]})" for v in variants]))

    (OUT_DIR / 'inventory.md').write_text("\n".join(lines), encoding='utf-8')

    # Build doc_review_matrix.csv
    matrix_path = OUT_DIR / 'doc_review_matrix.csv'
    with open(matrix_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for f in files:
            if f.suffix.lower() not in DOC_EXTS:
                continue
            rel = str(f.relative_to(ROOT))
            if content_lines:
                block = 100
                total = len(content_lines)
                for i in range(0, total, block):
                    start = i + 1
                    end = min(i + block, total)
                    status = '⚠️unclear'
            else:
                # Single block placeholder when extraction missing

if __name__ == '__main__':
    generate_inventory()
