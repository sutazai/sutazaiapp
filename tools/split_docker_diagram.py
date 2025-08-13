#!/usr/bin/env python3
import sys
from pathlib import Path
import hashlib

ROOT = Path("/opt/sutazaiapp")
SRC = ROOT / "Dockerdiagramdraft.md"
OUT_DIR = ROOT / "docs" / "diagrams"
FINAL = OUT_DIR / "Dockerdiagram.md"

MARKER = "/docker/"

def split_blocks(text: str):
    lines = text.splitlines()
    blocks = []
    current = []
    in_block = False
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped == MARKER:
            if in_block and current:
                blocks.append("\n".join(current).rstrip() + "\n")
                current = []
            in_block = True
            current.append(line_stripped)
        else:
            if in_block:
                current.append(line)
    if in_block and current:
        blocks.append("\n".join(current).rstrip() + "\n")
    return blocks

def unique_blocks(blocks):
    seen = set()
    unique = []
    for b in blocks:
        h = hashlib.sha256(b.encode("utf-8", errors="ignore")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(b)
    return unique

def main():
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    if not src_path.exists():
        print(f"Source not found: {src_path}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    content = src_path.read_text(encoding="utf-8", errors="ignore")
    blocks = split_blocks(content)
    if not blocks:
        print("No blocks detected with marker '/docker/'.")
        sys.exit(2)

    uniq = unique_blocks(blocks)

    # Expect at least: base, enhanced-training, self-coding
    labels = [
        ("Dockerdiagram-core.md", "Part 1 — Core (Base)"),
        ("Dockerdiagram-training.md", "Part 2 — Enhanced (Training)"),
        ("Dockerdiagram-self-coding.md", "Part 3 — Ultimate (Self-Coding + UltraThink)"),
    ]

    written = []
    for i, (filename, title) in enumerate(labels):
        if i < len(uniq):
            path = OUT_DIR / filename
            # Ensure each part starts with a title banner for clarity
            banner = f"# {title}\n\n" \
                     f"<!-- Auto-generated from Dockerdiagramdraft.md by tools/split_docker_diagram.py -->\n\n"
            path.write_text(banner + uniq[i], encoding="utf-8")
            written.append((title, path))

    # Build consolidated final if we have at least one part
    if written:
        toc = ["# Master Docker Ecosystem Diagram (Consolidated)",
               "", "Generated from Dockerdiagramdraft.md via tools/split_docker_diagram.py.",
               "", "Contents:"]
        for title, path in written:
            toc.append(f"- {title} — {path.name}")
        toc.append("")

        final_parts = []
        for title, path in written:
            final_parts.append(f"\n\n---\n\n# {title}\n\n")
            final_parts.append(path.read_text(encoding="utf-8"))
        FINAL.write_text("\n".join(toc) + "\n" + "".join(final_parts), encoding="utf-8")

    print("Created:")
    for title, path in written:
        print(f"- {title}: {path}")
    if written:
        print(f"- Consolidated: {FINAL}")

if __name__ == "__main__":
    main()

