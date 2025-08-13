#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$ROOT_DIR/.claude/agents"

missing_header=()
missing_routing=()
missing_model=()
banned_terms=()

if [[ ! -d "$AGENTS_DIR" ]]; then
  echo "Agents directory not found: $AGENTS_DIR" >&2
  exit 2
fi

# Scan all agent markdown files
while IFS= read -r -d '' file; do
  rel="${file#$ROOT_DIR/}"

  # 1) 20-rule header
  if ! grep -q "YOU ARE BOUND BY THE FOLLOWING 20" "$file"; then
    missing_header+=("$rel")
  fi

  # 2) Specialist routing block
  if ! grep -Eq "Specialist Agent Routing Matrix|Ultra Execution Protocol" "$file"; then
    missing_routing+=("$rel")
  fi

  # 3) Front matter model field present (between first two ---)
  if awk 'BEGIN{inFM=0; has=0}
           /^---/{inFM++; if(inFM==2) exit; next}
           { if(inFM==1 && $0 ~ /model:/) has=1 }
           END{ if(has==1) exit 0; else exit 1 }' "$file"; then
    :
  else
    missing_model+=("$rel")
  fi

  # 4) Banned/weak terms
  if grep -Eqi '\bencapsulated\b|\bfake\b|\bplaceholder\b' "$file"; then
    banned_terms+=("$rel")
  fi
done < <(find "$AGENTS_DIR" -type f -name '*.md' -print0 | sort -z)

status=0
report() {
  local title="$1"; shift
  local -n arr=$1
  if (( ${#arr[@]} > 0 )); then
    echo "\n$title (${#arr[@]}):"; printf ' - %s\n' "${arr[@]}"
    status=1
  fi
}

report "Missing 20-rule header" missing_header
report "Missing Specialist routing block" missing_routing
report "Missing model in front matter" missing_model
report "Contains banned/weak terms" banned_terms

if (( status == 0 )); then
  echo "All agents pass compliance checks."
else
  echo "\nCompliance check failed." >&2
fi

exit "$status"

