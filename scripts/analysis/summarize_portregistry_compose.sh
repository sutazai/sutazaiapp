#!/usr/bin/env bash
# Purpose: Summarize mismatches between IMPORTANT/diagrams/PortRegistry.md and docker-compose exposed host ports
# Output: /reports/cleanup/port_registry_mismatch_summary_<timestamp>.md
# Safety: Read-only analysis

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
PORT_REGISTRY_MD="${PROJECT_ROOT}/IMPORTANT/diagrams/PortRegistry.md"
REPORTS_DIR="${PROJECT_ROOT}/reports/cleanup"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S_UTC)"
OUT_MD="${REPORTS_DIR}/port_registry_mismatch_summary_${TIMESTAMP}.md"

mkdir -p "${REPORTS_DIR}"

# Collect declared ports from PortRegistry.md
# Pattern: lines like "- 10010: ..."
mapfile -t DECLARED < <(grep -E "^- +[0-9]{4,5}:" -n "${PORT_REGISTRY_MD}" | sed -E 's/^- +([0-9]{4,5}):.*/\1/' | sort -n | uniq)

# Collect exposed host ports from compose files
mapfile -t COMPOSE_FILES < <(find "${PROJECT_ROOT}" -type f -regextype posix-extended -regex ".*(/|^)compose.*\.ya?ml$|.*(/|^)docker-compose.*\.ya?ml$" | sort)

COMPOSE_PORTS=()
for f in "${COMPOSE_FILES[@]}"; do
  # Extract host:container pairs; capture host port (left of colon)
  while IFS= read -r line; do
    host=$(echo "$line" | sed -E 's/^[^0-9]*([0-9]{2,5}):[0-9]{2,5}.*/\1/')
    [[ -n "$host" ]] && COMPOSE_PORTS+=("$host") || true
  done < <(grep -E "^[[:space:]]*- [\"']?[0-9]{2,5}:[0-9]{2,5}[\"']?" -n "$f" || true)
  # Also catch long-form mapping like "- target: 7474\n  published: 10002"
  while IFS= read -r pub; do
    port=$(echo "$pub" | sed -E 's/.*published:[[:space:]]*([0-9]{2,5}).*/\1/')
    [[ -n "$port" ]] && COMPOSE_PORTS+=("$port") || true
  done < <(awk 'tolower($0) ~ /published:[[:space:]]*[0-9]{2,5}/ {print}' "$f" || true)

done

# Unique sorted compose ports
mapfile -t EXPOSED < <(printf "%s\n" "${COMPOSE_PORTS[@]:-}" | grep -E "^[0-9]+$" | sort -n | uniq)

# Build sets
DECL_FILE=$(mktemp); EXP_FILE=$(mktemp)
printf "%s\n" "${DECLARED[@]:-}" > "$DECL_FILE"
printf "%s\n" "${EXPOSED[@]:-}" > "$EXP_FILE"

# Compute set differences
MISSING_IN_COMPOSE=$(comm -23 "$DECL_FILE" "$EXP_FILE" || true)
NOT_IN_REGISTRY=$(comm -13 "$DECL_FILE" "$EXP_FILE" || true)
INTERSECTION=$(comm -12 "$DECL_FILE" "$EXP_FILE" || true)

{
  echo "# Port Registry Mismatch Summary"
  echo "Generated: ${TIMESTAMP}"
  echo
  echo "## Inputs"
  echo "- Registry: ${PORT_REGISTRY_MD}"
  echo "- Compose files:"
  for f in "${COMPOSE_FILES[@]:-}"; do echo "  - $f"; done
  echo
  echo "## Declared in PortRegistry.md (ports)"
  if [[ ${#DECLARED[@]:-0} -gt 0 ]]; then printf "  - %s\n" "${DECLARED[@]}"; else echo "  - (none)"; fi
  echo
  echo "## Exposed in compose (host ports)"
  if [[ ${#EXPOSED[@]:-0} -gt 0 ]]; then printf "  - %s\n" "${EXPOSED[@]}"; else echo "  - (none)"; fi
  echo
  echo "## Mismatches"
  echo "### Declared in Registry but NOT exposed in compose"
  if [[ -n "$MISSING_IN_COMPOSE" ]]; then echo "$MISSING_IN_COMPOSE" | sed 's/^/  - /'; else echo "  - (none)"; fi
  echo
  echo "### Exposed in compose but NOT declared in Registry"
  if [[ -n "$NOT_IN_REGISTRY" ]]; then echo "$NOT_IN_REGISTRY" | sed 's/^/  - /'; else echo "  - (none)"; fi
  echo
  echo "### Consistent (present in both)"
  if [[ -n "$INTERSECTION" ]]; then echo "$INTERSECTION" | sed 's/^/  - /'; else echo "  - (none)"; fi
  echo
  echo "## Notes"
  echo "- Static analysis only; does not validate runtime health."
  echo "- Before changing Registry or compose, attach this report to the CHANGELOG and cross-check with IMPORTANT/diagrams."
} > "$OUT_MD"

echo "Report written to: ${OUT_MD}"