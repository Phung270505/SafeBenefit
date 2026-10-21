#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MANIFEST="${1:?batch manifest required}"

test -f "$MANIFEST"
total="$(awk 'NF { count += 1 } END { print count + 0 }' "$MANIFEST")"
index=0

while IFS=$'\t' read -r -a fields; do
  [[ ${#fields[@]} -eq 18 ]] || {
    echo "invalid batch row: expected 18 fields, got ${#fields[@]}" >&2
    exit 2
  }
  # Accept an existing manifest produced by Windows text-mode output. New
  # manifests use LF, but stripping CR here makes resume backward-compatible.
  fields[17]="${fields[17]%$'\r'}"
  index=$((index + 1))
  echo "[batch] START ${index}/${total} ${fields[5]}"
  SAFEBENEFIT_ENV_READY=1 bash "${SCRIPT_DIR}/run_case.sh" "${fields[@]}"
  echo "[batch] DONE ${index}/${total} ${fields[5]}"
done < "$MANIFEST"

[[ "$index" -eq "$total" ]]
