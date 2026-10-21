#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly RESULT_ROOT="${SAFEBENEFIT_OUTPUT_ROOT:-${PROJECT_ROOT}/output/current}"
readonly OPP_WORKSPACE="${SAFEBENEFIT_OPP_WORKSPACE:-/home/opp_env/default_workspace}"
readonly OPP_ENV="${SAFEBENEFIT_OPP_ENV:-/home/opp_env/.venv/bin/opp_env}"
export PATH="/home/opp_env/.nix-profile/bin:/home/opp_env/.venv/bin:${PATH}"

batch_id="${1:?batch ID required}"
[[ "$batch_id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
readonly MANIFEST="${RESULT_ROOT}/cache/batches/${batch_id}.tsv"
test -s "$MANIFEST"

"$OPP_ENV" run omnetpp-6.4.0 inet-4.7.0 \
  -w "$OPP_WORKSPACE" \
  -c "SAFEBENEFIT_OUTPUT_ROOT='${RESULT_ROOT}' SAFEBENEFIT_OPP_WORKSPACE='${OPP_WORKSPACE}' SAFEBENEFIT_OPP_ENV='${OPP_ENV}' SAFEBENEFIT_ENV_READY=1 bash '${SCRIPT_DIR}/run_batch_inner.sh' '${MANIFEST}'"
