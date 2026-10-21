#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[case] failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly RESULT_ROOT="${SAFEBENEFIT_OUTPUT_ROOT:-${PROJECT_ROOT}/output/current}"
readonly OPP_WORKSPACE="${SAFEBENEFIT_OPP_WORKSPACE:-/home/opp_env/default_workspace}"
readonly OPP_ENV="${SAFEBENEFIT_OPP_ENV:-/home/opp_env/.venv/bin/opp_env}"
export PATH="/home/opp_env/.nix-profile/bin:/home/opp_env/.venv/bin:${PATH}"

dataset="${1:?dataset required}"
split="${2:?split required}"
method="${3:?method required}"
scenario="${4:?scenario required}"
initial="${5:?initial plan or registered required}"
run_id="${6:?run ID required}"
benefit_margin="${7:?benefit safety margin required}"
estimated_switch_wire="${8:?estimated switch wire bytes required}"
retry_interval="${9:?retry interval seconds required}"
worker_count="${10:-4}"
link_datarate="${11:-1Gbps}"
link_delay="${12:-50us}"
query_filename="${13:-queries.tsv}"
uncertainty_z="${14:-1.96}"
prediction_relative_std="${15:-0.05}"
scan_cpu_rate="${16:-10000000}"
codec_cpu_rate="${17:-10000000}"
io_rate="${18:-1000000000}"

[[ "$dataset" =~ ^(ldbc_snb_i1_v100_sf01|ldbc_snb_i1_v100_sf1|ldbc_snb_i1_v100_sf10)$ ]] || exit 2
[[ "$split" =~ ^(calibration|evaluation)$ ]] || exit 2
[[ "$method" =~ ^(STATIC|SWITCH_COST_THRESHOLD|LINUCB|SAFE_BENEFIT)$ ]] || exit 2
[[ "$scenario" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]] || exit 2
[[ "$initial" =~ ^(0|1|2|registered)$ ]] || exit 2
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$benefit_margin" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] || exit 2
[[ "$estimated_switch_wire" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
[[ "$retry_interval" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
[[ "$worker_count" =~ ^(1|2|4|8|16)$ ]] || exit 2
[[ "$link_datarate" =~ ^(100Mbps|1Gbps|10Gbps)$ ]] || exit 2
[[ "$link_delay" =~ ^(10us|50us|1ms|5ms)$ ]] || exit 2
[[ "$query_filename" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
[[ "$uncertainty_z" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
[[ "$prediction_relative_std" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] || exit 2
[[ "$scan_cpu_rate" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
[[ "$codec_cpu_rate" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
[[ "$io_rate" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2

readonly INPUT_ROOT="${RESULT_ROOT}/cache/materialized/${split}/${dataset}"
readonly OUTPUT="${RESULT_ROOT}/simulation/${run_id}"
readonly EVALUATION_POLICY="${RESULT_ROOT}/policies/candidate_policy.tsv"
readonly SWITCH_THRESHOLD_MODEL="${RESULT_ROOT}/policies/switch_threshold.tsv"
readonly INITIAL_PLANS="${RESULT_ROOT}/policies/initial_plans.tsv"
test -f "${RESULT_ROOT}/cache/build/build.sha256"
cd "$PROJECT_ROOT"
sha256sum --check --quiet "${RESULT_ROOT}/cache/build/build.sha256"
test -f "${INPUT_ROOT}/${query_filename}"
test -d "${INPUT_ROOT}"
if [[ "$method" == LINUCB ]]; then test -f "$EVALUATION_POLICY"; fi
if [[ "$method" == SWITCH_COST_THRESHOLD ]]; then test -f "$SWITCH_THRESHOLD_MODEL"; fi
if [[ "$initial" == registered ]]; then test -f "$INITIAL_PLANS"; fi
mkdir -p "$OUTPUT"
# An earlier OMNeT++ run may have stopped at sim-time-limit with exit code 0.
# Invalidate its marker before overwriting outputs so a failed rerun cannot
# masquerade as a completed case.
: > "${OUTPUT}/.complete"

initial_value="$initial"
initial_file=""
if [[ "$initial" == registered ]]; then initial_value=0; initial_file="$INITIAL_PLANS"; fi
policy_file=""
if [[ "$method" == LINUCB ]]; then policy_file="$EVALUATION_POLICY"; fi
switch_threshold_file=""
if [[ "$method" == SWITCH_COST_THRESHOLD ]]; then switch_threshold_file="$SWITCH_THRESHOLD_MODEL"; fi

RUN_COMMAND="cd '${PROJECT_ROOT}/simulator/omnetpp/graph_query' && "
RUN_COMMAND+="opp_run -u Cmdenv -n '.:${OPP_WORKSPACE}/inet-4.7.0/src' "
RUN_COMMAND+="-l '${OPP_WORKSPACE}/inet-4.7.0/src/INET' -l ./graph_query "
RUN_COMMAND+="-c '${scenario}' -r 0 --*.workers=${worker_count} "
RUN_COMMAND+="--*.linkDatarate=${link_datarate} --*.linkDelay=${link_delay} "
RUN_COMMAND+="--*.coordinator.app[0].workerCount=${worker_count} --*.worker[*].app[0].workerCount=${worker_count} "
RUN_COMMAND+="'--**.app[0].method=\"${method}\"' '--**.app[0].initialPlan=${initial_value}' "
RUN_COMMAND+="'--**.app[0].benefitSafetyMargin=${benefit_margin}' "
RUN_COMMAND+="'--**.app[0].uncertaintyZ=${uncertainty_z}' "
RUN_COMMAND+="'--**.app[0].predictionRelativeStd=${prediction_relative_std}' "
RUN_COMMAND+="'--**.app[0].scanCpuUnitsPerSecond=${scan_cpu_rate}' "
RUN_COMMAND+="'--**.app[0].codecCpuUnitsPerSecond=${codec_cpu_rate}' "
RUN_COMMAND+="'--**.app[0].ioBytesPerSecond=${io_rate}' "
RUN_COMMAND+="'--**.app[0].estimatedSwitchWireBytes=${estimated_switch_wire}' "
RUN_COMMAND+="'--**.app[0].retryInterval=${retry_interval}s' "
RUN_COMMAND+="'--**.app[0].partitionCount=8' "
RUN_COMMAND+="'--**.app[0].spoolDirectory=\"${OUTPUT}/spool\"' "
RUN_COMMAND+="'--**.app[0].inputDirectory=\"${INPUT_ROOT}\"' "
RUN_COMMAND+="'--**.app[0].materializedDirectory=\"${INPUT_ROOT}\"' "
RUN_COMMAND+="'--**.app[0].queryFile=\"${INPUT_ROOT}/${query_filename}\"' "
RUN_COMMAND+="'--**.app[0].policyFile=\"${policy_file}\"' "
RUN_COMMAND+="'--**.app[0].switchThresholdFile=\"${switch_threshold_file}\"' "
RUN_COMMAND+="'--**.app[0].initialPlanFile=\"${initial_file}\"' "
RUN_COMMAND+="'--*.coordinator.app[0].outputFile=\"${OUTPUT}/queries.jsonl\"' "
RUN_COMMAND+="'--output-scalar-file=${OUTPUT}/metrics.sca' omnetpp.ini"

if [[ "${SAFEBENEFIT_ENV_READY:-0}" == 1 ]]; then
  bash -c "$RUN_COMMAND"
else
  "$OPP_ENV" run omnetpp-6.4.0 inet-4.7.0 -w "$OPP_WORKSPACE" -c "$RUN_COMMAND"
fi
expected_queries="$(wc -l < "${INPUT_ROOT}/${query_filename}")"
actual_queries="$(wc -l < "${OUTPUT}/queries.jsonl")"
if [[ "$actual_queries" -ne "$expected_queries" ]]; then
  echo "incomplete simulation case ${run_id}: ${actual_queries}/${expected_queries} queries" >&2
  exit 1
fi
printf 'PASS\n' > "${OUTPUT}/.complete"
