#!/usr/bin/env bash
set -euo pipefail
readonly OPP_WORKSPACE="${SAFEBENEFIT_OPP_WORKSPACE:-/home/opp_env/default_workspace}"
readonly OPP_ENV="${SAFEBENEFIT_OPP_ENV:-/home/opp_env/.venv/bin/opp_env}"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly RESULT_ROOT="${SAFEBENEFIT_OUTPUT_ROOT:-${PROJECT_ROOT}/output/current}"
export PATH="/home/opp_env/.nix-profile/bin:/home/opp_env/.venv/bin:${PATH}"
"${OPP_ENV}" run omnetpp-6.4.0 inet-4.7.0 -w "${OPP_WORKSPACE}" \
  -c "cd '${PROJECT_ROOT}/simulator/omnetpp/graph_query' && '${OPP_WORKSPACE}/omnetpp-6.4.0/bin/opp_makemake' -f --make-so -o graph_query -I'${OPP_WORKSPACE}/inet-4.7.0/src' -L'${OPP_WORKSPACE}/inet-4.7.0/src' -lINET && make -j2 && mkdir -p '${RESULT_ROOT}/cache/build' && clang++ -std=c++17 -I. '${PROJECT_ROOT}/tests/cpp/reference_test.cc' -o '${RESULT_ROOT}/cache/build/reference_test' && '${RESULT_ROOT}/cache/build/reference_test' '${RESULT_ROOT}/cache/build/reference_test_files'"
mkdir -p "${RESULT_ROOT}/cache/build"
cd "${PROJECT_ROOT}"
sha256sum simulator/omnetpp/graph_query/GraphQueryApp.cc simulator/omnetpp/graph_query/ProjectionOperators.h simulator/omnetpp/graph_query/OperationCounters.h simulator/omnetpp/graph_query/SpillStore.h simulator/omnetpp/graph_query/GraphQuery.ned simulator/omnetpp/graph_query/omnetpp.ini simulator/omnetpp/graph_query/libgraph_query.so > "${RESULT_ROOT}/cache/build/build.sha256"
