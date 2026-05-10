#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NS3_ROOT="${NS3_ROOT:-${PROJECT_ROOT}/external_tools/ns3_lena/ns-3-dev}"

RAW_DIR="${PROJECT_ROOT}/data/external/ns3_lena/raw"
mkdir -p "${RAW_DIR}"

if [[ ! -d "${NS3_ROOT}" ]]; then
  echo "ERROR: NS3_ROOT does not exist: ${NS3_ROOT}" >&2
  echo "Set it explicitly, for example:" >&2
  echo "NS3_ROOT=/path/to/ns-3-dev bash simulators/ns3_lena/run_cttc_nr_demo_grid.sh" >&2
  exit 1
fi

if [[ ! -x "${NS3_ROOT}/ns3" ]]; then
  echo "ERROR: ns3 launcher not found or not executable at: ${NS3_ROOT}/ns3" >&2
  exit 1
fi

cd "${NS3_ROOT}"

rm -f "${RAW_DIR}"/cttc_nr_demo_*_load_seed_*.txt "${RAW_DIR}"/cttc_nr_demo_*_load_seed_*.console.txt

HELP_FILE="${RAW_DIR}/cttc_nr_demo_print_help.txt"
./ns3 run "cttc-nr-demo --PrintHelp" > "${HELP_FILE}" 2>&1 || true

MANIFEST="${RAW_DIR}/ns3_lena_run_manifest.csv"
printf "scenario_id,output_file,console_file,load_label,seed,gnb_count,ue_per_gnb,lambda_be,lambda_ull,packet_size_be,packet_size_ull,sim_time,command\n" > "${MANIFEST}"

csv_escape() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "${value}"
}

GNB_COUNT=1
UE_PER_GNB=3
PACKET_SIZE_BE=1252
PACKET_SIZE_ULL=100
LAMBDA_ULL=1
SIM_TIME="30000ms"

LOAD_LABELS=("low" "medium" "high")
LAMBDA_BE_VALUES=(2000 10000 30000)
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for i in "${!LOAD_LABELS[@]}"; do
  LOAD_LABEL="${LOAD_LABELS[$i]}"
  LAMBDA_BE="${LAMBDA_BE_VALUES[$i]}"

  for SEED in "${SEEDS[@]}"; do
    SCENARIO_ID="cttc_nr_demo_${LOAD_LABEL}_load_seed_${SEED}"
    OUTPUT_FILE="${SCENARIO_ID}.txt"
    CONSOLE_FILE="${SCENARIO_ID}.console.txt"

    PROGRAM_ARGS="cttc-nr-demo --gNbNum=${GNB_COUNT} --ueNumPergNb=${UE_PER_GNB} --lambdaBe=${LAMBDA_BE} --lambdaUll=${LAMBDA_ULL} --packetSizeBe=${PACKET_SIZE_BE} --packetSizeUll=${PACKET_SIZE_ULL} --simTime=${SIM_TIME} --simTag=${OUTPUT_FILE} --outputDir=${RAW_DIR} --RngRun=${SEED}"
    MANIFEST_PROGRAM_ARGS="cttc-nr-demo --gNbNum=${GNB_COUNT} --ueNumPergNb=${UE_PER_GNB} --lambdaBe=${LAMBDA_BE} --lambdaUll=${LAMBDA_ULL} --packetSizeBe=${PACKET_SIZE_BE} --packetSizeUll=${PACKET_SIZE_ULL} --simTime=${SIM_TIME} --simTag=${OUTPUT_FILE} --outputDir=data/external/ns3_lena/raw --RngRun=${SEED}"
    COMMAND="./ns3 run \"${MANIFEST_PROGRAM_ARGS}\""

    echo "Running ${SCENARIO_ID}"
    ./ns3 run "${PROGRAM_ARGS}" > "${RAW_DIR}/${CONSOLE_FILE}" 2>&1

    if [[ ! -s "${RAW_DIR}/${OUTPUT_FILE}" ]]; then
      echo "ERROR: expected simulator output was not created or is empty: ${RAW_DIR}/${OUTPUT_FILE}" >&2
      echo "Check console log: ${RAW_DIR}/${CONSOLE_FILE}" >&2
      exit 1
    fi

    {
      csv_escape "${SCENARIO_ID}"; printf ","
      csv_escape "${OUTPUT_FILE}"; printf ","
      csv_escape "${CONSOLE_FILE}"; printf ","
      csv_escape "${LOAD_LABEL}"; printf ","
      csv_escape "${SEED}"; printf ","
      csv_escape "${GNB_COUNT}"; printf ","
      csv_escape "${UE_PER_GNB}"; printf ","
      csv_escape "${LAMBDA_BE}"; printf ","
      csv_escape "${LAMBDA_ULL}"; printf ","
      csv_escape "${PACKET_SIZE_BE}"; printf ","
      csv_escape "${PACKET_SIZE_ULL}"; printf ","
      csv_escape "${SIM_TIME}"; printf ","
      csv_escape "${COMMAND}"; printf "\n"
    } >> "${MANIFEST}"
  done
done

echo "5G-LENA/ns-3 cttc-nr-demo grid complete."
echo "Raw traces: ${RAW_DIR}"
echo "Manifest: ${MANIFEST}"