#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$ROOT_DIR/data/Patient_Psi_CM_Dataset.json"
OUTPUT_ROOT="$ROOT_DIR/outputs"
THERAPIST_MODEL="gpt-4o-mini"
CLIENT_MODEL="gpt-4o-mini"
JUDGE_MODEL="gpt-4o-mini"
THERAPIST_PROVIDER="openai"
MAX_TURNS=15
SKIP_EXISTING=0

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

usage() {
  cat <<'EOF'
Usage: scripts/run_all_simulations.sh [options]

Run simulate_conversation for every case id in the dataset across easy, normal, and hard.

Options:
  --dataset PATH               Dataset JSON path
  --output-root PATH           Output root directory
  --therapist-model MODEL      Therapist model (default: gpt-4o-mini)
  --client-model MODEL         Client model (default: gpt-4o-mini)
  --judge-model MODEL          Judge model for masked mode (default: gpt-4o-mini)
  --therapist-provider NAME    openai or groq (default: openai)
  --max-turns N                Maximum therapist turns per session (default: 15)
  --skip-existing              Skip a simulation when the output JSON already exists
  --help                       Show this message

Examples:
  scripts/run_all_simulations.sh
  scripts/run_all_simulations.sh --output-root outputs/masked
  scripts/run_all_simulations.sh --skip-existing
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --therapist-model)
      THERAPIST_MODEL="$2"
      shift 2
      ;;
    --client-model)
      CLIENT_MODEL="$2"
      shift 2
      ;;
    --judge-model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --therapist-provider)
      THERAPIST_PROVIDER="$2"
      shift 2
      ;;
    --max-turns)
      MAX_TURNS="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found: $DATASET" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required. Set it in the environment or in $ROOT_DIR/.env." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"/easy "$OUTPUT_ROOT"/normal "$OUTPUT_ROOT"/hard

CASE_IDS=()
while IFS= read -r case_id; do
  CASE_IDS+=("$case_id")
done < <(
  python3 - "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
rows = json.loads(dataset_path.read_text())
for row in rows:
    print(row["id"])
PY
)

for difficulty in easy normal hard; do
  for case_id in "${CASE_IDS[@]}"; do
    output_path="$OUTPUT_ROOT/$difficulty/session_${case_id}.json"

    if [[ "$SKIP_EXISTING" -eq 1 && -f "$output_path" ]]; then
      echo "Skipping case $case_id at difficulty $difficulty because $output_path already exists"
      continue
    fi

    cmd=(
      python3 -m mind_voyager.simulate_conversation
      --case-id "$case_id"
      --dataset "$DATASET"
      --difficulty "$difficulty"
      --therapist-model "$THERAPIST_MODEL"
      --therapist-provider "$THERAPIST_PROVIDER"
      --client-model "$CLIENT_MODEL"
      --judge-model "$JUDGE_MODEL"
      --max-turns "$MAX_TURNS"
      --output "$output_path"
    )

    echo "Running case $case_id at difficulty $difficulty"
    "${cmd[@]}"
  done
done
