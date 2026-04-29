#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_PATH="$ROOT_DIR/data/Patient_Psi_CM_Dataset.json"
OUTPUT_ROOT="$ROOT_DIR/outputs/threshold_calibration"

THERAPIST_MODEL="gpt-4o-mini"
THERAPIST_PROVIDER="openai"
CLIENT_MODEL="gpt-4o-mini"
JUDGE_MODEL="gpt-4o-mini"
MODERATOR_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="text-embedding-3-small"
MAX_TURNS=15
CASE_COUNT=10
USE_MODERATOR=0

usage() {
  cat <<'EOF'
Run threshold-calibration self-play sessions across easy, normal, and hard.

Default behavior:
- uses the first 10 case ids from data/Patient_Psi_CM_Dataset.json
- runs 30 total sessions: 10 case ids x 3 difficulties
- disables moderator for equal turn budgets
- writes outputs to outputs/threshold_calibration/{easy,normal,hard}

Usage:
  scripts/run_threshold_calibration.sh [options] [case_id ...]

Options:
  --dataset PATH                Dataset JSON path
  --output-root PATH            Output directory root
  --therapist-model MODEL       Therapist model (default: gpt-4o-mini)
  --therapist-provider NAME     Therapist provider (default: openai)
  --client-model MODEL          Client model (default: gpt-4o-mini)
  --judge-model MODEL           Compatibility arg passed through (default: gpt-4o-mini)
  --moderator-model MODEL       Moderator model (default: gpt-4o-mini)
  --embedding-model MODEL       Embedding model (default: text-embedding-3-small)
  --max-turns N                 Maximum therapist turns per session (default: 15)
  --case-count N                Number of case ids to auto-select if none are provided (default: 10)
  --use-moderator               Enable moderator-based early stopping
  --help                        Show this help text

Examples:
  scripts/run_threshold_calibration.sh
  scripts/run_threshold_calibration.sh 1-1 2-1 3-1 4-1 5-1 6-1 7-1 8-1 9-1 10-1
  scripts/run_threshold_calibration.sh --case-count 12 --max-turns 12
EOF
}

CASE_IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_PATH="$2"
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
    --therapist-provider)
      THERAPIST_PROVIDER="$2"
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
    --moderator-model)
      MODERATOR_MODEL="$2"
      shift 2
      ;;
    --embedding-model)
      EMBEDDING_MODEL="$2"
      shift 2
      ;;
    --max-turns)
      MAX_TURNS="$2"
      shift 2
      ;;
    --case-count)
      CASE_COUNT="$2"
      shift 2
      ;;
    --use-moderator)
      USE_MODERATOR=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      CASE_IDS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset not found: $DATASET_PATH" >&2
  exit 1
fi

if [[ ${#CASE_IDS[@]} -eq 0 ]]; then
  mapfile -t CASE_IDS < <(
    python - "$DATASET_PATH" "$CASE_COUNT" <<'PY'
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
case_count = int(sys.argv[2])
rows = json.loads(dataset_path.read_text())
case_ids = [row["id"] for row in rows[:case_count]]
for case_id in case_ids:
    print(case_id)
PY
  )
fi

if [[ ${#CASE_IDS[@]} -eq 0 ]]; then
  echo "No case ids were selected." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"/easy "$OUTPUT_ROOT"/normal "$OUTPUT_ROOT"/hard

echo "Threshold calibration run"
echo "Root: $ROOT_DIR"
echo "Dataset: $DATASET_PATH"
echo "Output root: $OUTPUT_ROOT"
echo "Case ids (${#CASE_IDS[@]}): ${CASE_IDS[*]}"
echo "Therapist model: $THERAPIST_MODEL"
echo "Client model: $CLIENT_MODEL"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Max turns: $MAX_TURNS"
echo "Moderator: $([[ "$USE_MODERATOR" -eq 1 ]] && echo enabled || echo disabled)"
echo

for difficulty in easy normal hard; do
  for case_id in "${CASE_IDS[@]}"; do
    output_path="$OUTPUT_ROOT/$difficulty/session_${case_id}.json"
    echo "Running [$difficulty][$case_id] -> $output_path"

    cmd=(
      python -m mind_voyager.simulate_conversation
      --case-id "$case_id"
      --dataset "$DATASET_PATH"
      --difficulty "$difficulty"
      --therapist-model "$THERAPIST_MODEL"
      --therapist-provider "$THERAPIST_PROVIDER"
      --client-model "$CLIENT_MODEL"
      --judge-model "$JUDGE_MODEL"
      --moderator-model "$MODERATOR_MODEL"
      --max-turns "$MAX_TURNS"
      --output "$output_path"
    )

    if [[ "$USE_MODERATOR" -eq 0 ]]; then
      cmd+=(--no-moderator)
    fi

    OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    MIND_VOYAGER_EMBEDDING_MODEL="$EMBEDDING_MODEL" \
    "${cmd[@]}"
  done
done

echo
echo "Completed threshold calibration runs."
