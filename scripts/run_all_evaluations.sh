#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$ROOT_DIR/data/Patient_Psi_CM_Dataset.json"
INPUT_ROOT="$ROOT_DIR/outputs"
OUTPUT_ROOT="$ROOT_DIR/evaluations"
EXTRACTION_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
INCLUDE_CTRS=0
CTRS_MODEL="gpt-4o-mini"
SKIP_EXISTING=0

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

usage() {
  cat <<'EOF'
Usage: scripts/run_all_evaluations.sh [options]

Evaluate transcripts inside outputs/easy, outputs/normal, and outputs/hard
and save one evaluation JSON per session into matching evaluations directories.

Options:
  --dataset PATH              Dataset JSON path
  --input-root PATH           Root transcripts directory (default: outputs)
  --output-root PATH          Root evaluations directory (default: evaluations)
  --extraction-model MODEL    LLM used for internal-diagram extraction
  --embedding-model MODEL     Embedding model used for IDSS
  --include-ctrs              Also compute CTRS collaboration score
  --ctrs-model MODEL          LLM used for CTRS scoring
  --skip-existing             Skip if output file already exists
  --help                      Show this message

Examples:
  scripts/run_all_evaluations.sh
  scripts/run_all_evaluations.sh --input-root outputs/revealed_hidden --output-root evaluations/revealed_hidden
  scripts/run_all_evaluations.sh --skip-existing
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --extraction-model)
      EXTRACTION_MODEL="$2"
      shift 2
      ;;
    --embedding-model)
      EMBEDDING_MODEL="$2"
      shift 2
      ;;
    --include-ctrs)
      INCLUDE_CTRS=1
      shift
      ;;
    --ctrs-model)
      CTRS_MODEL="$2"
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

for difficulty in easy normal hard; do
  input_dir="$INPUT_ROOT/$difficulty"
  output_path="$OUTPUT_ROOT/$difficulty/evaluation.json"

  if [[ ! -d "$input_dir" ]]; then
    echo "Skipping $difficulty because input directory does not exist: $input_dir"
    continue
  fi

  shopt -s nullglob
  session_files=("$input_dir"/*.json)
  shopt -u nullglob

  if [[ ${#session_files[@]} -eq 0 ]]; then
    echo "Skipping $difficulty because no transcript JSON files were found in $input_dir"
    continue
  fi

  for session_file in "${session_files[@]}"; do
    session_name="$(basename "$session_file")"
    case_id="${session_name#session_}"
    case_id="${case_id%.json}"
    output_path="$OUTPUT_ROOT/$difficulty/eval_${case_id}.json"

    if [[ "$SKIP_EXISTING" -eq 1 && -f "$output_path" ]]; then
      echo "Skipping $difficulty/$case_id because $output_path already exists"
      continue
    fi

    cmd=(
      python3 -m mind_voyager.evaluate_dialogue
      --input "$session_file"
      --dataset "$DATASET"
      --extraction-model "$EXTRACTION_MODEL"
      --embedding-model "$EMBEDDING_MODEL"
      --output "$output_path"
    )

    if [[ "$INCLUDE_CTRS" -eq 1 ]]; then
      cmd+=(--include-ctrs --ctrs-model "$CTRS_MODEL")
    fi

    echo "Evaluating $difficulty/$case_id from $session_file"
    "${cmd[@]}"
  done
done
