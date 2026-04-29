#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$ROOT_DIR/data/Patient_Psi_CM_Dataset.json"
OUTPUT_ROOT="$ROOT_DIR/outputs/random10_comparison"
EVAL_ROOT="$ROOT_DIR/evaluations/random10_comparison"
SAMPLE_SIZE=10
SAMPLE_SEED=42
THERAPIST_MODEL="gpt-4o-mini"
CLIENT_MODEL="gpt-4o-mini"
JUDGE_MODEL="gpt-4o-mini"
MODERATOR_MODEL="gpt-4o-mini"
THERAPIST_PROVIDER="openai"
EXTRACTION_MODEL="gpt-4o"
EMBEDDING_MODEL="text-embedding-3-small"
MAX_TURNS=25
PROBE_TURNS=15
PROBE_INTERVAL=2
PROBE_ANCHOR_COUNT=1
NO_MODERATOR=1
SKIP_EXISTING=0

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

usage() {
  cat <<'EOF'
Usage: scripts/run_random10_comparison.sh [options]

Sample case IDs once, then run baseline, retrieval, and hybrid_probe across easy,
normal, and hard. Evaluate every transcript and write comparison summaries.

Options:
  --dataset PATH                 Dataset JSON path
  --output-root PATH             Output transcript root
  --eval-root PATH               Evaluation root
  --sample-size N                Number of random case IDs (default: 10)
  --sample-seed N                Seed for reproducible sampling (default: 42)
  --therapist-model MODEL        Therapist model (default: gpt-4o-mini)
  --client-model MODEL           Client model (default: gpt-4o-mini)
  --judge-model MODEL            Judge model (default: gpt-4o-mini)
  --moderator-model MODEL        Moderator model (default: gpt-4o-mini)
  --therapist-provider NAME      openai or groq (default: openai)
  --extraction-model MODEL       Evaluation extraction model (default: gpt-4o-mini)
  --embedding-model MODEL        Evaluation embedding model
  --max-turns N                  Maximum therapist turns (default: 25)
  --probe-turns N                Hybrid random-probe window (default: 15)
  --probe-interval N             Hybrid random-probe interval (default: 2)
  --probe-anchor-count N         Probe topics per probe turn (default: 1)
  --use-moderator                Enable moderator early stopping
  --skip-existing                Skip existing session/eval files
  --help                         Show this message

Output layout:
  outputs/random10_comparison/{baseline,retrieval,hybrid_probe}/{easy,normal,hard}/session_CASE.json
  evaluations/random10_comparison/{baseline,retrieval,hybrid_probe}/{easy,normal,hard}/eval_CASE.json
  evaluations/random10_comparison/comparison_summary.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
    --therapist-model) THERAPIST_MODEL="$2"; shift 2 ;;
    --client-model) CLIENT_MODEL="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --moderator-model) MODERATOR_MODEL="$2"; shift 2 ;;
    --therapist-provider) THERAPIST_PROVIDER="$2"; shift 2 ;;
    --extraction-model) EXTRACTION_MODEL="$2"; shift 2 ;;
    --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --probe-turns) PROBE_TURNS="$2"; shift 2 ;;
    --probe-interval) PROBE_INTERVAL="$2"; shift 2 ;;
    --probe-anchor-count) PROBE_ANCHOR_COUNT="$2"; shift 2 ;;
    --use-moderator) NO_MODERATOR=0; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
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

mkdir -p "$OUTPUT_ROOT" "$EVAL_ROOT"
SAMPLE_FILE="$EVAL_ROOT/sample_case_ids_seed_${SAMPLE_SEED}_n_${SAMPLE_SIZE}.txt"

python3 - "$DATASET" "$SAMPLE_SIZE" "$SAMPLE_SEED" "$SAMPLE_FILE" <<'PY'
import json
import random
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
sample_size = int(sys.argv[2])
seed = int(sys.argv[3])
output = Path(sys.argv[4])
rows = json.loads(dataset.read_text())
case_ids = [row["id"] for row in rows]
if sample_size > len(case_ids):
    raise SystemExit(f"sample-size {sample_size} exceeds dataset size {len(case_ids)}")
rng = random.Random(seed)
sample = sorted(rng.sample(case_ids, sample_size))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(sample) + "\n")
print(f"Sampled {len(sample)} case IDs with seed {seed}: {', '.join(sample)}")
PY

mapfile -t CASE_IDS < "$SAMPLE_FILE"
DIFFICULTIES=(easy normal hard)
APPROACHES=(baseline retrieval hybrid_probe)

run_simulation() {
  local approach="$1"
  local difficulty="$2"
  local case_id="$3"
  local output_path="$4"

  if [[ "$SKIP_EXISTING" -eq 1 && -f "$output_path" ]]; then
    echo "Skipping existing session: $output_path"
    return
  fi

  mkdir -p "$(dirname "$output_path")"
  local common_args=(
    --case-id "$case_id"
    --dataset "$DATASET"
    --difficulty "$difficulty"
    --therapist-model "$THERAPIST_MODEL"
    --therapist-provider "$THERAPIST_PROVIDER"
    --client-model "$CLIENT_MODEL"
    --judge-model "$JUDGE_MODEL"
    --moderator-model "$MODERATOR_MODEL"
    --max-turns "$MAX_TURNS"
    --output "$output_path"
  )
  if [[ "$NO_MODERATOR" -eq 1 ]]; then
    common_args+=(--no-moderator)
  fi

  echo "Simulating [$approach][$difficulty][$case_id] -> $output_path"
  case "$approach" in
    baseline)
      python3 -m mind_voyager.simulate_conversation "${common_args[@]}"
      ;;
    retrieval)
      python3 -m mind_voyager.simulate_conversation "${common_args[@]}" --use-retrieval
      ;;
    hybrid_probe)
      python3 -m mind_voyager.simulate_topic_probe_conversation \
        "${common_args[@]}" \
        --prompt-mode hybrid_probe \
        --probe-turns "$PROBE_TURNS" \
        --probe-interval "$PROBE_INTERVAL" \
        --probe-anchor-count "$PROBE_ANCHOR_COUNT"
      ;;
    *)
      echo "Unknown approach: $approach" >&2
      exit 1
      ;;
  esac
}

run_evaluation() {
  local session_path="$1"
  local eval_path="$2"

  if [[ "$SKIP_EXISTING" -eq 1 && -f "$eval_path" ]]; then
    echo "Skipping existing eval: $eval_path"
    return
  fi

  mkdir -p "$(dirname "$eval_path")"
  echo "Evaluating $session_path -> $eval_path"
  python3 -m mind_voyager.evaluate_dialogue \
    --input "$session_path" \
    --dataset "$DATASET" \
    --extraction-model "$EXTRACTION_MODEL" \
    --embedding-model "$EMBEDDING_MODEL" \
    --output "$eval_path" \
    --json-only
}

for approach in "${APPROACHES[@]}"; do
  for difficulty in "${DIFFICULTIES[@]}"; do
    for case_id in "${CASE_IDS[@]}"; do
      session_path="$OUTPUT_ROOT/$approach/$difficulty/session_${case_id}.json"
      eval_path="$EVAL_ROOT/$approach/$difficulty/eval_${case_id}.json"
      run_simulation "$approach" "$difficulty" "$case_id" "$session_path"
      run_evaluation "$session_path" "$eval_path"
    done
  done
done

python3 - "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1])
approaches = ["baseline", "retrieval", "hybrid_probe"]
difficulties = ["easy", "normal", "hard"]

summary = {"by_approach": {}, "by_difficulty": {}, "overall_ranking": []}
rows = []

def aggregate(files):
    payloads = [json.loads(p.read_text())["aggregate"] for p in files]
    if not payloads:
        return {"sessions": 0, "edss_average": None, "idss_average": None, "combined_average": None}
    edss = mean(p["edss"]["average"] for p in payloads)
    idss = mean(p["idss"]["average"] for p in payloads)
    return {
        "sessions": len(payloads),
        "edss_average": edss,
        "idss_average": idss,
        "combined_average": (edss + idss) / 2,
    }

for approach in approaches:
    summary["by_approach"][approach] = {}
    all_files = []
    for difficulty in difficulties:
        files = sorted((root / approach / difficulty).glob("eval_*.json"))
        all_files.extend(files)
        stats = aggregate(files)
        summary["by_approach"][approach][difficulty] = stats
        rows.append((approach, difficulty, stats))
    summary["by_approach"][approach]["overall"] = aggregate(all_files)
    ranking_stats = summary["by_approach"][approach]["overall"]
    summary["overall_ranking"].append({"approach": approach, **ranking_stats})

for difficulty in difficulties:
    summary["by_difficulty"][difficulty] = {}
    for approach in approaches:
        summary["by_difficulty"][difficulty][approach] = summary["by_approach"][approach][difficulty]

summary["overall_ranking"].sort(
    key=lambda item: item["combined_average"] if item["combined_average"] is not None else -1,
    reverse=True,
)

out = root / "comparison_summary.json"
out.write_text(json.dumps(summary, indent=2))

print("\nComparison summary")
print("------------------")
print(f"{'Approach':<14} {'Difficulty':<10} {'N':>4} {'EDSS':>9} {'IDSS':>9} {'Combined':>10}")
for approach, difficulty, stats in rows:
    if stats["sessions"] == 0:
        continue
    print(
        f"{approach:<14} {difficulty:<10} {stats['sessions']:>4} "
        f"{stats['edss_average'] * 100:>9.2f} "
        f"{stats['idss_average'] * 100:>9.2f} "
        f"{stats['combined_average'] * 100:>10.2f}"
    )
print("\nOverall ranking")
for idx, item in enumerate(summary["overall_ranking"], start=1):
    print(
        f"{idx}. {item['approach']}: "
        f"combined={item['combined_average'] * 100:.2f}, "
        f"EDSS={item['edss_average'] * 100:.2f}, "
        f"IDSS={item['idss_average'] * 100:.2f}, "
        f"N={item['sessions']}"
    )
print(f"\nSaved summary to {out}")
PY
