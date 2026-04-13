from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm import cosine_similarity, get_embedding
from mind_voyager.client_simulator import DEFAULT_DATASET
from mind_voyager.evaluate_dialogue import build_ground_truth


IDSS_FIELDS = (
    "relevant_history",
    "core_beliefs",
    "intermediate_beliefs",
    "coping_strategies",
)


def compare_ground_truth_diagrams(
    case_a: str,
    case_b: str,
    dataset: Path,
    embedding_model: str,
) -> dict[str, Any]:
    diagram_a = build_ground_truth(case_a, dataset)
    diagram_b = build_ground_truth(case_b, dataset)

    scores: dict[str, float] = {}
    for key in IDSS_FIELDS:
        embedding_a = get_embedding(diagram_a[key], model=embedding_model)
        embedding_b = get_embedding(diagram_b[key], model=embedding_model)
        scores[key] = cosine_similarity(embedding_a, embedding_b)
    scores["average"] = sum(scores.values()) / len(IDSS_FIELDS)

    return {
        "case_a": case_a,
        "case_b": case_b,
        "embedding_model": embedding_model,
        "ground_truth_internal_diagram_a": diagram_a,
        "ground_truth_internal_diagram_b": diagram_b,
        "idss_like_similarity": scores,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the ground-truth internal diagrams of two patients using "
            "the same field-wise embedding cosine similarity logic as IDSS."
        )
    )
    parser.add_argument("--case-a", required=True, help="First case id, e.g. 1-1")
    parser.add_argument("--case-b", required=True, help="Second case id, e.g. 2-3")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to Patient_Psi_CM_Dataset.json",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help=(
            "Embedding model used for similarity scoring. Supports OpenAI embedding "
            "models or sentence-transformers models such as "
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional path to save the comparison JSON. "
            "Defaults to outputs/comparisons/gt_compare_<case-a>_vs_<case-b>.json"
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = compare_ground_truth_diagrams(
        case_a=args.case_a,
        case_b=args.case_b,
        dataset=Path(args.dataset),
        embedding_model=args.embedding_model,
    )
    output_path = (
        Path(args.output)
        if args.output
        else Path("comparisons") / f"gt_compare_{args.case_a}_vs_{args.case_b}.json"
    )

    print("Ground-Truth CCD Similarity")
    print()
    print(f"Case A: {result['case_a']}")
    print(f"Case B: {result['case_b']}")
    print(f"Embedding model: {result['embedding_model']}")
    print()
    print("Field-wise similarity")
    for key in (*IDSS_FIELDS, "average"):
        print(f"- {key}: {result['idss_like_similarity'][key]:.4f}")
    print()
    print(json.dumps(result, indent=2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print()
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
