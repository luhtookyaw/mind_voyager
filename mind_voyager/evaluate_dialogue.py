from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm import call_llm, cosine_similarity, get_embedding
from mind_voyager.client_simulator import DEFAULT_DATASET, DIFFICULTIES, load_case, load_prompt


def load_transcript_payloads(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]
    payloads = []
    for file_path in files:
        payloads.append((file_path, json.loads(file_path.read_text())))
    return payloads


def normalize_transcript_entries(payload: dict[str, Any]) -> list[dict[str, str]]:
    if "transcript" in payload:
        return payload["transcript"]
    if "dialogue" in payload:
        normalized = []
        for item in payload["dialogue"]:
            role = item["role"]
            if role == "user":
                speaker = "therapist"
            elif role == "assistant":
                speaker = "client"
            else:
                speaker = role
            normalized.append({"speaker": speaker, "content": item["content"]})
        return normalized
    raise KeyError("Transcript payload must contain either 'transcript' or 'dialogue'.")


def transcript_text(payload: dict[str, Any]) -> str:
    lines = []
    for item in normalize_transcript_entries(payload):
        speaker = "Therapist" if item["speaker"] == "therapist" else "Client"
        lines.append(f"{speaker}: {item['content']}")
    return "\n".join(lines)


def build_ground_truth(case_id: str, dataset: Path) -> dict[str, str]:
    case = load_case(dataset, case_id)
    core_beliefs = " | ".join(case.core_beliefs) if case.core_beliefs else "unknown"
    intermediate = case.intermediate_belief or "unknown"
    if case.intermediate_belief_depression:
        intermediate = f"{intermediate}\nDepressive variant: {case.intermediate_belief_depression}"
    return {
        "relevant_history": case.history or "unknown",
        "core_beliefs": core_beliefs,
        "intermediate_beliefs": intermediate,
        "coping_strategies": case.coping_strategies or "unknown",
    }


def extract_internal_diagram(transcript: str, model: str) -> dict[str, str]:
    prompt = load_prompt("internal_diagram_extraction.txt").replace("{transcript}", transcript)
    raw = call_llm(system_prompt="", user_prompt=prompt, temperature=0.0, model=model)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse extraction JSON: {raw}") from exc
    return {
        "relevant_history": str(parsed.get("relevant_history", "unknown")),
        "core_beliefs": str(parsed.get("core_beliefs", "unknown")),
        "intermediate_beliefs": str(parsed.get("intermediate_beliefs", "unknown")),
        "coping_strategies": str(parsed.get("coping_strategies", "unknown")),
    }


def compute_cder(payload: dict[str, Any]) -> dict[str, int]:
    difficulty_name = payload["difficulty"]
    initial_visible = DIFFICULTIES[difficulty_name].initial_visible_experiences
    final_visible = int(
        payload.get(
            "final_visible_experience_count",
            payload.get("visible_experience_count", initial_visible),
        )
    )
    internal_revealed = bool(payload.get("internal_revealed", False))
    external_success = int(final_visible >= 3)
    internal_success = int(internal_revealed)
    return {
        "external": external_success,
        "internal": internal_success,
        "overall": int(external_success and internal_success),
    }


def compute_idss(
    predicted: dict[str, str],
    ground_truth: dict[str, str],
    embedding_model: str,
) -> dict[str, float]:
    scores = {}
    for key in (
        "relevant_history",
        "core_beliefs",
        "intermediate_beliefs",
        "coping_strategies",
    ):
        pred_embedding = get_embedding(predicted[key], model=embedding_model)
        truth_embedding = get_embedding(ground_truth[key], model=embedding_model)
        scores[key] = cosine_similarity(pred_embedding, truth_embedding)
    scores["average"] = sum(scores.values()) / 4
    return scores


def maybe_compute_ctrs(transcript: str, model: str) -> str:
    prompt = load_prompt("ctrs_evaluation.txt").format(conversation=transcript)
    return call_llm(system_prompt="", user_prompt=prompt, temperature=0.0, model=model).strip()


def evaluate_file(
    file_path: Path,
    payload: dict[str, Any],
    dataset: Path,
    extraction_model: str,
    embedding_model: str,
    include_ctrs: bool,
    ctrs_model: str,
) -> dict[str, Any]:
    transcript = transcript_text(payload)
    ground_truth = build_ground_truth(payload["case_id"], dataset)
    predicted = extract_internal_diagram(transcript, extraction_model)
    cder = compute_cder(payload)
    idss = compute_idss(predicted, ground_truth, embedding_model)
    result: dict[str, Any] = {
        "file": str(file_path),
        "case_id": payload["case_id"],
        "client_mode": payload.get("client_mode", "masked"),
        "difficulty": payload["difficulty"],
        "cder": cder,
        "predicted_internal_diagram": predicted,
        "ground_truth_internal_diagram": ground_truth,
        "idss": idss,
    }
    if include_ctrs:
        result["ctrs_collaboration"] = maybe_compute_ctrs(transcript, ctrs_model)
    return result


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    cder_external = sum(item["cder"]["external"] for item in results) / count
    cder_internal = sum(item["cder"]["internal"] for item in results) / count
    cder_overall = sum(item["cder"]["overall"] for item in results) / count
    idss_keys = [
        "relevant_history",
        "core_beliefs",
        "intermediate_beliefs",
        "coping_strategies",
        "average",
    ]
    idss = {
        key: sum(item["idss"][key] for item in results) / count
        for key in idss_keys
    }
    return {
        "num_sessions": count,
        "cder": {
            "external": cder_external,
            "internal": cder_internal,
            "overall": cder_overall,
        },
        "idss": idss,
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}"


def score(value: float) -> str:
    return f"{value:.4f}"


def pct_score(value: float) -> str:
    return f"{value * 100:.2f}"


def print_paper_style_tables(aggregate: dict[str, Any]) -> None:
    cder = aggregate["cder"]
    idss = aggregate["idss"]

    print("Table 1: Cognitive Diagram Exposure Rate (CDER)")
    print()
    print(f"{'Sessions':<10} {'E':>8} {'I':>8} {'G':>8}")
    print(
        f"{aggregate['num_sessions']:<10} "
        f"{pct(cder['external']):>8} "
        f"{pct(cder['internal']):>8} "
        f"{pct(cder['overall']):>8}"
    )
    print()

    print("Table 2: Induced Diagram Similarity Score (IDSS)")
    print()
    print(f"{'Sessions':<10} {'Avg.':>10} {'RH':>10} {'CB':>10} {'IB':>10} {'CS':>10}")
    print(
        f"{aggregate['num_sessions']:<10} "
        f"{pct_score(idss['average']):>10} "
        f"{pct_score(idss['relevant_history']):>10} "
        f"{pct_score(idss['core_beliefs']):>10} "
        f"{pct_score(idss['intermediate_beliefs']):>10} "
        f"{pct_score(idss['coping_strategies']):>10}"
    )
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MindVoyager dialogue transcripts")
    parser.add_argument("--input", required=True, help="Transcript JSON file or directory")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to Patient_Psi_CM_Dataset.json",
    )
    parser.add_argument(
        "--extraction-model",
        default="gpt-4o-mini",
        help="Model used to extract internal cognitive elements from transcript",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help=(
            "Embedding model used for IDSS similarity scoring. "
            "Supports OpenAI embedding models or sentence-transformers models "
            "such as sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    parser.add_argument("--include-ctrs", action="store_true", help="Also score CTRS collaboration")
    parser.add_argument(
        "--ctrs-model",
        default="gpt-4o-mini",
        help="Model used for CTRS collaboration scoring",
    )
    parser.add_argument("--output", help="Optional path to save evaluation JSON")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON output without the paper-style terminal table",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = Path(args.dataset)
    transcript_path = Path(args.input)
    payloads = load_transcript_payloads(transcript_path)
    results = [
        evaluate_file(
            file_path=file_path,
            payload=payload,
            dataset=dataset,
            extraction_model=args.extraction_model,
            embedding_model=args.embedding_model,
            include_ctrs=args.include_ctrs,
            ctrs_model=args.ctrs_model,
        )
        for file_path, payload in payloads
    ]
    aggregate = aggregate_results(results)
    output = {
        "aggregate": aggregate,
        "sessions": results,
    }
    if not args.json_only:
        print_paper_style_tables(aggregate)
    print(json.dumps(output, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
