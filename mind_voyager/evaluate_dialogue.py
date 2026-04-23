from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from llm import call_llm, cosine_similarity, get_embedding
from mind_voyager.client_simulator import DEFAULT_DATASET, load_case, load_prompt


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


def build_external_ground_truth(case_id: str, dataset: Path) -> dict[str, str]:
    case = load_case(dataset, case_id)
    return {
        "situation": case.situation or "unknown",
        "automatic_thought": case.automatic_thought or "unknown",
        "emotion": "; ".join(case.emotion) if case.emotion else "unknown",
        "behavior": case.behavior or "unknown",
    }


def render_extraction_prompt(
    prompt_name: str,
    transcript: str,
) -> str:
    prompt = load_prompt(prompt_name)
    replacements = {
        "{dialogue_history}": transcript,
        "{transcript}": transcript,
    }
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def render_internal_diagram_extraction_prompt(
    transcript: str,
) -> str:
    return render_extraction_prompt("internal_diagram_extraction.txt", transcript)


def render_external_diagram_extraction_prompt(
    transcript: str,
) -> str:
    return render_extraction_prompt("external_diagram_extraction.txt", transcript)


def normalize_extracted_value(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() == "null":
        return "unknown"
    return text


def normalize_extracted_field(value: Any) -> str:
    if isinstance(value, dict):
        return normalize_extracted_value(value.get("value"))
    return normalize_extracted_value(value)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected extraction JSON object, got {type(parsed).__name__}: {raw}")
    return parsed


def extract_internal_diagram(
    transcript: str,
    model: str,
) -> dict[str, str]:
    prompt = render_internal_diagram_extraction_prompt(transcript)
    raw = call_llm(system_prompt="", user_prompt=prompt, temperature=0.0, model=model)
    try:
        parsed = parse_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Failed to parse extraction JSON: {raw}") from exc
    return {
        "relevant_history": normalize_extracted_field(parsed.get("relevant_history")),
        "core_beliefs": normalize_extracted_field(parsed.get("core_beliefs")),
        "intermediate_beliefs": normalize_extracted_field(parsed.get("intermediate_beliefs")),
        "coping_strategies": normalize_extracted_field(parsed.get("coping_strategies")),
    }


def extract_external_diagram(
    transcript: str,
    model: str,
) -> dict[str, str]:
    prompt = render_external_diagram_extraction_prompt(transcript)
    raw = call_llm(system_prompt="", user_prompt=prompt, temperature=0.0, model=model)
    try:
        parsed = parse_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Failed to parse external extraction JSON: {raw}") from exc
    return {
        "situation": normalize_extracted_field(parsed.get("situation")),
        "automatic_thought": normalize_extracted_field(parsed.get("automatic_thought")),
        "emotion": normalize_extracted_field(parsed.get("emotion")),
        "behavior": normalize_extracted_field(parsed.get("behavior")),
    }


def compute_similarity_scores(
    predicted: dict[str, str],
    ground_truth: dict[str, str],
    embedding_model: str,
    keys: tuple[str, ...],
) -> dict[str, float]:
    scores = {}
    for key in keys:
        pred_embedding = get_embedding(predicted[key], model=embedding_model)
        truth_embedding = get_embedding(ground_truth[key], model=embedding_model)
        scores[key] = cosine_similarity(pred_embedding, truth_embedding)
    scores["average"] = sum(scores.values()) / len(keys)
    return scores


def compute_idss(
    predicted: dict[str, str],
    ground_truth: dict[str, str],
    embedding_model: str,
) -> dict[str, float]:
    return compute_similarity_scores(
        predicted,
        ground_truth,
        embedding_model,
        (
            "relevant_history",
            "core_beliefs",
            "intermediate_beliefs",
            "coping_strategies",
        ),
    )


def compute_edss(
    predicted: dict[str, str],
    ground_truth: dict[str, str],
    embedding_model: str,
) -> dict[str, float]:
    return compute_similarity_scores(
        predicted,
        ground_truth,
        embedding_model,
        (
            "situation",
            "automatic_thought",
            "emotion",
            "behavior",
        ),
    )


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
    internal_ground_truth = build_ground_truth(payload["case_id"], dataset)
    external_ground_truth = build_external_ground_truth(payload["case_id"], dataset)
    predicted_internal = extract_internal_diagram(transcript, extraction_model)
    predicted_external = extract_external_diagram(transcript, extraction_model)
    idss = compute_idss(predicted_internal, internal_ground_truth, embedding_model)
    edss = compute_edss(predicted_external, external_ground_truth, embedding_model)
    result: dict[str, Any] = {
        "file": str(file_path),
        "case_id": payload["case_id"],
        "client_mode": payload.get("client_mode", "masked"),
        "difficulty": payload["difficulty"],
        "predicted_external_diagram": predicted_external,
        "ground_truth_external_diagram": external_ground_truth,
        "predicted_internal_diagram": predicted_internal,
        "ground_truth_internal_diagram": internal_ground_truth,
        "edss": edss,
        "idss": idss,
    }
    if include_ctrs:
        result["ctrs_collaboration"] = maybe_compute_ctrs(transcript, ctrs_model)
    return result


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
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
    edss_keys = [
        "situation",
        "automatic_thought",
        "emotion",
        "behavior",
        "average",
    ]
    edss = {
        key: sum(item["edss"][key] for item in results) / count
        for key in edss_keys
    }
    return {
        "num_sessions": count,
        "edss": edss,
        "idss": idss,
    }


def pct_score(value: float) -> str:
    return f"{value * 100:.2f}"


def print_paper_style_tables(aggregate: dict[str, Any]) -> None:
    edss = aggregate["edss"]
    idss = aggregate["idss"]

    print("External Diagram Similarity Score (EDSS)")
    print()
    print(f"{'Sessions':<10} {'Avg.':>10} {'S':>10} {'AT':>10} {'E':>10} {'B':>10}")
    print(
        f"{aggregate['num_sessions']:<10} "
        f"{pct_score(edss['average']):>10} "
        f"{pct_score(edss['situation']):>10} "
        f"{pct_score(edss['automatic_thought']):>10} "
        f"{pct_score(edss['emotion']):>10} "
        f"{pct_score(edss['behavior']):>10}"
    )
    print()

    print("Induced Diagram Similarity Score (IDSS)")
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
        default="gpt-4o",
        help="Model used to extract internal cognitive elements from transcript",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help=(
            "Embedding model used for EDSS and IDSS similarity scoring. "
            "Supports OpenAI embedding models or sentence-transformers models "
            "such as sentence-transformers/all-mpnet-base-v2"
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
