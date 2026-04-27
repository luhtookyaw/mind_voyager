from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from llm import call_llm
from retrieve_hybrid_topic_graph import (
    DEFAULT_ANCHOR_TYPES,
    DEFAULT_GRAPH_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_TRAVERSAL_RELATIONS,
    retrieve_hybrid_topic_graph_context,
)


REQUIRED_KEYS = [
    "name",
    "id",
    "type",
    "history",
    "helpless_belief",
    "unlovable_belief",
    "worthless_belief",
    "intermediate_belief",
    "intermediate_belief_depression",
    "coping_strategies",
    "situation",
    "auto_thought",
    "emotion",
    "behavior",
]


def load_text(path: Path) -> str:
    return path.read_text().strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    return payload


def generate_case_id(query_text: str, prefix: str = "synthetic") -> str:
    digest = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def normalize_case_payload(payload: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["name"] = str(normalized.get("name", "Unknown")).strip() or "Unknown"
    normalized["id"] = str(normalized.get("id", fallback_id)).strip() or fallback_id
    normalized["type"] = as_string_list(normalized.get("type"))
    normalized["history"] = str(normalized.get("history", "")).strip()
    normalized["helpless_belief"] = as_string_list(normalized.get("helpless_belief"))
    normalized["unlovable_belief"] = as_string_list(normalized.get("unlovable_belief"))
    normalized["worthless_belief"] = as_string_list(normalized.get("worthless_belief"))
    normalized["intermediate_belief"] = str(normalized.get("intermediate_belief", "")).strip()
    normalized["intermediate_belief_depression"] = str(
        normalized.get("intermediate_belief_depression", "")
    ).strip()
    normalized["coping_strategies"] = str(normalized.get("coping_strategies", "")).strip()
    normalized["situation"] = str(normalized.get("situation", "")).strip()
    normalized["auto_thought"] = str(normalized.get("auto_thought", "")).strip()
    normalized["emotion"] = as_string_list(normalized.get("emotion"))
    normalized["behavior"] = str(normalized.get("behavior", "")).strip()
    return normalized


def validate_case_payload(payload: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")

    if not isinstance(payload["type"], list):
        raise ValueError("'type' must be a list of strings.")
    for key in ("helpless_belief", "unlovable_belief", "worthless_belief", "emotion"):
        if not isinstance(payload[key], list):
            raise ValueError(f"'{key}' must be a list of strings.")


def summarize_case_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    def texts(section: str) -> list[str]:
        return [node.get("text") or node.get("label", "") for node in bundle.get(section, [])]

    return {
        "case_id": bundle["case_record"].get("case_id", bundle["case_record"].get("id")),
        "label": bundle["case_record"]["label"],
        "style_traits": texts("style_traits"),
        "history": texts("history"),
        "core_beliefs": texts("core_beliefs"),
        "intermediate_beliefs": texts("intermediate_beliefs"),
        "coping_strategies": texts("coping_strategies"),
        "situation": texts("situation"),
        "automatic_thought": texts("automatic_thought"),
        "emotions": texts("emotions"),
        "behavior": texts("behavior"),
    }


def build_generation_prompt(
    query_text: str,
    retrieval_result: dict[str, Any],
    candidate_case_limit: int,
) -> str:
    anchors = retrieval_result.get("anchors", [])[:5]
    candidate_cases = retrieval_result.get("candidate_cases", [])[:candidate_case_limit]
    case_bundles = [
        summarize_case_bundle(bundle)
        for bundle in retrieval_result.get("case_bundles", [])[:candidate_case_limit]
    ]

    return (
        "Construct one new Patient-psi-style CBT case as valid JSON.\n\n"
        "Use the source text as the primary evidence for:\n"
        "- situation\n"
        "- auto_thought\n"
        "- emotion\n"
        "- behavior\n\n"
        "Use the retrieved graph context only to infer the most plausible:\n"
        "- history\n"
        "- helpless_belief\n"
        "- unlovable_belief\n"
        "- worthless_belief\n"
        "- intermediate_belief\n"
        "- intermediate_belief_depression\n"
        "- coping_strategies\n"
        "- type\n\n"
        "Requirements:\n"
        "- Output only one JSON object.\n"
        "- Do not wrap the JSON in markdown.\n"
        "- Match the Patient-psi field schema exactly.\n"
        "- Do not copy a retrieved case verbatim.\n"
        "- Keep the external episode close to the source text.\n"
        "- Keep the internal formulation coherent with the retrieved cases.\n"
        "- Use the smallest coherent belief set. Empty lists are allowed when a belief family is unsupported.\n"
        "- 'type' must be a list of short style traits.\n"
        "- 'emotion' must be a list of short emotion strings.\n"
        "- Belief fields must be lists of strings.\n"
        "- If uncertainty is high, stay sparse rather than inventing extra detail.\n\n"
        f"Required JSON keys: {', '.join(REQUIRED_KEYS)}\n\n"
        f"Source text:\n{query_text}\n\n"
        "Retrieved anchors:\n"
        f"{json.dumps(anchors, indent=2)}\n\n"
        "Retrieved candidate cases:\n"
        f"{json.dumps(candidate_cases, indent=2)}\n\n"
        "Retrieved case bundles:\n"
        f"{json.dumps(case_bundles, indent=2)}\n"
    )


def construct_case(
    query_text: str,
    retrieval_result: dict[str, Any],
    model: str,
    temperature: float,
    case_id: str | None,
    candidate_case_limit: int,
) -> dict[str, Any]:
    prompt = build_generation_prompt(query_text, retrieval_result, candidate_case_limit)
    print(prompt)
    raw = call_llm(
        system_prompt="You produce valid JSON only.",
        user_prompt=prompt,
        temperature=temperature,
        model=model,
    )
    fallback_id = case_id or generate_case_id(query_text)
    payload = parse_json_object(raw)
    normalized = normalize_case_payload(payload, fallback_id=fallback_id)
    validate_case_payload(normalized)
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a new Patient-psi-style case from a source post and hybrid-graph retrieval context."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query", help="Source post or extracted external episode text")
    input_group.add_argument("--query-file", help="Path to a text file containing the source post")
    input_group.add_argument(
        "--retrieval-json",
        help=(
            "Path to a previously saved retrieval JSON from retrieve_hybrid_topic_graph.py. "
            "If used, the source query is taken from that file."
        ),
    )
    parser.add_argument(
        "--graph",
        default=str(DEFAULT_GRAPH_PATH),
        help="Path to hybrid_topic_graph.json when running retrieval inline",
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to hybrid node_embeddings.json when running retrieval inline",
    )
    parser.add_argument(
        "--anchor-types",
        default=",".join(DEFAULT_ANCHOR_TYPES),
        help="Comma-separated anchor types used for inline retrieval",
    )
    parser.add_argument(
        "--anchor-top-k",
        type=int,
        default=5,
        help="Number of anchor nodes to keep during inline retrieval",
    )
    parser.add_argument(
        "--relations",
        default=",".join(DEFAULT_TRAVERSAL_RELATIONS),
        help="Comma-separated relations used during inline retrieval traversal",
    )
    parser.add_argument(
        "--max-related",
        type=int,
        default=40,
        help="Maximum related nodes to keep during inline retrieval",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum inline retrieval traversal depth",
    )
    parser.add_argument(
        "--case-top-k",
        type=int,
        default=5,
        help="Maximum candidate cases to retrieve and summarize",
    )
    parser.add_argument(
        "--candidate-case-limit",
        type=int,
        default=3,
        help="How many retrieved candidate cases to include in the generation prompt",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model used to construct the case JSON",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for case construction",
    )
    parser.add_argument(
        "--case-id",
        help="Optional explicit id for the constructed case",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the constructed case JSON",
    )
    parser.add_argument(
        "--save-retrieval",
        help="Optional path to also save the retrieval result when running retrieval inline",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.retrieval_json:
        retrieval_path = Path(args.retrieval_json)
        retrieval_result = json.loads(retrieval_path.read_text())
        query_text = retrieval_result["query"]
    else:
        query_text = args.query or load_text(Path(args.query_file))
        retrieval_result = retrieve_hybrid_topic_graph_context(
            query=query_text,
            graph_path=Path(args.graph),
            index_path=Path(args.index),
            anchor_types=tuple(part.strip() for part in args.anchor_types.split(",") if part.strip())
            or DEFAULT_ANCHOR_TYPES,
            anchor_top_k=args.anchor_top_k,
            traversal_relations=tuple(part.strip() for part in args.relations.split(",") if part.strip())
            or DEFAULT_TRAVERSAL_RELATIONS,
            max_related=args.max_related,
            max_depth=args.max_depth,
            case_top_k=args.case_top_k,
        )
        if args.save_retrieval:
            save_path = Path(args.save_retrieval)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(retrieval_result, indent=2))

    case_payload = construct_case(
        query_text=query_text,
        retrieval_result=retrieval_result,
        model=args.model,
        temperature=args.temperature,
        case_id=args.case_id,
        candidate_case_limit=args.candidate_case_limit,
    )

    print(json.dumps(case_payload, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(case_payload, indent=2))


if __name__ == "__main__":
    main()
