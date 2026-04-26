from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm import cosine_similarity, get_embedding


DEFAULT_GRAPH_PATH = ROOT_DIR / "data" / "topic_graph" / "topic_graph.json"
DEFAULT_INDEX_PATH = ROOT_DIR / "data" / "topic_graph" / "node_embeddings.json"

DEFAULT_RELATIONS = (
    "evokes",
    "associated_with",
    "manifests_as",
    "suggests_prompt_type",
)

PROMPT_PRIORITY = {
    "emotion_cluster": 0,
    "coping_type": 1,
    "behavior_pattern": 2,
    "intermediate_belief_type": 3,
    "prompt_type": 4,
    "core_belief_cluster": 5,
    "sub_topic": 6,
    "super_topic": 7,
}

RELATION_PRIORITY = {
    "evokes": 0,
    "associated_with": 1,
    "manifests_as": 2,
    "suggests_prompt_type": 3,
    "leads_to": 4,
    "maintained_by": 5,
    "contains": 6,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def build_indexes(
    spec: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    nodes_by_id = {node["id"]: node for node in spec["nodes"]}
    edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for edge in spec["edges"]:
        edges_by_source[edge["source"]].append(edge)

    return nodes_by_id, edges_by_source


def score_nodes(
    query: str,
    index: dict[str, Any],
    allowed_types: set[str],
) -> list[dict[str, Any]]:
    model = index["metadata"]["embedding_model"]
    query_embedding = get_embedding(query, model=model)
    scored: list[dict[str, Any]] = []

    for record in index["records"]:
        if allowed_types and record["type"] not in allowed_types:
            continue
        scored.append(
            {
                "id": record["id"],
                "type": record["type"],
                "label": record["label"],
                "score": cosine_similarity(query_embedding, record["embedding"]),
                "text": record["text"],
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def select_anchor_nodes(
    query: str,
    index: dict[str, Any],
    anchor_types: tuple[str, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    allowed_types = set(anchor_types)
    return score_nodes(query=query, index=index, allowed_types=allowed_types)[:top_k]


def expand_anchor_nodes(
    anchors: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    edges_by_source: dict[str, list[dict[str, Any]]],
    relation_filter: tuple[str, ...],
    per_anchor_limit: int,
    type_limits: dict[str, int],
) -> list[dict[str, Any]]:
    allowed_relations = set(relation_filter)
    expanded: dict[str, dict[str, Any]] = {}
    counts_by_type: dict[str, int] = defaultdict(int)

    for anchor in anchors:
        candidates: list[dict[str, Any]] = []
        for edge in edges_by_source.get(anchor["id"], []):
            if allowed_relations and edge["relation"] not in allowed_relations:
                continue
            target_node = nodes_by_id[edge["target"]]
            target_type = target_node["type"]
            limit = type_limits.get(target_type)
            if limit is not None and counts_by_type[target_type] >= limit:
                continue

            candidates.append(
                {
                    "id": target_node["id"],
                    "label": target_node["label"],
                    "type": target_type,
                    "relation": edge["relation"],
                    "weight": edge["weight"],
                    "via_anchor_id": anchor["id"],
                    "via_anchor_label": anchor["label"],
                    "anchor_score": anchor["score"],
                    "combined_score": anchor["score"] * edge["weight"],
                }
            )

        candidates.sort(
            key=lambda item: (
                item["combined_score"],
                -RELATION_PRIORITY.get(item["relation"], 999),
            ),
            reverse=True,
        )

        taken = 0
        for candidate in candidates:
            if taken >= per_anchor_limit:
                break
            existing = expanded.get(candidate["id"])
            target_type = candidate["type"]
            limit = type_limits.get(target_type)

            if existing is None and limit is not None and counts_by_type[target_type] >= limit:
                continue

            if existing is None or candidate["combined_score"] > existing["combined_score"]:
                if existing is None:
                    counts_by_type[target_type] += 1
                expanded[candidate["id"]] = candidate
            taken += 1

    expanded_nodes = sorted(
        expanded.values(),
        key=lambda item: (
            PROMPT_PRIORITY.get(item["type"], 999),
            -item["combined_score"],
            item["label"],
        ),
    )
    return expanded_nodes


def build_type_limits(
    emotion_limit: int,
    coping_limit: int,
    behavior_limit: int,
    intermediate_limit: int,
    prompt_limit: int,
    core_belief_limit: int,
) -> dict[str, int]:
    return {
        "emotion_cluster": emotion_limit,
        "coping_type": coping_limit,
        "behavior_pattern": behavior_limit,
        "intermediate_belief_type": intermediate_limit,
        "prompt_type": prompt_limit,
        "core_belief_cluster": core_belief_limit,
    }


def group_by_type(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["type"]].append(item)
    return dict(grouped)


def build_prompt_style_lines(
    expanded: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    max_prompt_types: int = 2,
) -> list[str]:
    prompt_nodes = [item for item in expanded if item["type"] == "prompt_type"][:max_prompt_types]
    if not prompt_nodes:
        return []

    lines = ["- Possible follow-up styles:"]
    for item in prompt_nodes:
        node = nodes_by_id[item["id"]]
        templates = node.get("prompt_templates") or []
        if not templates:
            continue
        lines.append(f'  - {item["label"]}: "{templates[0]}"')
    return lines


def build_prompt_context(
    anchors: list[dict[str, Any]],
    expanded: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> str:
    grouped = group_by_type(expanded)

    def labels(node_type: str) -> str:
        values = grouped.get(node_type, [])
        if not values:
            return "none"
        return ", ".join(item["label"] for item in values)

    anchor_labels = ", ".join(anchor["label"] for anchor in anchors) if anchors else "none"
    lines = [
        "Retrieved topic graph context:",
        f"- Likely sub-topics: {anchor_labels}",
        f"- Likely emotions: {labels('emotion_cluster')}",
        f"- Likely coping styles: {labels('coping_type')}",
        f"- Likely behavior patterns: {labels('behavior_pattern')}",
        f"- Likely intermediate beliefs: {labels('intermediate_belief_type')}",
        f"- Suggested prompt types: {labels('prompt_type')}",
        f"- Possible core beliefs: {labels('core_belief_cluster')}",
    ]
    lines.extend(build_prompt_style_lines(expanded, nodes_by_id))
    return "\n".join(lines)


def retrieve_topic_graph_context(
    query: str,
    graph_path: Path,
    index_path: Path,
    anchor_types: tuple[str, ...],
    anchor_top_k: int,
    relation_filter: tuple[str, ...],
    per_anchor_limit: int,
    type_limits: dict[str, int],
) -> dict[str, Any]:
    spec = load_json(graph_path)
    index = load_json(index_path)
    nodes_by_id, edges_by_source = build_indexes(spec)

    anchors = select_anchor_nodes(
        query=query,
        index=index,
        anchor_types=anchor_types,
        top_k=anchor_top_k,
    )
    expanded = expand_anchor_nodes(
        anchors=anchors,
        nodes_by_id=nodes_by_id,
        edges_by_source=edges_by_source,
        relation_filter=relation_filter,
        per_anchor_limit=per_anchor_limit,
        type_limits=type_limits,
    )

    return {
        "query": query,
        "graph": {
            "path": str(graph_path),
            "name": spec["metadata"]["name"],
            "node_count": spec["metadata"]["node_count"],
            "edge_count": spec["metadata"]["edge_count"],
        },
        "embedding_index": {
            "path": str(index_path),
            "model": index["metadata"]["embedding_model"],
            "node_count": index["metadata"]["node_count"],
        },
        "retrieval_config": {
            "anchor_types": list(anchor_types),
            "anchor_top_k": anchor_top_k,
            "relation_filter": list(relation_filter),
            "per_anchor_limit": per_anchor_limit,
            "type_limits": type_limits,
        },
        "anchors": anchors,
        "expanded": expanded,
        "grouped_expanded": group_by_type(expanded),
        "prompt_context": build_prompt_context(anchors, expanded, nodes_by_id),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve therapist-prompt context from the topic graph by embedding a "
            "query, retrieving anchor nodes, and expanding to related graph components."
        )
    )
    parser.add_argument("--query", required=True, help="Client utterance or query text to retrieve against")
    parser.add_argument(
        "--graph",
        default=str(DEFAULT_GRAPH_PATH),
        help="Path to topic_graph.json",
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to node_embeddings.json",
    )
    parser.add_argument(
        "--anchor-types",
        default="sub_topic",
        help="Comma-separated node types used for first-stage retrieval",
    )
    parser.add_argument(
        "--anchor-top-k",
        type=int,
        default=3,
        help="Number of anchor nodes to keep after first-stage retrieval",
    )
    parser.add_argument(
        "--relations",
        default=",".join(DEFAULT_RELATIONS),
        help="Comma-separated edge relations to use during graph expansion",
    )
    parser.add_argument(
        "--per-anchor-limit",
        type=int,
        default=6,
        help="Maximum number of expanded items considered from each anchor",
    )
    parser.add_argument("--emotion-limit", type=int, default=2, help="Maximum emotion nodes to return")
    parser.add_argument("--coping-limit", type=int, default=2, help="Maximum coping nodes to return")
    parser.add_argument("--behavior-limit", type=int, default=2, help="Maximum behavior nodes to return")
    parser.add_argument(
        "--intermediate-limit",
        type=int,
        default=2,
        help="Maximum intermediate belief nodes to return",
    )
    parser.add_argument("--prompt-limit", type=int, default=2, help="Maximum prompt type nodes to return")
    parser.add_argument(
        "--core-belief-limit",
        type=int,
        default=1,
        help="Maximum core belief nodes to return",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the retrieval result as JSON",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = retrieve_topic_graph_context(
        query=args.query,
        graph_path=Path(args.graph),
        index_path=Path(args.index),
        anchor_types=parse_csv(args.anchor_types) or ("sub_topic",),
        anchor_top_k=args.anchor_top_k,
        relation_filter=parse_csv(args.relations) or DEFAULT_RELATIONS,
        per_anchor_limit=args.per_anchor_limit,
        type_limits=build_type_limits(
            emotion_limit=args.emotion_limit,
            coping_limit=args.coping_limit,
            behavior_limit=args.behavior_limit,
            intermediate_limit=args.intermediate_limit,
            prompt_limit=args.prompt_limit,
            core_belief_limit=args.core_belief_limit,
        ),
    )

    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
