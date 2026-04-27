from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm import cosine_similarity, get_embedding


DEFAULT_GRAPH_PATH = ROOT_DIR / "data" / "hybrid_topic_graph" / "hybrid_topic_graph.json"
DEFAULT_INDEX_PATH = ROOT_DIR / "data" / "hybrid_topic_graph" / "node_embeddings.json"

DEFAULT_ANCHOR_TYPES = (
    "style_trait",
    "situation_statement",
    "automatic_thought_statement",
    "emotion_statement",
    "behavior_statement",
    "history_statement",
    "core_belief_statement",
    "intermediate_belief_statement",
    "coping_strategy_statement",
    "sub_topic",
    "core_belief_cluster",
    "emotion_cluster",
    "behavior_pattern",
    "coping_type",
)

DEFAULT_TRAVERSAL_RELATIONS = (
    "instance_of",
    "associated_with_topic",
    "has_case_record",
    "has_external_episode",
    "has_style_trait",
    "has_history",
    "has_core_belief",
    "has_intermediate_belief_default",
    "has_intermediate_belief_depression",
    "has_coping_strategy",
    "has_situation",
    "has_automatic_thought",
    "has_emotion",
    "has_behavior",
    "history_supports_belief",
    "belief_expressed_as",
    "intermediate_shapes_thought",
    "triggers_automatic_thought",
    "evokes_emotion",
    "influences_behavior",
    "manifests_as_behavior",
)

CASE_RELATIONS = {
    "has_history",
    "has_core_belief",
    "has_intermediate_belief_default",
    "has_intermediate_belief_depression",
    "has_coping_strategy",
    "has_situation",
    "has_automatic_thought",
    "has_emotion",
    "has_behavior",
}

DISPLAY_TYPE_ORDER = {
    "situation_statement": 0,
    "automatic_thought_statement": 1,
    "emotion_statement": 2,
    "behavior_statement": 3,
    "history_statement": 4,
    "core_belief_statement": 5,
    "intermediate_belief_statement": 6,
    "coping_strategy_statement": 7,
    "sub_topic": 8,
    "core_belief_cluster": 9,
    "emotion_cluster": 10,
    "behavior_pattern": 11,
    "coping_type": 12,
    "style_trait": 13,
    "case_record": 14,
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
    dict[str, list[dict[str, Any]]],
]:
    nodes_by_id = {node["id"]: node for node in spec["nodes"]}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for edge in spec["edges"]:
        outgoing[edge["source"]].append(edge)
        incoming[edge["target"]].append(edge)

    return nodes_by_id, outgoing, incoming


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
                "layer": record.get("layer"),
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
    return score_nodes(query=query, index=index, allowed_types=set(anchor_types))[:top_k]


def traverse_related_nodes(
    anchors: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    incoming: dict[str, list[dict[str, Any]]],
    allowed_relations: set[str],
    max_depth: int,
    max_related: int,
) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}

    for anchor in anchors:
        queue = deque([(anchor["id"], 0, anchor["score"], [anchor["id"]], [])])
        seen_depth: dict[str, int] = {anchor["id"]: 0}

        while queue:
            node_id, depth, score, path_nodes, path_edges = queue.popleft()
            if depth >= max_depth:
                continue

            neighbors = []
            for edge in outgoing.get(node_id, []):
                if allowed_relations and edge["relation"] not in allowed_relations:
                    continue
                neighbors.append(("out", edge))
            for edge in incoming.get(node_id, []):
                if allowed_relations and edge["relation"] not in allowed_relations:
                    continue
                neighbors.append(("in", edge))

            for direction, edge in neighbors:
                other_id = edge["target"] if direction == "out" else edge["source"]
                next_depth = depth + 1
                next_score = score * edge["weight"]
                if other_id in seen_depth and seen_depth[other_id] <= next_depth:
                    continue
                seen_depth[other_id] = next_depth

                if other_id != anchor["id"]:
                    node = nodes_by_id[other_id]
                    candidate = {
                        "id": other_id,
                        "label": node["label"],
                        "type": node["type"],
                        "layer": node.get("layer"),
                        "best_score": next_score,
                        "via_anchor_id": anchor["id"],
                        "via_anchor_label": anchor["label"],
                        "depth": next_depth,
                        "path_nodes": path_nodes + [other_id],
                        "path_edges": path_edges + [f"{direction}:{edge['relation']}"],
                    }
                    existing = best.get(other_id)
                    if existing is None or candidate["best_score"] > existing["best_score"]:
                        best[other_id] = candidate

                queue.append(
                    (
                        other_id,
                        next_depth,
                        next_score,
                        path_nodes + [other_id],
                        path_edges + [f"{direction}:{edge['relation']}"],
                    )
                )

    items = sorted(
        best.values(),
        key=lambda item: (
            DISPLAY_TYPE_ORDER.get(item["type"], 999),
            -item["best_score"],
            item["label"],
        ),
    )
    return items[:max_related]


def collect_case_records(
    anchors: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    incoming: dict[str, list[dict[str, Any]]],
    allowed_relations: set[str],
    max_depth: int,
    top_k: int,
) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    for anchor in anchors:
        queue = deque([(anchor["id"], 0, anchor["score"], [], [])])
        seen_depth: dict[str, int] = {anchor["id"]: 0}

        while queue:
            node_id, depth, score, path_nodes, path_edges = queue.popleft()
            node = nodes_by_id[node_id]
            if node["type"] == "case_record" and node_id != anchor["id"]:
                existing = cases.get(node_id)
                candidate = {
                    "id": node_id,
                    "label": node["label"],
                    "score": score,
                    "via_anchor_id": anchor["id"],
                    "via_anchor_label": anchor["label"],
                    "depth": depth,
                    "path_nodes": path_nodes + [node_id],
                    "path_edges": path_edges,
                }
                if existing is None or candidate["score"] > existing["score"]:
                    cases[node_id] = candidate
                continue

            if depth >= max_depth:
                continue

            neighbors = []
            for edge in outgoing.get(node_id, []):
                if allowed_relations and edge["relation"] not in allowed_relations:
                    continue
                neighbors.append(("out", edge))
            for edge in incoming.get(node_id, []):
                if allowed_relations and edge["relation"] not in allowed_relations:
                    continue
                neighbors.append(("in", edge))

            for direction, edge in neighbors:
                other_id = edge["target"] if direction == "out" else edge["source"]
                next_depth = depth + 1
                if other_id in seen_depth and seen_depth[other_id] <= next_depth:
                    continue
                seen_depth[other_id] = next_depth
                queue.append(
                    (
                        other_id,
                        next_depth,
                        score * edge["weight"],
                        path_nodes + [node_id],
                        path_edges + [f"{direction}:{edge['relation']}"],
                    )
                )

    return sorted(cases.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def extract_case_bundle(
    case_node_id: str,
    nodes_by_id: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "case_record": nodes_by_id[case_node_id],
        "style_traits": [],
        "history": [],
        "core_beliefs": [],
        "intermediate_beliefs": [],
        "coping_strategies": [],
        "situation": [],
        "automatic_thought": [],
        "emotions": [],
        "behavior": [],
        "ontology_links": defaultdict(list),
    }

    episode_ids: list[str] = []
    for edge in outgoing.get(case_node_id, []):
        relation = edge["relation"]
        target = nodes_by_id[edge["target"]]
        if relation == "has_history":
            bundle["history"].append(target)
        elif relation == "has_style_trait":
            bundle["style_traits"].append(target)
        elif relation == "has_core_belief":
            bundle["core_beliefs"].append(target)
        elif relation in {"has_intermediate_belief_default", "has_intermediate_belief_depression"}:
            bundle["intermediate_beliefs"].append(target)
        elif relation == "has_coping_strategy":
            bundle["coping_strategies"].append(target)
        elif relation == "has_external_episode":
            episode_ids.append(target["id"])

    for episode_id in episode_ids:
        for edge in outgoing.get(episode_id, []):
            target = nodes_by_id[edge["target"]]
            if edge["relation"] == "has_situation":
                bundle["situation"].append(target)
            elif edge["relation"] == "has_automatic_thought":
                bundle["automatic_thought"].append(target)
            elif edge["relation"] == "has_emotion":
                bundle["emotions"].append(target)
            elif edge["relation"] == "has_behavior":
                bundle["behavior"].append(target)

    for section in [
        "history",
        "core_beliefs",
        "intermediate_beliefs",
        "coping_strategies",
        "situation",
        "automatic_thought",
        "emotions",
        "behavior",
        "style_traits",
    ]:
        for node in bundle[section]:
            for edge in outgoing.get(node["id"], []):
                if edge["relation"] in {"instance_of", "associated_with_topic"}:
                    bundle["ontology_links"][node["id"]].append(
                        {
                            "relation": edge["relation"],
                            "target": nodes_by_id[edge["target"]],
                            "weight": edge["weight"],
                        }
                    )

    bundle["ontology_links"] = dict(bundle["ontology_links"])
    return bundle


def compact_node_text(node: dict[str, Any]) -> str:
    return node.get("text") or node.get("label") or ""


def build_prompt_context(
    anchors: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
    case_bundles: list[dict[str, Any]],
) -> str:
    anchor_labels = ", ".join(f"{item['label']} ({item['type']})" for item in anchors) or "none"
    lines = [
        "Retrieved hybrid graph context:",
        f"- Top matched nodes: {anchor_labels}",
    ]

    if candidate_cases:
        lines.append("- Candidate case matches:")
        for case in candidate_cases[:3]:
            lines.append(f"  - {case['label']} (score={case['score']:.4f}) via {case['via_anchor_label']}")

    if case_bundles:
        top_case = case_bundles[0]
        case_label = top_case["case_record"]["label"]
        lines.append(f"- Top case profile: {case_label}")

        def section_text(section: str) -> str:
            values = top_case[section]
            if not values:
                return "none"
            return " | ".join(compact_node_text(node) for node in values[:2])

        lines.append(f"- Possible style traits: {section_text('style_traits')}")
        lines.append(f"- Situation examples: {section_text('situation')}")
        lines.append(f"- Automatic thought examples: {section_text('automatic_thought')}")
        lines.append(f"- Emotion examples: {section_text('emotions')}")
        lines.append(f"- Behavior examples: {section_text('behavior')}")
        lines.append(f"- Possible history: {section_text('history')}")
        lines.append(f"- Possible core beliefs: {section_text('core_beliefs')}")
        lines.append(f"- Possible intermediate beliefs: {section_text('intermediate_beliefs')}")
        lines.append(f"- Possible coping: {section_text('coping_strategies')}")

    return "\n".join(lines)


def retrieve_hybrid_topic_graph_context(
    query: str,
    graph_path: Path,
    index_path: Path,
    anchor_types: tuple[str, ...],
    anchor_top_k: int,
    traversal_relations: tuple[str, ...],
    max_related: int,
    max_depth: int,
    case_top_k: int,
) -> dict[str, Any]:
    spec = load_json(graph_path)
    index = load_json(index_path)
    nodes_by_id, outgoing, incoming = build_indexes(spec)
    allowed_relations = set(traversal_relations)

    anchors = select_anchor_nodes(
        query=query,
        index=index,
        anchor_types=anchor_types,
        top_k=anchor_top_k,
    )

    related = traverse_related_nodes(
        anchors=anchors,
        nodes_by_id=nodes_by_id,
        outgoing=outgoing,
        incoming=incoming,
        allowed_relations=allowed_relations,
        max_depth=max_depth,
        max_related=max_related,
    )

    candidate_cases = collect_case_records(
        anchors=anchors,
        nodes_by_id=nodes_by_id,
        outgoing=outgoing,
        incoming=incoming,
        allowed_relations=allowed_relations,
        max_depth=max_depth,
        top_k=case_top_k,
    )
    case_bundles = [extract_case_bundle(item["id"], nodes_by_id, outgoing) for item in candidate_cases]

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
            "traversal_relations": list(traversal_relations),
            "max_related": max_related,
            "max_depth": max_depth,
            "case_top_k": case_top_k,
        },
        "anchors": anchors,
        "related_nodes": related,
        "candidate_cases": candidate_cases,
        "case_bundles": case_bundles,
        "prompt_context": build_prompt_context(anchors, candidate_cases, case_bundles),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve case-grounded context from the hybrid topic graph by "
            "embedding a query, selecting anchor nodes, then expanding through "
            "bridge and case-structure edges."
        )
    )
    parser.add_argument("--query", required=True, help="Post text or extracted query text")
    parser.add_argument(
        "--graph",
        default=str(DEFAULT_GRAPH_PATH),
        help="Path to hybrid_topic_graph.json",
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to hybrid node_embeddings.json",
    )
    parser.add_argument(
        "--anchor-types",
        default=",".join(DEFAULT_ANCHOR_TYPES),
        help="Comma-separated node types used for first-stage retrieval",
    )
    parser.add_argument(
        "--anchor-top-k",
        type=int,
        default=5,
        help="Number of anchor nodes to keep after first-stage retrieval",
    )
    parser.add_argument(
        "--relations",
        default=",".join(DEFAULT_TRAVERSAL_RELATIONS),
        help="Comma-separated relations traversed during expansion",
    )
    parser.add_argument(
        "--max-related",
        type=int,
        default=40,
        help="Maximum number of related nodes to return",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum traversal depth from each anchor",
    )
    parser.add_argument(
        "--case-top-k",
        type=int,
        default=5,
        help="Maximum number of candidate case records to return",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the retrieval result JSON",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = retrieve_hybrid_topic_graph_context(
        query=args.query,
        graph_path=Path(args.graph),
        index_path=Path(args.index),
        anchor_types=parse_csv(args.anchor_types) or DEFAULT_ANCHOR_TYPES,
        anchor_top_k=args.anchor_top_k,
        traversal_relations=parse_csv(args.relations) or DEFAULT_TRAVERSAL_RELATIONS,
        max_related=args.max_related,
        max_depth=args.max_depth,
        case_top_k=args.case_top_k,
    )

    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
