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

from llm import get_embedding

DEFAULT_GRAPH_PATH = ROOT_DIR / "data" / "hybrid_topic_graph" / "hybrid_topic_graph.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "hybrid_topic_graph" / "node_embeddings.json"

DEFAULT_NODE_TYPES = {
    "style_trait",
    "history_statement",
    "core_belief_statement",
    "intermediate_belief_statement",
    "coping_strategy_statement",
    "situation_statement",
    "automatic_thought_statement",
    "emotion_statement",
    "behavior_statement",
    "sub_topic",
    "core_belief_cluster",
    "intermediate_belief_type",
    "coping_type",
    "emotion_cluster",
    "behavior_pattern",
    "super_topic",
}


def format_type_label(node_type: str) -> str:
    return node_type.replace("_", " ").title()


def load_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_indexes(
    spec: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    nodes_by_id = {node["id"]: node for node in spec["nodes"]}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for edge in spec["edges"]:
        outgoing[edge["source"]].append(edge)
        incoming[edge["target"]].append(edge)

    return nodes_by_id, outgoing, incoming


def labels_for_relation(
    node_id: str,
    relation: str,
    edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    source_side: bool,
) -> list[str]:
    labels: list[str] = []
    for edge in edges:
        if edge["relation"] != relation:
            continue
        other_id = edge["source"] if source_side else edge["target"]
        other = nodes_by_id.get(other_id)
        if other:
            labels.append(other["label"])
    return sorted(set(labels))


def build_node_text(
    node: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    incoming: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        f"Type: {format_type_label(node['type'])}",
        f"Label: {node['label']}",
    ]

    if node.get("layer"):
        lines.append(f"Layer: {node['layer']}")
    if node.get("belief_family"):
        lines.append(f"Belief family: {node['belief_family']}")
    if node.get("variant"):
        lines.append(f"Variant: {node['variant']}")

    text = node.get("text")
    if text:
        lines.append(f"Text: {text}")

    description = node.get("description")
    if description:
        lines.append(f"Description: {description}")

    aliases = node.get("aliases")
    if aliases:
        lines.append(f"Aliases: {', '.join(aliases)}")

    source_cases = node.get("source_case_ids")
    if source_cases and node["type"] != "sub_topic":
        lines.append(f"Source cases: {', '.join(source_cases[:10])}")

    if node["type"] == "sub_topic":
        parent_topics = labels_for_relation(
            node["id"], "contains", incoming.get(node["id"], []), nodes_by_id, source_side=True
        )
        if parent_topics:
            lines.append(f"Parent topics: {', '.join(parent_topics)}")
        return "\n".join(lines)

    if node["type"] == "style_trait":
        linked_cases = labels_for_relation(
            node["id"], "has_style_trait", incoming.get(node["id"], []), nodes_by_id, source_side=True
        )
        if linked_cases:
            lines.append(f"Observed in cases: {', '.join(linked_cases[:10])}")
        return "\n".join(lines)

    associated_topics = labels_for_relation(
        node["id"], "associated_with_topic", outgoing.get(node["id"], []), nodes_by_id, source_side=False
    )
    if associated_topics:
        lines.append(f"Associated topics: {', '.join(associated_topics)}")

    cluster_links = labels_for_relation(
        node["id"], "instance_of", outgoing.get(node["id"], []), nodes_by_id, source_side=False
    )
    if cluster_links:
        lines.append(f"Abstract mappings: {', '.join(cluster_links)}")

    if node["type"] == "history_statement":
        supported = labels_for_relation(
            node["id"], "history_supports_belief", outgoing.get(node["id"], []), nodes_by_id, source_side=False
        )
        if supported:
            lines.append(f"Supports beliefs: {', '.join(supported)}")

    if node["type"] == "core_belief_statement":
        expressed = labels_for_relation(
            node["id"], "belief_expressed_as", outgoing.get(node["id"], []), nodes_by_id, source_side=False
        )
        if expressed:
            lines.append(f"Expressed as: {', '.join(expressed)}")

    if node["type"] == "intermediate_belief_statement":
        thoughts = labels_for_relation(
            node["id"], "intermediate_shapes_thought", outgoing.get(node["id"], []), nodes_by_id, source_side=False
        )
        if thoughts:
            lines.append(f"Shapes thoughts: {', '.join(thoughts)}")

    if node["type"] == "situation_statement":
        triggered = labels_for_relation(
            node["id"], "triggers_automatic_thought", outgoing.get(node["id"], []), nodes_by_id, source_side=False
        )
        if triggered:
            lines.append(f"Triggers thoughts: {', '.join(triggered)}")

    if node["type"] == "automatic_thought_statement":
        emotions = labels_for_relation(
            node["id"], "evokes_emotion", outgoing.get(node["id"], []), nodes_by_id, source_side=False
        )
        behaviors = labels_for_relation(
            node["id"], "influences_behavior", outgoing.get(node["id"], []), nodes_by_id, source_side=False
        )
        if emotions:
            lines.append(f"Evokes emotions: {', '.join(emotions)}")
        if behaviors:
            lines.append(f"Influences behaviors: {', '.join(behaviors)}")

    if node["type"] == "coping_strategy_statement":
        behaviors = labels_for_relation(
            node["id"], "manifests_as_behavior", outgoing.get(node["id"], []), nodes_by_id, source_side=False
        )
        if behaviors:
            lines.append(f"Manifests as: {', '.join(behaviors)}")

    if node["type"] in {
        "core_belief_cluster",
        "intermediate_belief_type",
        "coping_type",
        "emotion_cluster",
        "behavior_pattern",
        "super_topic",
    }:
        related = labels_for_relation(
            node["id"], "suggests_prompt_type", outgoing.get(node["id"], []), nodes_by_id, source_side=False
        )
        if related:
            lines.append(f"Suggested prompt types: {', '.join(related)}")

    return "\n".join(lines)


def parse_node_types(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def build_embedding_index(
    spec: dict[str, Any],
    model: str,
    selected_types: set[str] | None,
    show_progress: bool = False,
) -> dict[str, Any]:
    nodes_by_id, outgoing, incoming = build_indexes(spec)
    records: list[dict[str, Any]] = []
    nodes_to_embed = [
        node
        for node in spec["nodes"]
        if selected_types is None or node["type"] in selected_types
    ]

    if show_progress:
        print(
            f"Embedding {len(nodes_to_embed)} hybrid graph node(s) with model '{model}'"
            f" from graph '{spec['metadata']['name']}'..."
        )

    for index, node in enumerate(nodes_to_embed, start=1):
        text = build_node_text(node, nodes_by_id, outgoing, incoming)
        embedding = get_embedding(text, model=model)
        records.append(
            {
                "id": node["id"],
                "type": node["type"],
                "label": node["label"],
                "layer": node.get("layer"),
                "text": text,
                "embedding": embedding,
            }
        )
        if show_progress:
            print(f"[{index}/{len(nodes_to_embed)}] Embedded {node['id']} ({node['type']})")

    return {
        "metadata": {
            "embedding_model": model,
            "graph_name": spec["metadata"]["name"],
            "node_count": len(records),
            "source_graph_node_count": len(spec["nodes"]),
            "source_graph_edge_count": len(spec["edges"]),
            "selected_node_types": sorted(selected_types) if selected_types is not None else "all",
        },
        "records": records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build embeddings for hybrid topic graph nodes")
    parser.add_argument(
        "--graph",
        default=str(DEFAULT_GRAPH_PATH),
        help="Path to hybrid_topic_graph.json",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where the node embedding index JSON will be written",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model name passed to llm.get_embedding",
    )
    parser.add_argument(
        "--node-types",
        help=(
            "Optional comma-separated node types to embed. "
            "If omitted, embeds the default text-bearing hybrid node types."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress output",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    graph_path = Path(args.graph)
    output_path = Path(args.output)
    selected_types = parse_node_types(args.node_types)
    if selected_types is None:
        selected_types = set(DEFAULT_NODE_TYPES)

    spec = load_graph(graph_path)
    index = build_embedding_index(
        spec=spec,
        model=args.embedding_model,
        selected_types=selected_types,
        show_progress=not args.quiet,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2))

    print(
        json.dumps(
            {
                "written_file": str(output_path),
                "metadata": index["metadata"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
