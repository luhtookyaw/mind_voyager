from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm import get_embedding

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_PATH = ROOT_DIR / "data" / "topic_graph" / "topic_graph.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "topic_graph" / "node_embeddings.json"


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
        labels.append(nodes_by_id[other_id]["label"])
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

    if node.get("description"):
        lines.append(f"Description: {node['description']}")

    aliases = node.get("aliases")
    if aliases:
        lines.append(f"Aliases: {', '.join(aliases)}")

    prompt_templates = node.get("prompt_templates")
    if prompt_templates:
        lines.append(f"Prompt templates: {' | '.join(prompt_templates)}")

    parent_topics = labels_for_relation(
        node["id"],
        "contains",
        incoming.get(node["id"], []),
        nodes_by_id,
        source_side=True,
    )
    if parent_topics:
        lines.append(f"Parent topics: {', '.join(parent_topics)}")

    contained_subtopics = labels_for_relation(
        node["id"],
        "contains",
        outgoing.get(node["id"], []),
        nodes_by_id,
        source_side=False,
    )
    if contained_subtopics:
        lines.append(f"Sub-topics: {', '.join(contained_subtopics)}")

    associated = labels_for_relation(
        node["id"],
        "associated_with",
        outgoing.get(node["id"], []),
        nodes_by_id,
        source_side=False,
    )
    if associated:
        lines.append(f"Associated concepts: {', '.join(associated)}")

    evokes = labels_for_relation(
        node["id"],
        "evokes",
        outgoing.get(node["id"], []),
        nodes_by_id,
        source_side=False,
    )
    if evokes:
        lines.append(f"Evokes emotions: {', '.join(evokes)}")

    manifests = labels_for_relation(
        node["id"],
        "manifests_as",
        outgoing.get(node["id"], []),
        nodes_by_id,
        source_side=False,
    )
    if manifests:
        lines.append(f"Behavior patterns: {', '.join(manifests)}")

    prompt_types = labels_for_relation(
        node["id"],
        "suggests_prompt_type",
        outgoing.get(node["id"], []),
        nodes_by_id,
        source_side=False,
    )
    if prompt_types:
        lines.append(f"Suggested prompt types: {', '.join(prompt_types)}")

    leads_to = labels_for_relation(
        node["id"],
        "leads_to",
        outgoing.get(node["id"], []),
        nodes_by_id,
        source_side=False,
    )
    if leads_to:
        lines.append(f"Leads to: {', '.join(leads_to)}")

    maintained_by = labels_for_relation(
        node["id"],
        "maintained_by",
        outgoing.get(node["id"], []),
        nodes_by_id,
        source_side=False,
    )
    if maintained_by:
        lines.append(f"Maintained by: {', '.join(maintained_by)}")

    return "\n".join(lines)


def parse_node_types(raw: str | None) -> set[str] | None:
    if not raw:
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
            f"Embedding {len(nodes_to_embed)} node(s) with model '{model}'"
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
                "text": text,
                "embedding": embedding,
            }
        )
        if show_progress:
            print(
                f"[{index}/{len(nodes_to_embed)}] Embedded {node['id']}"
                f" ({node['type']})"
            )

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
    parser = argparse.ArgumentParser(description="Build local embeddings for topic graph nodes")
    parser.add_argument(
        "--graph",
        default=str(DEFAULT_GRAPH_PATH),
        help="Path to topic_graph.json",
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
        help="Optional comma-separated node types to embed, for example sub_topic,super_topic",
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
