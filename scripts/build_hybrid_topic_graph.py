from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import build_topic_graph as ontology

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT_DIR / "data" / "Patient_Psi_CM_Dataset.json"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "hybrid_topic_graph"


BELIEF_STATEMENT_TO_CLUSTER = {
    "i am a failure, loser.": "failure_identity",
    "i am a victim.": "helplessness",
    "i am defective.": "defectiveness",
    "i am helpless.": "helplessness",
    "i am incompetent.": "helplessness",
    "i am needy.": "helplessness",
    "i am out of control.": "loss_of_control",
    "i am powerless, weak, vulnerable.": "helplessness",
    "i am trapped.": "helplessness",
    "i am bound to be abandoned.": "abandonment",
    "i am bound to be alone.": "abandonment",
    "i am bound to be rejected.": "unlovability",
    "i am undesirable, unwanted.": "unlovability",
    "i am unlovable.": "unlovability",
    "i am bad - dangerous, toxic, evil.": "worthlessness",
    "i am immoral.": "worthlessness",
    "i am worthless, waste.": "worthlessness",
}


EMOTION_HINTS = {
    "anxiety": ["anxious", "worried", "fearful", "scared", "tense", "panic"],
    "fear": ["afraid", "terror", "alarmed", "fear"],
    "shame": ["ashamed", "embarrassed", "humiliated"],
    "sadness": ["sad", "down", "unhappy", "grief", "discouraged"],
    "anger": ["angry", "mad", "frustrated", "rage", "resentful"],
    "irritability": ["irritated", "annoyed", "agitated", "touchy"],
    "guilt": ["guilty", "remorseful", "regretful"],
    "loneliness": ["lonely", "alone", "isolated", "disconnected"],
    "hopelessness": ["hopeless", "defeated", "futile"],
    "numbness": ["numb", "empty", "shut down"],
}


INTERMEDIATE_HINTS = {
    "rejection_rule": ["reject", "true self", "show my true self", "approval", "accepted"],
    "avoidance_rule": ["avoid", "stay away", "not put myself", "step away"],
    "perfectionism_rule": ["perfect", "mistake", "successful", "competent"],
    "control_rule": ["control", "under control", "must manage", "stay clean"],
    "self_silencing_rule": ["keep it to myself", "hide", "do not say", "stay quiet"],
    "mistrust_rule": ["cannot trust", "betray", "harm me", "against me"],
    "hopelessness_rule": ["nothing i can do", "why even try", "futile", "bad things just happen"],
    "dependency_rule": ["need support", "need them", "cannot cope alone", "depend on"],
    "overresponsibility_rule": ["must fix", "must manage others", "my responsibility"],
    "emotional_inhibition_rule": ["do not feel", "do not express", "should hide emotions"],
}


COPING_HINTS = {
    "avoidance": ["avoid", "distancing", "stay away", "not respond", "ignored"],
    "withdrawal": ["withdraw", "pull away", "distance", "isolate"],
    "escape": ["escape", "distract", "zone out", "fantasy", "numb"],
    "suppression": ["push down", "suppress", "hold in", "bury"],
    "people_pleasing": ["approval", "appease", "keep the peace", "please"],
    "reassurance_seeking": ["reassurance", "check if", "need confirmation"],
    "overplanning": ["plan", "planning", "schedule", "prepared"],
    "self_attack": ["beat myself up", "self-criticism", "harsh to myself"],
    "substance_use_coping": ["substance", "drunk", "beer", "drink", "using"],
    "emotional_control": ["control my emotions", "keep control", "calm myself down"],
    "safety_behaviors": ["safety", "protect myself", "stay guarded", "monitor"],
    "rumination": ["ruminate", "overthink", "replay", "think about it constantly"],
    "help_seeking": ["seek support", "ask for help", "talk to someone"],
    "activity_scheduling": ["pleasant activities", "schedule", "plan my day"],
}


BEHAVIOR_HINTS = {
    "social_avoidance": ["ignored the invitation", "did not respond", "skip", "avoid gathering"],
    "conflict_avoidance": ["gave an excuse", "avoid confrontation", "did not bring it up"],
    "isolation": ["ignored phone calls", "stay alone", "pull away", "distancing himself"],
    "self_attacking_behavior": ["punish myself", "beat myself up"],
    "overworking": ["worked more", "kept working", "overworking"],
    "procrastination": ["put it off", "did not do anything", "delay"],
    "checking_monitoring": ["check", "monitor", "scan", "watch closely"],
    "appeasement": ["apologize", "placate", "people-please"],
    "emotional_withdrawal": ["shut down", "emotionally withdrew", "zoning out"],
    "relapse_behavior": ["got drunk", "had a beer", "used again"],
    "routine_control": ["schedule", "rigid routine", "plan the day"],
    "reassurance_loop": ["kept asking", "repeatedly check if", "seek reassurance"],
}


HYBRID_GRAPH_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://mind-voyager.local/schemas/hybrid_topic_graph.schema.json",
    "title": "MindVoyager Hybrid Topic Graph",
    "type": "object",
    "required": ["metadata", "nodes", "edges"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["name", "description", "node_count", "edge_count"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "node_count": {"type": "integer", "minimum": 0},
                "edge_count": {"type": "integer", "minimum": 0},
                "dataset_path": {"type": "string"},
                "base_ontology_name": {"type": "string"},
                "layer_counts": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 0},
                },
            },
            "additionalProperties": True,
        },
        "nodes": {
            "type": "array",
            "items": {"$ref": "#/$defs/node"},
        },
        "edges": {
            "type": "array",
            "items": {"$ref": "#/$defs/edge"},
        },
    },
    "$defs": {
        "node": {
            "type": "object",
            "required": ["id", "type", "label", "layer"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "label": {"type": "string"},
                "layer": {
                    "type": "string",
                    "enum": ["ontology", "dataset_grounded", "bridge"],
                },
                "description": {"type": "string"},
                "text": {"type": "string"},
                "canonical_text": {"type": "string"},
                "source_dataset": {"type": "string"},
                "source_case_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "patient_name": {"type": "string"},
                "patient_group": {"type": "string"},
                "belief_family": {"type": "string"},
                "variant": {"type": "string"},
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "prompt_templates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": True,
        },
        "edge": {
            "type": "object",
            "required": ["source", "target", "relation", "weight", "layer"],
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "relation": {"type": "string"},
                "weight": {"type": "number"},
                "layer": {
                    "type": "string",
                    "enum": ["ontology", "dataset_grounded", "bridge", "causal"],
                },
                "source_case_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": False,
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify_text(text: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", normalize_text(text).lower()).strip("_")
    base = base[:60].rstrip("_") or "text"
    digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:10]
    return f"{base}_{digest}"


def add_node(nodes: dict[str, dict[str, Any]], node_id: str, **attrs: Any) -> None:
    if node_id not in nodes:
        nodes[node_id] = {"id": node_id}
    node = nodes[node_id]
    for key, value in attrs.items():
        if value is None:
            continue
        if key == "source_case_ids":
            existing = set(node.get("source_case_ids", []))
            existing.update(value)
            node["source_case_ids"] = sorted(existing)
        elif key == "aliases":
            existing = set(node.get("aliases", []))
            existing.update(value)
            node["aliases"] = sorted(existing)
        else:
            node[key] = value


def add_edge(
    edges: dict[tuple[str, str, str, str], dict[str, Any]],
    source: str,
    target: str,
    relation: str,
    weight: float,
    layer: str,
    source_case_id: str | None = None,
) -> None:
    key = (source, target, relation, layer)
    if key not in edges:
        edges[key] = {
            "source": source,
            "target": target,
            "relation": relation,
            "weight": weight,
            "layer": layer,
            "source_case_ids": [],
        }
    edge = edges[key]
    edge["weight"] = max(edge["weight"], weight)
    if source_case_id and source_case_id not in edge["source_case_ids"]:
        edge["source_case_ids"].append(source_case_id)
        edge["source_case_ids"].sort()


def statement_node_id(prefix: str, text: str) -> str:
    return f"{prefix}::{slugify_text(text)}"


def score_matches(text: str, hints: dict[str, list[str]]) -> list[str]:
    text_l = normalize_text(text).lower()
    matches: list[tuple[int, str]] = []
    for target, keywords in hints.items():
        score = sum(1 for keyword in keywords if keyword in text_l)
        if score:
            matches.append((score, target))
    return [target for _, target in sorted(matches, reverse=True)]


def match_emotion_clusters(text: str) -> list[str]:
    return score_matches(text, EMOTION_HINTS)


def match_intermediate_types(text: str) -> list[str]:
    return score_matches(text, INTERMEDIATE_HINTS)


def match_coping_types(text: str) -> list[str]:
    return score_matches(text, COPING_HINTS)


def match_behavior_patterns(text: str) -> list[str]:
    return score_matches(text, BEHAVIOR_HINTS)


def build_subtopic_index(base_spec: dict[str, Any]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for node in base_spec["nodes"]:
        if node["type"] != "sub_topic":
            continue
        terms = set(node.get("aliases", []))
        terms.add(node["label"])
        terms.add(node["id"].replace("_", " "))
        index[node["id"]] = {normalize_text(term).lower() for term in terms}
    return index


def match_subtopics(text: str, subtopic_index: dict[str, set[str]]) -> list[str]:
    text_l = normalize_text(text).lower()
    hits: list[str] = []
    for subtopic_id, terms in subtopic_index.items():
        if any(term and term in text_l for term in terms):
            hits.append(subtopic_id)
    return sorted(hits)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of case rows in {path}")
    return payload


def build_hybrid_graph_spec(dataset_path: Path) -> dict[str, Any]:
    base_spec = ontology.build_graph_spec()
    rows = load_dataset(dataset_path)

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    subtopic_index = build_subtopic_index(base_spec)

    for node in base_spec["nodes"]:
        add_node(nodes, node["id"], layer="ontology", **{k: v for k, v in node.items() if k != "id"})
    for edge in base_spec["edges"]:
        add_edge(
            edges,
            edge["source"],
            edge["target"],
            edge["relation"],
            edge["weight"],
            layer="ontology",
        )

    for row in rows:
        case_id = str(row["id"])
        patient_group = case_id.split("-")[0]
        patient_name = row["name"]
        patient_node_id = f"patient::{patient_group}"
        case_node_id = f"case::{case_id}"
        episode_node_id = f"episode::{case_id}"

        add_node(
            nodes,
            patient_node_id,
            type="patient_profile",
            label=patient_name,
            layer="dataset_grounded",
            patient_name=patient_name,
            patient_group=patient_group,
            source_dataset="Patient_Psi_CM_Dataset",
            source_case_ids=[case_id],
        )
        add_node(
            nodes,
            case_node_id,
            type="case_record",
            label=f"{patient_name} {case_id}",
            layer="dataset_grounded",
            patient_name=patient_name,
            patient_group=patient_group,
            source_dataset="Patient_Psi_CM_Dataset",
            source_case_ids=[case_id],
            case_id=case_id,
        )
        add_node(
            nodes,
            episode_node_id,
            type="external_episode",
            label=f"Episode {case_id}",
            layer="dataset_grounded",
            source_dataset="Patient_Psi_CM_Dataset",
            source_case_ids=[case_id],
        )
        add_edge(edges, patient_node_id, case_node_id, "has_case_record", 1.0, "dataset_grounded", case_id)
        add_edge(edges, case_node_id, episode_node_id, "has_external_episode", 1.0, "dataset_grounded", case_id)

        for style_trait in row.get("type", []) or []:
            style_text = normalize_text(style_trait)
            if not style_text:
                continue
            style_id = statement_node_id("style_trait", style_text)
            add_node(
                nodes,
                style_id,
                type="style_trait",
                label=style_text,
                text=style_text,
                canonical_text=style_text,
                layer="dataset_grounded",
                source_dataset="Patient_Psi_CM_Dataset",
                source_case_ids=[case_id],
            )
            add_edge(edges, case_node_id, style_id, "has_style_trait", 1.0, "dataset_grounded", case_id)

        history_text = normalize_text(row.get("history", ""))
        if history_text:
            history_id = statement_node_id("history", history_text)
            add_node(
                nodes,
                history_id,
                type="history_statement",
                label=history_text[:80],
                text=history_text,
                canonical_text=history_text,
                layer="dataset_grounded",
                source_dataset="Patient_Psi_CM_Dataset",
                source_case_ids=[case_id],
            )
            add_edge(edges, case_node_id, history_id, "has_history", 1.0, "dataset_grounded", case_id)
            for subtopic_id in match_subtopics(history_text, subtopic_index):
                add_edge(edges, history_id, subtopic_id, "associated_with_topic", 0.6, "bridge", case_id)

        belief_node_ids: list[str] = []
        for field_name, belief_family in (
            ("helpless_belief", "helpless"),
            ("unlovable_belief", "unlovable"),
            ("worthless_belief", "worthless"),
        ):
            for belief_text in row.get(field_name, []) or []:
                belief_text = normalize_text(belief_text)
                if not belief_text:
                    continue
                belief_id = statement_node_id("core_belief", belief_text)
                belief_node_ids.append(belief_id)
                add_node(
                    nodes,
                    belief_id,
                    type="core_belief_statement",
                    label=belief_text,
                    text=belief_text,
                    canonical_text=belief_text,
                    layer="dataset_grounded",
                    source_dataset="Patient_Psi_CM_Dataset",
                    source_case_ids=[case_id],
                    belief_family=belief_family,
                )
                add_edge(edges, case_node_id, belief_id, "has_core_belief", 1.0, "dataset_grounded", case_id)
                cluster_id = BELIEF_STATEMENT_TO_CLUSTER.get(belief_text.lower())
                if cluster_id:
                    add_edge(edges, belief_id, cluster_id, "instance_of", 0.95, "bridge", case_id)

        intermediate_node_ids: list[str] = []
        for field_name, variant in (
            ("intermediate_belief", "default"),
            ("intermediate_belief_depression", "depression"),
        ):
            intermediate_text = normalize_text(row.get(field_name, ""))
            if not intermediate_text:
                continue
            intermediate_id = statement_node_id("intermediate_belief", f"{variant}:{intermediate_text}")
            intermediate_node_ids.append(intermediate_id)
            add_node(
                nodes,
                intermediate_id,
                type="intermediate_belief_statement",
                label=intermediate_text[:80],
                text=intermediate_text,
                canonical_text=intermediate_text,
                layer="dataset_grounded",
                source_dataset="Patient_Psi_CM_Dataset",
                source_case_ids=[case_id],
                variant=variant,
            )
            add_edge(
                edges,
                case_node_id,
                intermediate_id,
                f"has_intermediate_belief_{variant}",
                1.0,
                "dataset_grounded",
                case_id,
            )
            for intermediate_type in match_intermediate_types(intermediate_text):
                add_edge(edges, intermediate_id, intermediate_type, "instance_of", 0.75, "bridge", case_id)

        coping_text = normalize_text(row.get("coping_strategies", ""))
        coping_id = None
        if coping_text:
            coping_id = statement_node_id("coping", coping_text)
            add_node(
                nodes,
                coping_id,
                type="coping_strategy_statement",
                label=coping_text[:80],
                text=coping_text,
                canonical_text=coping_text,
                layer="dataset_grounded",
                source_dataset="Patient_Psi_CM_Dataset",
                source_case_ids=[case_id],
            )
            add_edge(edges, case_node_id, coping_id, "has_coping_strategy", 1.0, "dataset_grounded", case_id)
            for coping_type in match_coping_types(coping_text):
                add_edge(edges, coping_id, coping_type, "instance_of", 0.7, "bridge", case_id)

        situation_text = normalize_text(row.get("situation", ""))
        situation_id = None
        if situation_text:
            situation_id = statement_node_id("situation", situation_text)
            add_node(
                nodes,
                situation_id,
                type="situation_statement",
                label=situation_text[:80],
                text=situation_text,
                canonical_text=situation_text,
                layer="dataset_grounded",
                source_dataset="Patient_Psi_CM_Dataset",
                source_case_ids=[case_id],
            )
            add_edge(edges, episode_node_id, situation_id, "has_situation", 1.0, "dataset_grounded", case_id)
            for subtopic_id in match_subtopics(situation_text, subtopic_index):
                add_edge(edges, situation_id, subtopic_id, "associated_with_topic", 0.75, "bridge", case_id)

        thought_text = normalize_text(row.get("auto_thought", ""))
        thought_id = None
        if thought_text:
            thought_id = statement_node_id("automatic_thought", thought_text)
            add_node(
                nodes,
                thought_id,
                type="automatic_thought_statement",
                label=thought_text[:80],
                text=thought_text,
                canonical_text=thought_text,
                layer="dataset_grounded",
                source_dataset="Patient_Psi_CM_Dataset",
                source_case_ids=[case_id],
            )
            add_edge(edges, episode_node_id, thought_id, "has_automatic_thought", 1.0, "dataset_grounded", case_id)
            for subtopic_id in match_subtopics(thought_text, subtopic_index):
                add_edge(edges, thought_id, subtopic_id, "associated_with_topic", 0.65, "bridge", case_id)

        behavior_text = normalize_text(row.get("behavior", ""))
        behavior_id = None
        if behavior_text:
            behavior_id = statement_node_id("behavior", behavior_text)
            add_node(
                nodes,
                behavior_id,
                type="behavior_statement",
                label=behavior_text[:80],
                text=behavior_text,
                canonical_text=behavior_text,
                layer="dataset_grounded",
                source_dataset="Patient_Psi_CM_Dataset",
                source_case_ids=[case_id],
            )
            add_edge(edges, episode_node_id, behavior_id, "has_behavior", 1.0, "dataset_grounded", case_id)
            for behavior_pattern in match_behavior_patterns(behavior_text):
                add_edge(edges, behavior_id, behavior_pattern, "instance_of", 0.7, "bridge", case_id)

        emotion_node_ids: list[str] = []
        for emotion_text in row.get("emotion", []) or []:
            emotion_text = normalize_text(emotion_text)
            if not emotion_text:
                continue
            emotion_id = statement_node_id("emotion", emotion_text)
            emotion_node_ids.append(emotion_id)
            add_node(
                nodes,
                emotion_id,
                type="emotion_statement",
                label=emotion_text,
                text=emotion_text,
                canonical_text=emotion_text,
                layer="dataset_grounded",
                source_dataset="Patient_Psi_CM_Dataset",
                source_case_ids=[case_id],
            )
            add_edge(edges, episode_node_id, emotion_id, "has_emotion", 1.0, "dataset_grounded", case_id)
            for emotion_cluster in match_emotion_clusters(emotion_text):
                add_edge(edges, emotion_id, emotion_cluster, "instance_of", 0.8, "bridge", case_id)

        if history_text:
            history_id = statement_node_id("history", history_text)
            for belief_id in belief_node_ids:
                add_edge(edges, history_id, belief_id, "history_supports_belief", 0.8, "causal", case_id)

        for belief_id in belief_node_ids:
            for intermediate_id in intermediate_node_ids:
                add_edge(edges, belief_id, intermediate_id, "belief_expressed_as", 0.8, "causal", case_id)

        if situation_id and thought_id:
            add_edge(edges, situation_id, thought_id, "triggers_automatic_thought", 1.0, "causal", case_id)
        if thought_id:
            for emotion_id in emotion_node_ids:
                add_edge(edges, thought_id, emotion_id, "evokes_emotion", 0.9, "causal", case_id)
            if behavior_id:
                add_edge(edges, thought_id, behavior_id, "influences_behavior", 0.85, "causal", case_id)
        for intermediate_id in intermediate_node_ids:
            if thought_id:
                add_edge(edges, intermediate_id, thought_id, "intermediate_shapes_thought", 0.8, "causal", case_id)
        if coping_id and behavior_id:
            add_edge(edges, coping_id, behavior_id, "manifests_as_behavior", 0.8, "causal", case_id)

    spec_nodes = [nodes[node_id] for node_id in sorted(nodes)]
    spec_edges = [edges[key] for key in sorted(edges)]

    layer_counts: dict[str, int] = defaultdict(int)
    for node in spec_nodes:
        layer_counts[node["layer"]] += 1

    return {
        "metadata": {
            "name": "mind_voyager_hybrid_topic_graph",
            "description": (
                "Hybrid graph that keeps the generalized CBT ontology and adds "
                "Patient-psi-grounded statement nodes, case records, external episodes, "
                "and causal/bridge edges for dataset expansion and retrieval."
            ),
            "node_count": len(spec_nodes),
            "edge_count": len(spec_edges),
            "dataset_path": str(dataset_path),
            "base_ontology_name": base_spec["metadata"]["name"],
            "layer_counts": dict(sorted(layer_counts.items())),
        },
        "nodes": spec_nodes,
        "edges": spec_edges,
    }


def export_graph(spec: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    graph_path = output_dir / "hybrid_topic_graph.json"
    graph_path.write_text(json.dumps(spec, indent=2))
    written.append(graph_path)

    summary = {
        "metadata": spec["metadata"],
        "node_types": {},
        "node_layers": {},
        "edge_relations": {},
        "edge_layers": {},
    }

    for node in spec["nodes"]:
        summary["node_types"][node["type"]] = summary["node_types"].get(node["type"], 0) + 1
        summary["node_layers"][node["layer"]] = summary["node_layers"].get(node["layer"], 0) + 1
    for edge in spec["edges"]:
        summary["edge_relations"][edge["relation"]] = summary["edge_relations"].get(edge["relation"], 0) + 1
        summary["edge_layers"][edge["layer"]] = summary["edge_layers"].get(edge["layer"], 0) + 1

    summary_path = output_dir / "hybrid_topic_graph_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    written.append(summary_path)

    schema_path = output_dir / "hybrid_topic_graph_schema.json"
    schema_path.write_text(json.dumps(HYBRID_GRAPH_SCHEMA, indent=2))
    written.append(schema_path)

    try:
        import networkx as nx
    except ImportError:
        return written

    def graphml_safe(attrs: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in attrs.items():
            if isinstance(value, (list, dict)):
                safe[key] = json.dumps(value, ensure_ascii=False)
            else:
                safe[key] = value
        return safe

    graph = nx.DiGraph()
    for node in spec["nodes"]:
        graph.add_node(node["id"], **graphml_safe({k: v for k, v in node.items() if k != "id"}))
    for edge in spec["edges"]:
        graph.add_edge(
            edge["source"],
            edge["target"],
            **graphml_safe({k: v for k, v in edge.items() if k not in {"source", "target"}}),
        )

    graphml_path = output_dir / "hybrid_topic_graph.graphml"
    nx.write_graphml(graph, graphml_path)
    written.append(graphml_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the hybrid Patient-psi + ontology topic graph")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to Patient_Psi_CM_Dataset.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory where the hybrid graph files will be written",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    spec = build_hybrid_graph_spec(dataset_path)
    written = export_graph(spec, output_dir)
    print(
        json.dumps(
            {
                "written_files": [str(path) for path in written],
                "metadata": spec["metadata"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
