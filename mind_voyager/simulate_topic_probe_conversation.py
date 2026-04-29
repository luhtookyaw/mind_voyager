from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mind_voyager.client_simulator import (
    DEFAULT_DATASET,
    DIFFICULTIES,
    HISTORY_WINDOW_SIZE,
    SimulatorState,
    ensure_api_key,
    format_judge_status,
    load_case,
    load_prompt,
    maybe_update_state,
    render_client_system_prompt,
    render_client_user_prompt,
)
from mind_voyager.therapist_simulator import (
    generate_therapist_reply,
    latest_client_utterance,
    render_therapist_prompt,
)
from llm import call_llm
from scripts.retrieve_topic_graph import (
    DEFAULT_GRAPH_PATH,
    DEFAULT_INDEX_PATH,
    load_json,
    select_anchor_nodes,
)


ProbeMode = Literal["baseline", "retrieval", "random_probe", "hybrid_probe"]

NO_RETRIEVAL_CONTEXT = (
    "No client-specific sub-topic is available for this turn. "
    "Use general reflective exploration and invite the client to choose what matters."
)

PROBE_GUIDANCE = """

Sub-topic exploration guidance:
- You are not inferring a hidden formulation about the client.
- The sub-topics below are only conversation starters for gentle exploration.
- Do not treat any sub-topic as true about the client.
- Do not mention graph data, ontology, labels, or internal categories.
- Pick one natural direction from the sub-topics and ask about it in everyday language.
- Offer the possibility with uncertainty, and explicitly leave room for the client to correct you.
- If the client does not respond to the topic, back off and follow their lead.
""".strip()

HYBRID_GUIDANCE = """

Hybrid sub-topic guidance:
- On random-probe turns, use only the sampled sub-topics as conversation starters.
- On retrieval turns, use the latest client message only to choose a relevant sub-topic direction.
- Do not mention graph data, ontology, labels, or internal categories.
- Prefer rapport and client autonomy over topic coverage.
""".strip()


def retrieve_context_for_therapist(
    query: str | None,
    graph_path: Path,
    index_path: Path,
) -> dict[str, Any] | None:
    if not query or not query.strip():
        return None
    spec = load_json(graph_path)
    index = load_json(index_path)
    parent_by_subtopic = build_subtopic_parent_map(spec)
    nodes_by_id = {node["id"]: node for node in spec["nodes"]}
    anchors = select_anchor_nodes(
        query=query,
        index=index,
        anchor_types=("sub_topic",),
        top_k=3,
    )
    for anchor in anchors:
        parent = nodes_by_id.get(parent_by_subtopic.get(anchor["id"], ""), {})
        anchor["selection_method"] = "subtopic_similarity"
        anchor["super_topic_id"] = parent.get("id")
        anchor["super_topic_label"] = parent.get("label")
    return {
        "query": query,
        "anchors": anchors,
        "expanded": [],
        "prompt_context": build_subtopic_prompt_context(anchors),
    }


def generate_masked_client_reply(state: SimulatorState, client_model: str) -> str:
    return call_llm(
        system_prompt=render_client_system_prompt(state),
        user_prompt=render_client_user_prompt(state.dialogue, window_size=HISTORY_WINDOW_SIZE),
        temperature=0.3,
        model=client_model,
    ).strip()


def transcript_records_text(transcript_records: list[dict[str, str | int]]) -> str:
    lines = []
    for item in transcript_records:
        speaker = "Therapist" if item["speaker"] == "therapist" else "Client"
        lines.append(f"{speaker}: {item['content']}")
    return "\n".join(lines)


def should_end_conversation(
    transcript_records: list[dict[str, str | int]],
    model: str,
) -> tuple[bool, str]:
    prompt = load_prompt("moderator.txt").format(
        conversation=transcript_records_text(transcript_records)
    )
    response = call_llm(
        system_prompt="",
        user_prompt=prompt,
        temperature=0.0,
        model=model,
    ).strip()
    return response.upper().startswith("YES"), response


def build_subtopic_parent_map(spec: dict[str, Any]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for edge in spec["edges"]:
        if edge["relation"] == "contains":
            parents[edge["target"]] = edge["source"]
    return parents


def choose_random_subtopic_anchors(
    spec: dict[str, Any],
    rng: random.Random,
    count: int,
    used_anchor_ids: set[str],
) -> list[dict[str, Any]]:
    nodes_by_id = {node["id"]: node for node in spec["nodes"]}
    parent_by_subtopic = build_subtopic_parent_map(spec)
    subtopics = [node for node in spec["nodes"] if node["type"] == "sub_topic"]
    unused = [node for node in subtopics if node["id"] not in used_anchor_ids]
    pool = unused or subtopics

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in pool:
        by_parent[parent_by_subtopic.get(node["id"], "unknown")].append(node)

    parent_ids = list(by_parent)
    rng.shuffle(parent_ids)

    chosen: list[dict[str, Any]] = []
    for parent_id in parent_ids:
        if len(chosen) >= count:
            break
        node = rng.choice(by_parent[parent_id])
        chosen.append(node)

    if len(chosen) < count:
        remaining = [node for node in pool if node["id"] not in {item["id"] for item in chosen}]
        rng.shuffle(remaining)
        chosen.extend(remaining[: count - len(chosen)])

    anchors = []
    for node in chosen:
        parent = nodes_by_id.get(parent_by_subtopic.get(node["id"], ""), {})
        anchors.append(
            {
                "id": node["id"],
                "type": node["type"],
                "label": node["label"],
                "score": 1.0,
                "selection_method": "random_subtopic_probe",
                "super_topic_id": parent.get("id"),
                "super_topic_label": parent.get("label"),
            }
        )
    return anchors


def build_random_probe_context(
    spec: dict[str, Any],
    rng: random.Random,
    used_anchor_ids: set[str],
    anchor_count: int,
) -> dict[str, Any]:
    anchors = choose_random_subtopic_anchors(
        spec=spec,
        rng=rng,
        count=anchor_count,
        used_anchor_ids=used_anchor_ids,
    )
    used_anchor_ids.update(anchor["id"] for anchor in anchors)
    prompt_context = build_subtopic_prompt_context(anchors)
    return {
        "anchors": anchors,
        "expanded": [],
        "prompt_context": prompt_context,
    }


def should_random_probe_turn(
    turn: int,
    mode: ProbeMode,
    probe_turns: int,
    probe_interval: int,
) -> bool:
    if mode == "random_probe":
        return True
    if mode != "hybrid_probe":
        return False
    if turn > probe_turns:
        return False
    return turn == 1 or (probe_interval > 0 and (turn - 1) % probe_interval == 0)


def build_subtopic_prompt_context(anchors: list[dict[str, Any]]) -> str:
    lines = []
    for anchor in anchors:
        super_topic = anchor.get("super_topic_label") or "General"
        lines.append(f"- {anchor['label']} ({super_topic})")
    lines.extend(
        [
            "",
            (
                "Explicitly discuss this topic in normal, client-friendly language. "
                "Ask whether it fits the client, while leaving room for them to say it does not."
            ),
        ]
    )
    return "\n".join(lines)


def render_topic_probe_therapist_prompt(case: Any, sub_topic_context: str) -> str:
    prompt = load_prompt("topic_probe_therapist_prompt.txt").strip()
    return prompt.format(
        client_name=case.name,
        sub_topic_context=sub_topic_context,
    )


def build_probe_therapist_prompt(case: Any, context: str, mode: ProbeMode, source: str) -> str:
    guidance = HYBRID_GUIDANCE if mode == "hybrid_probe" else PROBE_GUIDANCE
    source_line = f"\n\nCurrent sub-topic source: {source}."
    base_prompt = render_therapist_prompt(case)
    return (
        f"{base_prompt}\n\n"
        f"{guidance}{source_line}\n\n"
        f"{context}"
    )


def run_simulation(
    case_id: str,
    dataset: Path,
    difficulty_name: str,
    therapist_model: str,
    therapist_provider: str,
    client_model: str,
    judge_model: str,
    moderator_model: str,
    max_turns: int,
    output: Path | None,
    prompt_mode: ProbeMode,
    graph_path: Path,
    index_path: Path,
    use_moderator: bool,
    probe_turns: int,
    probe_interval: int,
    probe_anchor_count: int,
    random_seed: int | None,
) -> None:
    ensure_api_key()
    case = load_case(dataset, case_id)
    difficulty = DIFFICULTIES[difficulty_name]
    baseline_therapist_prompt = render_therapist_prompt(case)
    graph_spec = load_json(graph_path) if prompt_mode in {"random_probe", "hybrid_probe"} else None
    rng = random.Random(random_seed)
    used_probe_anchor_ids: set[str] = set()

    print("Simulation setup")
    print("----------------")
    print(f"Case ID: {case.case_id}")
    print(f"Client: {case.name}")
    print("Client mode: masked")
    print(f"Difficulty: {difficulty.name}")
    print(f"Therapist provider: {therapist_provider}")
    print(f"Therapist model: {therapist_model}")
    print(f"Client model: {client_model}")
    print(f"Judge model: {judge_model}")
    print(f"Moderator: {'enabled' if use_moderator else 'disabled'}")
    if use_moderator:
        print(f"Moderator model: {moderator_model}")
    print(f"Therapist prompt mode: {prompt_mode}")
    if prompt_mode in {"random_probe", "hybrid_probe"}:
        print(f"Probe turns: {probe_turns}")
        print(f"Probe interval: {probe_interval}")
        print(f"Probe anchor count: {probe_anchor_count}")
        print(f"Random seed: {random_seed}")
    print()

    therapist_transcript: list[dict[str, str]] = []
    state = SimulatorState(case=case, difficulty=difficulty)

    transcript_records: list[dict[str, str | int]] = []
    moderator_events: list[str] = []
    retrieval_events: list[dict[str, Any]] = []
    topic_probe_events: list[dict[str, Any]] = []
    stop_reason = "max_turns"

    for turn in range(1, max_turns + 1):
        current_therapist_prompt = baseline_therapist_prompt
        graph_context_source = "baseline"

        if should_random_probe_turn(turn, prompt_mode, probe_turns, probe_interval):
            assert graph_spec is not None
            probe = build_random_probe_context(
                spec=graph_spec,
                rng=rng,
                used_anchor_ids=used_probe_anchor_ids,
                anchor_count=probe_anchor_count,
            )
            current_therapist_prompt = render_topic_probe_therapist_prompt(
                case=case,
                sub_topic_context=probe["prompt_context"],
            )
            graph_context_source = "random_subtopic_probe"
            event = {
                "turn": turn,
                "prompt_context": probe["prompt_context"],
                "anchors": probe["anchors"],
                "expanded": probe["expanded"],
            }
            topic_probe_events.append(event)
            anchor_labels = ", ".join(item["label"] for item in probe["anchors"])
            print(f"Topic probe> anchors={anchor_labels}")

        elif prompt_mode in {"retrieval", "hybrid_probe"}:
            query = latest_client_utterance(therapist_transcript)
            retrieval = retrieve_context_for_therapist(
                query=query,
                graph_path=graph_path,
                index_path=index_path,
            )
            retrieved_context = (
                retrieval["prompt_context"] if retrieval is not None else NO_RETRIEVAL_CONTEXT
            )
            current_therapist_prompt = build_probe_therapist_prompt(
                case=case,
                context=retrieved_context,
                mode=prompt_mode,
                source="retrieval" if retrieval is not None else "no_retrieval_context",
            )
            graph_context_source = "retrieval" if retrieval is not None else "no_retrieval_context"
            retrieval_event = {
                "turn": turn,
                "query": query,
                "prompt_context": retrieved_context,
            }
            if retrieval is not None:
                retrieval_event["anchors"] = retrieval["anchors"]
                retrieval_event["expanded"] = retrieval["expanded"]
            retrieval_events.append(retrieval_event)
            if retrieval is not None:
                anchor_labels = ", ".join(item["label"] for item in retrieval["anchors"])
                print(f"Retrieval> anchors={anchor_labels}")
            else:
                print("Retrieval> no client utterance yet; using baseline opening guidance")

        therapist_reply = generate_therapist_reply(
            therapist_prompt=current_therapist_prompt,
            transcript=therapist_transcript,
            model=therapist_model,
            provider=therapist_provider,
        )
        print(f"Therapist {turn}> {therapist_reply}\n")
        therapist_transcript.append({"role": "assistant", "content": therapist_reply})
        transcript_records.append(
            {"turn": turn, "speaker": "therapist", "content": therapist_reply}
        )

        state.dialogue.append({"role": "user", "content": therapist_reply})
        state.therapist_turns += 1
        judge_status = maybe_update_state(state, judge_model)
        print(format_judge_status(state.therapist_turns, judge_status))
        client_reply = generate_masked_client_reply(state=state, client_model=client_model)
        state.dialogue.append({"role": "assistant", "content": client_reply})

        print(f"Client {turn}> {client_reply}\n")
        therapist_transcript.append({"role": "user", "content": client_reply})
        transcript_records.append({"turn": turn, "speaker": "client", "content": client_reply})

        if topic_probe_events and topic_probe_events[-1]["turn"] == turn:
            topic_probe_events[-1]["client_reply"] = client_reply
            topic_probe_events[-1]["graph_context_source"] = graph_context_source

        if use_moderator:
            should_end, moderator_response = should_end_conversation(
                transcript_records=transcript_records,
                model=moderator_model,
            )
            moderator_event = f"Turn {turn}: {moderator_response.strip()}"
            moderator_events.append(moderator_event)
            print(f"Moderator> {moderator_response}\n")
            if should_end:
                stop_reason = "moderator_yes"
                break

    if state.events:
        print("Mediator events")
        print("---------------")
        for event in state.events:
            print(event)

    if output is not None:
        payload: dict[str, object] = {
            "case_id": case.case_id,
            "client_name": case.name,
            "client_mode": "masked",
            "difficulty": difficulty.name,
            "openness_interval": difficulty.openness_interval,
            "therapist_provider": therapist_provider,
            "therapist_model": therapist_model,
            "therapist_prompt_mode": prompt_mode,
            "client_model": client_model,
            "moderator_enabled": use_moderator,
            "moderator_model": moderator_model,
            "max_turns": max_turns,
            "therapist_turns": state.therapist_turns,
            "stop_reason": stop_reason,
            "transcript": transcript_records,
            "judge_model": judge_model,
            "mediator_events": state.events,
            "moderator_events": moderator_events,
            "retrieval_events": retrieval_events,
            "topic_probe_events": topic_probe_events,
            "topic_probe_config": {
                "probe_turns": probe_turns,
                "probe_interval": probe_interval,
                "probe_anchor_count": probe_anchor_count,
                "random_seed": random_seed,
            },
            "final_openness_score": state.current_openness_score,
            "final_visible_experience_count": state.visible_experience_count,
            "internal_revealed": state.internal_revealed,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved transcript to {output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a therapist/client conversation with random or hybrid topic-graph probing."
    )
    parser.add_argument("--case-id", required=True, help="Dataset case id, e.g. 1-1")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to Patient_Psi_CM_Dataset.json",
    )
    parser.add_argument(
        "--difficulty",
        choices=sorted(DIFFICULTIES),
        default="normal",
        help="Difficulty used for client openness/metacognition",
    )
    parser.add_argument("--therapist-model", default="gpt-4o-mini", help="Therapist model")
    parser.add_argument(
        "--therapist-provider",
        choices=["openai", "groq"],
        default="openai",
        help="Backend provider for therapist generation",
    )
    parser.add_argument("--client-model", default="gpt-4o-mini", help="Client model")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Mediator judge model")
    parser.add_argument("--moderator-model", default="gpt-4o-mini", help="Moderator model")
    parser.add_argument(
        "--no-moderator",
        action="store_true",
        help="Disable moderator-based early stopping",
    )
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum therapist turns")
    parser.add_argument("--output", help="Optional path to save the transcript as JSON")
    parser.add_argument(
        "--prompt-mode",
        choices=["baseline", "retrieval", "random_probe", "hybrid_probe"],
        default="hybrid_probe",
        help=(
            "Therapist graph guidance mode. random_probe samples graph sub-topics; "
            "hybrid_probe probes early turns then uses retrieval."
        ),
    )
    parser.add_argument(
        "--topic-graph",
        default=str(DEFAULT_GRAPH_PATH),
        help="Path to topic_graph.json",
    )
    parser.add_argument(
        "--topic-index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to node_embeddings.json used for retrieval/hybrid modes",
    )
    parser.add_argument(
        "--probe-turns",
        type=int,
        default=6,
        help="In hybrid mode, use random probes only through this therapist turn.",
    )
    parser.add_argument(
        "--probe-interval",
        type=int,
        default=2,
        help="In hybrid mode, random-probe every N turns during --probe-turns.",
    )
    parser.add_argument(
        "--probe-anchor-count",
        type=int,
        default=1,
        help="Number of random sub-topics to place in each probe context. Use 1 for explicit single-topic probing.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Optional seed for reproducible random topic probes.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_simulation(
        case_id=args.case_id,
        dataset=Path(args.dataset),
        difficulty_name=args.difficulty,
        therapist_model=args.therapist_model,
        therapist_provider=args.therapist_provider,
        client_model=args.client_model,
        judge_model=args.judge_model,
        moderator_model=args.moderator_model,
        max_turns=args.max_turns,
        output=Path(args.output) if args.output else None,
        prompt_mode=args.prompt_mode,
        graph_path=Path(args.topic_graph),
        index_path=Path(args.topic_index),
        use_moderator=not args.no_moderator,
        probe_turns=args.probe_turns,
        probe_interval=args.probe_interval,
        probe_anchor_count=max(1, args.probe_anchor_count),
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
