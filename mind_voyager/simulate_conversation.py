from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from groq_req import call_groq_messages
from llm import call_llm, call_llm_messages
from mind_voyager.client_simulator import (
    ClientCase,
    DEFAULT_DATASET,
    DIFFICULTIES,
    GOODBYE_RE,
    SimulatorState,
    ensure_api_key,
    load_case,
    load_prompt,
    format_judge_status,
    maybe_update_state,
    render_client_system_prompt,
)
from scripts.retrieve_topic_graph import (
    DEFAULT_GRAPH_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_RELATIONS,
    build_type_limits,
    retrieve_topic_graph_context,
)


NO_RETRIEVAL_CONTEXT = (
    "No retrieved graph context is available for this turn. "
    "Use general reflective exploration based only on the conversation."
)


def render_therapist_prompt(case: ClientCase) -> str:
    prompt = load_prompt("therapist_prompt.txt").strip()
    case_context = (
        "\n\nCase context:\n"
        f"- Client name: {case.name}"
    )
    return f"{prompt}{case_context}"


def render_retrieval_therapist_prompt(case: ClientCase, retrieved_context: str) -> str:
    prompt = load_prompt("retrieval_therapist_prompt.txt").strip()
    prompt = prompt.replace("{{RETRIEVED_CCD_CONTEXT}}", retrieved_context)
    case_context = (
        "\n\nCase context:\n"
        f"- Client name: {case.name}"
    )
    return f"{prompt}{case_context}"


def generate_therapist_reply(
    therapist_prompt: str,
    transcript: list[dict[str, str]],
    model: str,
    provider: str,
) -> str:
    messages = [{"role": "system", "content": therapist_prompt}, *transcript]
    if provider == "groq":
        return call_groq_messages(messages=messages, temperature=0.3, model=model).strip()
    return call_llm_messages(messages=messages, temperature=0.3, model=model).strip()


def latest_client_utterance(transcript: list[dict[str, str]]) -> str | None:
    for item in reversed(transcript):
        if item["role"] == "user":
            return item["content"]
    return None


def retrieve_context_for_therapist(
    query: str | None,
    graph_path: Path,
    index_path: Path,
) -> dict[str, Any] | None:
    if not query or not query.strip():
        return None
    return retrieve_topic_graph_context(
        query=query,
        graph_path=graph_path,
        index_path=index_path,
        anchor_types=("sub_topic",),
        anchor_top_k=3,
        relation_filter=DEFAULT_RELATIONS,
        per_anchor_limit=6,
        type_limits=build_type_limits(
            emotion_limit=2,
            coping_limit=2,
            behavior_limit=2,
            intermediate_limit=2,
            prompt_limit=2,
            core_belief_limit=1,
        ),
    )


def generate_masked_client_reply(
    state: SimulatorState,
    client_model: str,
) -> str:
    return call_llm_messages(
        messages=[{"role": "system", "content": render_client_system_prompt(state)}, *state.dialogue],
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
    use_retrieval: bool,
    graph_path: Path,
    index_path: Path,
    use_moderator: bool,
) -> None:
    ensure_api_key()
    case = load_case(dataset, case_id)
    difficulty = DIFFICULTIES[difficulty_name]
    therapist_prompt = render_therapist_prompt(case)

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
    print(f"Graph retrieval: {'enabled' if use_retrieval else 'disabled'}")
    print()

    therapist_transcript: list[dict[str, str]] = []
    state = SimulatorState(case=case, difficulty=difficulty)

    transcript_records: list[dict[str, str | int]] = []
    moderator_events: list[str] = []
    retrieval_events: list[dict[str, Any]] = []
    stop_reason = "max_turns"

    for turn in range(1, max_turns + 1):
        current_therapist_prompt = therapist_prompt
        if use_retrieval:
            query = latest_client_utterance(therapist_transcript)
            retrieval = retrieve_context_for_therapist(
                query=query,
                graph_path=graph_path,
                index_path=index_path,
            )
            retrieved_context = (
                retrieval["prompt_context"] if retrieval is not None else NO_RETRIEVAL_CONTEXT
            )
            current_therapist_prompt = render_retrieval_therapist_prompt(
                case=case,
                retrieved_context=retrieved_context,
            )
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

        if GOODBYE_RE.search(therapist_reply):
            stop_reason = "therapist_goodbye"
            break

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
            "therapist_prompt_mode": "retrieval" if use_retrieval else "baseline",
            "client_model": client_model,
            "moderator_enabled": use_moderator,
            "moderator_model": moderator_model,
            "max_turns": max_turns,
            "therapist_turns": state.therapist_turns,
            "stop_reason": stop_reason,
            "transcript": transcript_records,
        }
        payload["judge_model"] = judge_model
        payload["mediator_events"] = state.events
        payload["moderator_events"] = moderator_events
        payload["retrieval_events"] = retrieval_events
        payload["final_openness_score"] = state.current_openness_score
        payload["final_visible_experience_count"] = state.visible_experience_count
        payload["internal_revealed"] = state.internal_revealed
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved transcript to {output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate therapist/client self-play conversation")
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
        "--use-retrieval",
        action="store_true",
        help="Use topic-graph retrieval context in the therapist system prompt",
    )
    parser.add_argument(
        "--topic-graph",
        default=str(DEFAULT_GRAPH_PATH),
        help="Path to topic_graph.json used with --use-retrieval",
    )
    parser.add_argument(
        "--topic-index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to node_embeddings.json used with --use-retrieval",
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
        use_retrieval=args.use_retrieval,
        graph_path=Path(args.topic_graph),
        index_path=Path(args.topic_index),
        use_moderator=not args.no_moderator,
    )


if __name__ == "__main__":
    main()
