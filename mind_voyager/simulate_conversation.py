from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm import call_llm
from mind_voyager.client_simulator import (
    DEFAULT_DATASET,
    DIFFICULTIES,
    HISTORY_WINDOW_SIZE,
    SimulatorState,
    ensure_api_key,
    render_client_user_prompt,
    load_case,
    load_prompt,
    format_judge_status,
    maybe_update_state,
    render_client_system_prompt,
)
from mind_voyager.therapist_simulator import (
    generate_therapist_reply,
    render_therapist_prompt,
)


def generate_masked_client_reply(
    state: SimulatorState,
    client_model: str,
) -> str:
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


def run_simulation(
    case_id: str,
    dataset: Path,
    difficulty_name: str,
    therapist_model: str,
    therapist_provider: str,
    client_model: str,
    judge_model: str,
    moderator_model: str,
    embedding_model: str,
    max_turns: int,
    output: Path | None,
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
    print("Reveal engine: deterministic field-level unlock")
    print(f"Moderator: {'enabled' if use_moderator else 'disabled'}")
    if use_moderator:
        print(f"Moderator model: {moderator_model}")
    print()

    therapist_transcript: list[dict[str, str]] = []
    state = SimulatorState(case=case, difficulty=difficulty, embedding_model=embedding_model)

    transcript_records: list[dict[str, str | int]] = []
    moderator_events: list[str] = []
    stop_reason = "max_turns"

    for turn in range(1, max_turns + 1):
        therapist_reply = generate_therapist_reply(
            therapist_prompt=therapist_prompt,
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
        reveal_status = maybe_update_state(state, judge_model)
        print(format_judge_status(state.therapist_turns, reveal_status))
        client_reply = generate_masked_client_reply(state=state, client_model=client_model)
        state.dialogue.append({"role": "assistant", "content": client_reply})

        print(f"Client {turn}> {client_reply}\n")
        therapist_transcript.append({"role": "user", "content": client_reply})
        transcript_records.append({"turn": turn, "speaker": "client", "content": client_reply})

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
            "embedding_model": state.embedding_model,
            "therapist_provider": therapist_provider,
            "therapist_model": therapist_model,
            "therapist_prompt_mode": "baseline",
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
        payload["revealed_fields"] = state.revealed_fields
        payload["best_similarity_by_field"] = state.best_similarity_by_field
        payload["reveal_events"] = state.reveal_events
        payload["turn_similarity_events"] = state.turn_similarity_events
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
        help="Difficulty used for per-field reveal thresholds",
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
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model used for therapist-to-CCD similarity scoring",
    )
    parser.add_argument(
        "--no-moderator",
        action="store_true",
        help="Disable moderator-based early stopping",
    )
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum therapist turns")
    parser.add_argument("--output", help="Optional path to save the transcript as JSON")
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
        embedding_model=args.embedding_model,
        max_turns=args.max_turns,
        output=Path(args.output) if args.output else None,
        use_moderator=not args.no_moderator,
    )


if __name__ == "__main__":
    main()
