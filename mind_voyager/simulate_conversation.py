from __future__ import annotations

import argparse
import json
from pathlib import Path

from groq_req import call_groq_messages
from llm import call_llm_messages
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
from mind_voyager.revealed_client_simulator import render_revealed_client_prompt


def render_therapist_prompt(case: ClientCase) -> str:
    prompt = load_prompt("therapist_prompt.txt").strip()
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


def generate_masked_client_reply(
    state: SimulatorState,
    client_model: str,
) -> str:
    return call_llm_messages(
        messages=[{"role": "system", "content": render_client_system_prompt(state)}, *state.dialogue],
        temperature=0.3,
        model=client_model,
    ).strip()


def generate_revealed_client_reply(
    system_prompt: str,
    dialogue: list[dict[str, str]],
    client_model: str,
) -> str:
    return call_llm_messages(
        messages=[{"role": "system", "content": system_prompt}, *dialogue],
        temperature=0.3,
        model=client_model,
    ).strip()


def run_simulation(
    case_id: str,
    dataset: Path,
    difficulty_name: str,
    therapist_model: str,
    therapist_provider: str,
    client_model: str,
    judge_model: str,
    max_turns: int,
    revealed_client: bool,
    hide_all: bool,
    output: Path | None,
) -> None:
    ensure_api_key()
    case = load_case(dataset, case_id)
    difficulty = DIFFICULTIES[difficulty_name]
    therapist_prompt = render_therapist_prompt(case)

    print("Simulation setup")
    print("----------------")
    print(f"Case ID: {case.case_id}")
    print(f"Client: {case.name}")
    print(f"Client mode: {'revealed' if revealed_client else 'masked'}")
    if revealed_client:
        print(f"Hide all client fields: {'yes' if hide_all else 'no'}")
    print(f"Difficulty: {difficulty.name}")
    print(f"Therapist provider: {therapist_provider}")
    print(f"Therapist model: {therapist_model}")
    print(f"Client model: {client_model}")
    if not revealed_client:
        print(f"Judge model: {judge_model}")
    print()

    therapist_transcript: list[dict[str, str]] = []

    if revealed_client:
        client_system_prompt = render_revealed_client_prompt(case, difficulty, hide_all=hide_all)
        client_dialogue: list[dict[str, str]] = []
    else:
        state = SimulatorState(case=case, difficulty=difficulty)

    transcript_records: list[dict[str, str | int]] = []

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

        if revealed_client:
            client_dialogue.append({"role": "user", "content": therapist_reply})
            client_reply = generate_revealed_client_reply(
                system_prompt=client_system_prompt,
                dialogue=client_dialogue,
                client_model=client_model,
            )
            client_dialogue.append({"role": "assistant", "content": client_reply})
        else:
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
            break

    if not revealed_client and state.events:
        print("Mediator events")
        print("---------------")
        for event in state.events:
            print(event)

    if output is not None:
        payload: dict[str, object] = {
            "case_id": case.case_id,
            "client_name": case.name,
            "client_mode": "revealed" if revealed_client else "masked",
            "hide_all": hide_all if revealed_client else False,
            "difficulty": difficulty.name,
            "therapist_provider": therapist_provider,
            "therapist_model": therapist_model,
            "client_model": client_model,
            "max_turns": max_turns,
            "transcript": transcript_records,
        }
        if revealed_client:
            payload["final_visible_experience_count"] = 3
            payload["internal_revealed"] = True
        else:
            payload["judge_model"] = judge_model
            payload["mediator_events"] = state.events
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
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum therapist turns")
    parser.add_argument(
        "--revealed-client",
        action="store_true",
        help="Use reveal_client_prompt.txt and bypass the mediator",
    )
    parser.add_argument(
        "--hide-all",
        action="store_true",
        help="With --revealed-client, force all client-side prompt fields except the client name to 'unknown'",
    )
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
        max_turns=args.max_turns,
        revealed_client=args.revealed_client,
        hide_all=args.hide_all,
        output=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
