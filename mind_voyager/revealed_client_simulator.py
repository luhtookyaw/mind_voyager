from __future__ import annotations

import argparse
from pathlib import Path

from llm import call_llm_messages
from mind_voyager.client_simulator import (
    DEFAULT_DATASET,
    GOODBYE_RE,
    ClientCase,
    DIFFICULTIES,
    DifficultyConfig,
    ensure_api_key,
    load_case,
    load_prompt,
)


def build_revealed_payload(
    case: ClientCase,
    difficulty: DifficultyConfig,
    hide_all: bool = False,
) -> dict[str, str]:
    if hide_all:
        return {
            "name": case.name,
            "openness": "unknown",
            "metacognition": "unknown",
            "history": "unknown",
            "core_belief": "unknown",
            "intermediate_belief": "unknown",
            "coping_strategy": "unknown",
            "situation": "unknown",
            "reaction": "unknown",
        }

    core_beliefs = " | ".join(case.core_beliefs) if case.core_beliefs else "unknown"
    intermediate = case.intermediate_belief or "unknown"
    if case.intermediate_belief_depression:
        intermediate = f"{intermediate}\nDepressive variant: {case.intermediate_belief_depression}"
    reaction = "\n".join(
        [
            f"Automatic thought: {case.automatic_thought or 'unknown'}",
            f"Emotion: {'; '.join(case.emotion) if case.emotion else 'unknown'}",
            f"Behavior: {case.behavior or 'unknown'}",
        ]
    )
    return {
        "name": case.name,
        "openness": difficulty.openness,
        "metacognition": difficulty.metacognition,
        "history": case.history or "unknown",
        "core_belief": core_beliefs,
        "intermediate_belief": intermediate,
        "coping_strategy": case.coping_strategies or "unknown",
        "situation": case.situation or "unknown",
        "reaction": reaction,
    }


def render_revealed_client_prompt(
    case: ClientCase,
    difficulty: DifficultyConfig,
    hide_all: bool = False,
) -> str:
    prompt = load_prompt("reveal_client_prompt.txt").replace("[Client]", case.name)
    prompt = prompt.format(**build_revealed_payload(case, difficulty, hide_all=hide_all))
    print(prompt)
    return prompt


def therapist_intake(case: ClientCase, difficulty: DifficultyConfig, hide_all: bool = False) -> str:
    reveal_mode = "all client-side fields hidden as unknown" if hide_all else "full cognitive diagram visible from the start"
    return (
        f"Case ID: {case.case_id}\n"
        f"Client: {case.name}\n"
        f"Presenting situation: {case.situation}\n"
        f"Difficulty label: {difficulty.name}\n"
        f"Reveal mode: {reveal_mode}"
    )


def run_dry_run(case: ClientCase, difficulty: DifficultyConfig, hide_all: bool = False) -> None:
    print("Therapist intake")
    print("----------------")
    print(therapist_intake(case, difficulty, hide_all=hide_all))
    print("\nRevealed client system prompt")
    print("-----------------------------")
    print(render_revealed_client_prompt(case, difficulty, hide_all=hide_all))


def run_interactive_session(
    case: ClientCase,
    difficulty: DifficultyConfig,
    model: str,
    max_turns: int,
    hide_all: bool = False,
) -> None:
    ensure_api_key()
    dialogue: list[dict[str, str]] = []
    system_prompt = render_revealed_client_prompt(case, difficulty, hide_all=hide_all)

    print("Therapist intake")
    print("----------------")
    print(therapist_intake(case, difficulty, hide_all=hide_all))
    print("\nType therapist messages. Type 'exit' to stop.\n")

    for _ in range(max_turns):
        therapist = input("Therapist> ").strip()
        if not therapist:
            continue
        if therapist.lower() in {"exit", "quit"}:
            break

        dialogue.append({"role": "user", "content": therapist})
        reply = call_llm_messages(
            messages=[{"role": "system", "content": system_prompt}, *dialogue],
            temperature=0.3,
            model=model,
        ).strip()
        dialogue.append({"role": "assistant", "content": reply})
        print(f"Client> {reply}\n")

        if GOODBYE_RE.search(therapist):
            break


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MindVoyager revealed client simulator")
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
        help="Difficulty label used only to set openness/metacognition text in the prompt",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Client model")
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum therapist turns")
    parser.add_argument("--dry-run", action="store_true", help="Print intake and prompt only")
    parser.add_argument(
        "--hide-all",
        action="store_true",
        help="Force all client-side prompt fields except the client name to 'unknown'",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    case = load_case(Path(args.dataset), args.case_id)
    difficulty = DIFFICULTIES[args.difficulty]
    if args.dry_run:
        run_dry_run(case, difficulty, hide_all=args.hide_all)
        return
    run_interactive_session(
        case=case,
        difficulty=difficulty,
        model=args.model,
        max_turns=args.max_turns,
        hide_all=args.hide_all,
    )


if __name__ == "__main__":
    main()
