from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm import call_llm, call_llm_messages


ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "mind_voyager" / "prompts"
DEFAULT_DATASET = ROOT / "data" / "Patient_Psi_CM_Dataset.json"
GOODBYE_RE = re.compile(r"\bgoodbye\b", re.IGNORECASE)


@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    openness: str
    metacognition: str
    initial_visible_experiences: int
    exploration_interval: int


DIFFICULTIES = {
    "easy": DifficultyConfig("easy", "high", "high", 3, 1),
    "normal": DifficultyConfig("normal", "medium", "medium", 2, 2),
    "hard": DifficultyConfig("hard", "low", "low", 1, 3),
}


@dataclass
class ClientCase:
    case_id: str
    name: str
    type_traits: list[str]
    history: str
    core_beliefs: list[str]
    intermediate_belief: str
    intermediate_belief_depression: str
    coping_strategies: str
    situation: str
    automatic_thought: str
    emotion: list[str]
    behavior: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ClientCase":
        core_beliefs = []
        for key in ("helpless_belief", "unlovable_belief", "worthless_belief"):
            core_beliefs.extend(item for item in row.get(key, []) if item)
        return cls(
            case_id=row["id"],
            name=row["name"],
            type_traits=row.get("type", []),
            history=row.get("history", "").strip(),
            core_beliefs=core_beliefs,
            intermediate_belief=row.get("intermediate_belief", "").strip(),
            intermediate_belief_depression=row.get("intermediate_belief_depression", "").strip(),
            coping_strategies=row.get("coping_strategies", "").strip(),
            situation=row.get("situation", "").strip(),
            automatic_thought=row.get("auto_thought", "").strip(),
            emotion=[item.strip() for item in row.get("emotion", []) if item.strip()],
            behavior=row.get("behavior", "").strip(),
        )


@dataclass
class SimulatorState:
    case: ClientCase
    difficulty: DifficultyConfig
    visible_experience_count: int = field(init=False)
    internal_revealed: bool = False
    therapist_turns: int = 0
    dialogue: list[dict[str, str]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.visible_experience_count = self.difficulty.initial_visible_experiences

    @property
    def rapport_level(self) -> str:
        if self.visible_experience_count <= 1:
            return "low"
        if self.visible_experience_count == 2:
            return "building"
        return "strong"

    def experience_block(self) -> tuple[str, str]:
        reaction_lines = [
            f"Automatic thought: {self.case.automatic_thought or 'unknown'}",
            f"Emotion: {'; '.join(self.case.emotion) if self.case.emotion else 'unknown'}",
            f"Behavior: {self.case.behavior or 'unknown'}",
        ]
        return self.case.situation or "unknown", "\n".join(reaction_lines)

    def visible_experiences(self) -> list[tuple[str, str]]:
        situation, reaction = self.experience_block()
        return [(situation, reaction) for _ in range(self.visible_experience_count)]

    def masked_experience_slots(self) -> list[int]:
        return list(range(self.visible_experience_count + 1, 4))

    def visible_internal(self) -> dict[str, str]:
        if not self.internal_revealed:
            return {}
        core = " | ".join(self.case.core_beliefs) if self.case.core_beliefs else "unknown"
        intermediate = self.case.intermediate_belief
        if self.case.intermediate_belief_depression:
            intermediate = (
                f"{self.case.intermediate_belief}\nDepressive variant: "
                f"{self.case.intermediate_belief_depression}"
            )
        return {
            "history": self.case.history or "unknown",
            "core_beliefs": core,
            "intermediate_beliefs": intermediate or "unknown",
            "coping_strategies": self.case.coping_strategies or "unknown",
        }

    def therapist_intake(self) -> str:
        traits = ", ".join(self.case.type_traits) if self.case.type_traits else "not specified"
        return (
            f"Case ID: {self.case.case_id}\n"
            f"Client: {self.case.name}\n"
            f"Presenting situation: {self.case.situation}\n"
            f"Observed style tendencies: {traits}\n"
            f"Difficulty: {self.difficulty.name}\n"
            f"Initial visible experience blocks: {self.visible_experience_count}/3\n"
            f"Metacognition evaluation interval: every {self.difficulty.exploration_interval} therapist turn(s)"
        )


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def load_case(dataset_path: Path, case_id: str) -> ClientCase:
    rows = json.loads(dataset_path.read_text())
    for row in rows:
        if row.get("id") == case_id:
            return ClientCase.from_row(row)
    raise ValueError(f"Case id not found: {case_id}")


def build_base_prompt_payload(state: SimulatorState) -> dict[str, str]:
    visible_internal = state.visible_internal()
    situation, reaction = state.experience_block()
    visible_count = state.visible_experience_count

    return {
        "name": state.case.name,
        "openness": state.difficulty.openness,
        "metacognition": state.difficulty.metacognition,
        "history": visible_internal.get("history", "unknown"),
        "core_belief": visible_internal.get("core_beliefs", "unknown"),
        "intermediate_belief": visible_internal.get("intermediate_beliefs", "unknown"),
        "coping_strategy": visible_internal.get("coping_strategies", "unknown"),
        "situation1": situation if visible_count >= 1 else "unknown",
        "reaction1": reaction if visible_count >= 1 else "unknown",
        "situation2": situation if visible_count >= 2 else "unknown",
        "reaction2": reaction if visible_count >= 2 else "unknown",
        "situation3": situation if visible_count >= 3 else "unknown",
        "reaction3": reaction if visible_count >= 3 else "unknown",
    }


def render_client_system_prompt(state: SimulatorState) -> str:
    prompt = load_prompt("base_client_prompt.txt").replace("[Client]", state.case.name)
    prompt = prompt.format(**build_base_prompt_payload(state))
    return prompt


def transcript_text(dialogue: list[dict[str, str]]) -> str:
    lines = []
    for turn in dialogue:
        role = "Therapist" if turn["role"] == "user" else "Client"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def parse_first_rating(text: str) -> int | None:
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))
    return None


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for non-dry-run simulator usage.")


def judge_openness(dialogue: list[dict[str, str]], model: str) -> tuple[int | None, str]:
    prompt = load_prompt("openness_judge.txt").format(dialogue_context=transcript_text(dialogue))
    response = call_llm(
        system_prompt="",
        user_prompt=prompt,
        temperature=0.3,
        model=model,
    )
    return parse_first_rating(response), response


def judge_exploration(dialogue: list[dict[str, str]], model: str) -> tuple[int | None, str]:
    prompt = load_prompt("exploration_judge.txt").format(dialogue_history=transcript_text(dialogue))
    response = call_llm(
        system_prompt="",
        user_prompt=prompt,
        temperature=0.3,
        model=model,
    )
    return parse_first_rating(response), response


def format_judge_status(turn: int, judge_status: dict[str, str]) -> str:
    return (
        f"Judge scores after turn {turn}: "
        f"openness={judge_status['openness']}; "
        f"exploration={judge_status['exploration']}"
    )


def maybe_update_state(state: SimulatorState, judge_model: str) -> dict[str, str]:
    if not state.dialogue:
        return {
            "openness": "not run (no dialogue yet)",
            "exploration": "not run (no dialogue yet)",
        }

    did_rapport_check = state.therapist_turns % 4 == 0
    did_exploration_check = state.therapist_turns % state.difficulty.exploration_interval == 0
    judge_status = {
        "openness": "not scheduled",
        "exploration": "not scheduled",
    }

    if state.visible_experience_count >= 3:
        judge_status["openness"] = "not run (already fully externally revealed)"
    elif did_rapport_check:
        rating, raw = judge_openness(state.dialogue, judge_model)
        if rating is not None and rating >= 4:
            state.visible_experience_count += 1
            state.events.append(
                f"Rapport judge rated {rating}/5 and revealed one more experience block."
            )
            judge_status["openness"] = f"{rating}/5 (revealed one more experience block)"
        else:
            state.events.append(
                f"Rapport judge rated {rating or 'unknown'}/5; no experience reveal."
            )
            judge_status["openness"] = f"{rating or 'unknown'}/5 (no external reveal)"
        state.events.append(f"Rapport judge output: {raw.strip()}")

    if state.internal_revealed:
        judge_status["exploration"] = "not run (internal elements already revealed)"
    elif did_exploration_check:
        rating, raw = judge_exploration(state.dialogue, judge_model)
        if rating is not None and rating >= 4:
            state.internal_revealed = True
            state.events.append(
                f"Exploration judge rated {rating}/5 and revealed internal cognitive elements."
            )
            judge_status["exploration"] = f"{rating}/5 (revealed internal elements)"
        else:
            state.events.append(
                f"Exploration judge rated {rating or 'unknown'}/5; internal elements remain masked."
            )
            judge_status["exploration"] = f"{rating or 'unknown'}/5 (internal still masked)"
        state.events.append(f"Exploration judge output: {raw.strip()}")

    return judge_status


def generate_client_reply(state: SimulatorState, model: str) -> str:
    messages = [{"role": "system", "content": render_client_system_prompt(state)}, *state.dialogue]
    return call_llm_messages(messages=messages, temperature=0.3, model=model).strip()


def run_interactive_session(
    state: SimulatorState,
    model: str,
    judge_model: str,
    max_turns: int,
) -> None:
    ensure_api_key()
    print("Therapist intake")
    print("----------------")
    print(state.therapist_intake())
    print("\nType therapist messages. Type 'exit' to stop.\n")

    for _ in range(max_turns):
        therapist = input("Therapist> ").strip()
        if not therapist:
            continue
        if therapist.lower() in {"exit", "quit"}:
            break

        state.dialogue.append({"role": "user", "content": therapist})
        state.therapist_turns += 1
        judge_status = maybe_update_state(state, judge_model)
        print(format_judge_status(state.therapist_turns, judge_status))

        reply = generate_client_reply(state, model)
        state.dialogue.append({"role": "assistant", "content": reply})
        print(f"Client> {reply}\n")

        if GOODBYE_RE.search(therapist):
            break

    if state.events:
        print("Mediator events")
        print("---------------")
        for event in state.events:
            print(event)


def run_dry_run(state: SimulatorState) -> None:
    print("Therapist intake")
    print("----------------")
    print(state.therapist_intake())
    print("\nMasked client system prompt")
    print("---------------------------")
    print(render_client_system_prompt(state))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MindVoyager client simulator")
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
        help="Simulator difficulty mapped to openness/metacognition",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Client model")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Model used by the openness and exploration critics",
    )
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum therapist turns")
    parser.add_argument("--dry-run", action="store_true", help="Print intake and masked prompt only")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_path = Path(args.dataset)
    case = load_case(dataset_path, args.case_id)
    state = SimulatorState(case=case, difficulty=DIFFICULTIES[args.difficulty])
    if args.dry_run:
        run_dry_run(state)
        return
    run_interactive_session(
        state=state,
        model=args.model,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
