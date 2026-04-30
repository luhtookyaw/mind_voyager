from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llm import call_llm, cosine_similarity, get_embedding


ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "mind_voyager" / "prompts"
DEFAULT_DATASET = ROOT / "data" / "Patient_Psi_CM_Dataset.json"
DEFAULT_TRANSCRIPT_DIR = ROOT / "transcripts"
HISTORY_WINDOW_SIZE = 12
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_ONE_SHOT_MARGIN = 0.08

ALL_FIELDS = (
    "situation",
    "automatic_thought",
    "emotion",
    "behavior",
    "relevant_history",
    "core_beliefs",
    "intermediate_beliefs",
    "coping_strategies",
)

EXTERNAL_FIELDS = ("situation", "automatic_thought", "emotion", "behavior")
INTERNAL_FIELDS = (
    "relevant_history",
    "core_beliefs",
    "intermediate_beliefs",
    "coping_strategies",
)

FIELD_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "situation": (),
    "automatic_thought": ("situation",),
    "emotion": (),
    "behavior": ("situation", "emotion"),
    "relevant_history": (),
    "core_beliefs": ("intermediate_beliefs",),
    "intermediate_beliefs": ("automatic_thought",),
    "coping_strategies": ("emotion", "behavior"),
}

FIELD_THRESHOLDS: dict[str, dict[str, float]] = {
    "easy": {
        "situation": 0.30,
        "automatic_thought": 0.36,
        "emotion": 0.28,
        "behavior": 0.30,
        "relevant_history": 0.40,
        "core_beliefs": 0.44,
        "intermediate_beliefs": 0.42,
        "coping_strategies": 0.36,
    },
    "normal": {
        "situation": 0.38,
        "automatic_thought": 0.44,
        "emotion": 0.36,
        "behavior": 0.38,
        "relevant_history": 0.50,
        "core_beliefs": 0.54,
        "intermediate_beliefs": 0.52,
        "coping_strategies": 0.46,
    },
    "hard": {
        "situation": 0.48,
        "automatic_thought": 0.54,
        "emotion": 0.46,
        "behavior": 0.48,
        "relevant_history": 0.60,
        "core_beliefs": 0.64,
        "intermediate_beliefs": 0.62,
        "coping_strategies": 0.56,
    },
}

@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    field_thresholds: dict[str, float]


DIFFICULTIES = {
    name: DifficultyConfig(name=name, field_thresholds=FIELD_THRESHOLDS[name])
    for name in ("easy", "normal", "hard")
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


def build_broad_presenting_concern(case: ClientCase) -> str:
    text = " ".join(
        part
        for part in (
            case.situation.strip(),
            case.behavior.strip(),
            case.history.strip(),
        )
        if part
    ).lower()

    if any(word in text for word in ("family", "mother", "father", "cousin", "wedding")):
        return "feeling tense and avoidant around family interactions"
    if any(word in text for word in ("friend", "relationship", "rejected", "lonely", "alone")):
        return "having difficulty with relationships and fear of rejection"
    if any(word in text for word in ("work", "job", "project", "school", "study", "performance")):
        return "feeling pressured and discouraged around work or performance"
    if any(word in text for word in ("body", "weight", "appearance")):
        return "feeling distressed about body image and social judgment"
    if any(word in text for word in ("stress", "overwhelmed", "anxious", "avoid")):
        return "feeling overwhelmed and stuck in a recurring personal struggle"
    return "dealing with a recurring personal struggle that feels hard to talk about"


def build_case_field_texts(case: ClientCase) -> dict[str, str]:
    intermediate = case.intermediate_belief or "unknown"
    if case.intermediate_belief_depression:
        intermediate = (
            f"{intermediate}\nDepressive variant: {case.intermediate_belief_depression}"
        )

    return {
        "situation": case.situation or "unknown",
        "automatic_thought": case.automatic_thought or "unknown",
        "emotion": "; ".join(case.emotion) if case.emotion else "unknown",
        "behavior": case.behavior or "unknown",
        "relevant_history": case.history or "unknown",
        "core_beliefs": " | ".join(case.core_beliefs) if case.core_beliefs else "unknown",
        "intermediate_beliefs": intermediate or "unknown",
        "coping_strategies": case.coping_strategies or "unknown",
    }


@dataclass
class SimulatorState:
    case: ClientCase
    difficulty: DifficultyConfig
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    use_prerequisites: bool = False
    enable_one_shot: bool = False
    enable_two_hit: bool = False
    one_shot_margin: float = DEFAULT_ONE_SHOT_MARGIN
    therapist_turns: int = 0
    dialogue: list[dict[str, str]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    field_texts: dict[str, str] = field(init=False)
    field_embeddings: dict[str, list[float]] = field(init=False)
    revealed_fields: dict[str, bool] = field(init=False)
    best_similarity_by_field: dict[str, float] = field(init=False)
    reveal_events: list[dict[str, Any]] = field(default_factory=list)
    turn_similarity_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.field_texts = build_case_field_texts(self.case)
        self.field_embeddings = {
            field_name: get_embedding(text, model=self.embedding_model)
            for field_name, text in self.field_texts.items()
        }
        self.revealed_fields = {field_name: False for field_name in ALL_FIELDS}
        self.best_similarity_by_field = {field_name: 0.0 for field_name in ALL_FIELDS}

    @property
    def visible_experience_count(self) -> int:
        return int(self.revealed_fields["situation"])

    @property
    def current_openness_score(self) -> int | None:
        return None

    @property
    def internal_revealed(self) -> bool:
        return any(self.revealed_fields[field_name] for field_name in INTERNAL_FIELDS)

    @property
    def rapport_level(self) -> str:
        return "field-level-reveal"

    def therapist_intake(self) -> str:
        traits = ", ".join(self.case.type_traits) if self.case.type_traits else "not specified"
        return (
            f"Case ID: {self.case.case_id}\n"
            f"Client: {self.case.name}\n"
            f"Broad presenting concern: {build_broad_presenting_concern(self.case)}\n"
            f"Observed style tendencies: {traits}\n"
            f"Difficulty: {self.difficulty.name}\n"
            f"Embedding model: {self.embedding_model}\n"
            f"Use prerequisites: {self.use_prerequisites}\n"
            f"Reveal options: one_shot={self.enable_one_shot}, "
            f"two_hit={self.enable_two_hit}, margin={self.one_shot_margin:.2f}\n"
            "Initial visibility: name, traits, broad presenting concern only"
        )

    def therapist_history(self, window_size: int) -> list[str]:
        therapist_turns = [item["content"] for item in self.dialogue if item["role"] == "user"]
        return therapist_turns[-window_size:]

    def visible_field_values(self) -> dict[str, str]:
        return {
            field_name: self.field_texts[field_name] if self.revealed_fields[field_name] else "unknown"
            for field_name in ALL_FIELDS
        }


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def load_case(dataset_path: Path, case_id: str) -> ClientCase:
    rows = json.loads(dataset_path.read_text())
    for row in rows:
        if row.get("id") == case_id:
            return ClientCase.from_row(row)
    raise ValueError(f"Case id not found: {case_id}")


def build_base_prompt_payload(state: SimulatorState) -> dict[str, str]:
    visible = state.visible_field_values()
    traits = ", ".join(state.case.type_traits) if state.case.type_traits else "unknown"
    unlocked_lines = []
    labels = {
        "relevant_history": "Relevant Histories",
        "core_beliefs": "Core Beliefs",
        "intermediate_beliefs": "Intermediate Beliefs",
        "coping_strategies": "Coping Strategies",
        "situation": "Current Situation",
        "automatic_thought": "Automatic Thought",
        "emotion": "Emotion",
        "behavior": "Behavior",
    }
    for field_name in ALL_FIELDS:
        value = visible[field_name]
        if value == "unknown":
            continue
        unlocked_lines.append(f"{labels[field_name]}: {value}")

    return {
        "name": state.case.name,
        "traits": traits,
        "broad_presenting_concern": build_broad_presenting_concern(state.case),
        "unlocked_formulation_details": "\n".join(unlocked_lines)
        if unlocked_lines
        else "(No formulation details have been unlocked yet.)",
    }


def render_client_system_prompt(state: SimulatorState) -> str:
    prompt = load_prompt("base_client_prompt.txt").replace("[Client]", state.case.name)
    prompt = prompt.format(**build_base_prompt_payload(state))
    return prompt


def default_transcript_path(state: SimulatorState) -> Path:
    return DEFAULT_TRANSCRIPT_DIR / (
        f"client_simulator_{state.case.case_id}_{state.difficulty.name}.json"
    )


def transcript_text(
    dialogue: list[dict[str, str]],
    window_size: int | None = None,
) -> str:
    entries = dialogue
    if window_size is not None and window_size > 0:
        entries = dialogue[-window_size:]
    if not entries:
        return "(No prior dialogue yet.)"

    lines = []
    for turn in entries:
        role = "Therapist" if turn["role"] == "user" else "Client"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def render_client_user_prompt(
    dialogue: list[dict[str, str]],
    window_size: int = HISTORY_WINDOW_SIZE,
) -> str:
    template = load_prompt("client_user_prompt.txt")
    return template.format(
        dialogue_history=transcript_text(dialogue, window_size=window_size),
        history_window_size=window_size,
    )


def ensure_api_key() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for non-dry-run simulator usage.")


def _prerequisites_met(state: SimulatorState, field_name: str) -> bool:
    if not state.use_prerequisites:
        return True
    prerequisites = FIELD_PREREQUISITES[field_name]
    if not prerequisites:
        return True
    return any(state.revealed_fields[dependency] for dependency in prerequisites)


def _therapist_query_for_field(state: SimulatorState, field_name: str) -> str:
    window = 1 if field_name in EXTERNAL_FIELDS else 2
    history = state.therapist_history(window)
    if not history:
        return ""
    return "\n".join(history)


def _maybe_reveal_field(
    state: SimulatorState,
    field_name: str,
    therapist_turn_embedding: list[float] | None,
    query_cache: dict[str, tuple[str, list[float]]],
) -> dict[str, Any]:
    threshold = state.difficulty.field_thresholds[field_name]
    prerequisites_met = _prerequisites_met(state, field_name)
    already_revealed = state.revealed_fields[field_name]
    if already_revealed or not prerequisites_met:
        return {
            "turn": state.therapist_turns,
            "field": field_name,
            "score": None,
            "threshold": threshold,
            "revealed": False,
            "already_revealed": already_revealed,
            "prerequisites_met": prerequisites_met,
            "query_text": None,
        }

    query_text = _therapist_query_for_field(state, field_name)
    if not query_text:
        return {
            "turn": state.therapist_turns,
            "field": field_name,
            "score": None,
            "threshold": threshold,
            "revealed": False,
            "already_revealed": False,
            "prerequisites_met": True,
            "query_text": query_text,
        }

    if field_name in EXTERNAL_FIELDS and therapist_turn_embedding is not None:
        query_embedding = therapist_turn_embedding
    else:
        cached = query_cache.get(query_text)
        if cached is None:
            cached = (query_text, get_embedding(query_text, model=state.embedding_model))
            query_cache[query_text] = cached
        query_embedding = cached[1]

    score = cosine_similarity(query_embedding, state.field_embeddings[field_name])
    previous_best = state.best_similarity_by_field[field_name]
    state.best_similarity_by_field[field_name] = max(previous_best, score)

    if state.enable_one_shot or state.enable_two_hit:
        should_reveal = False
        if state.enable_one_shot and score >= threshold + state.one_shot_margin:
            should_reveal = True
        if state.enable_two_hit and previous_best >= threshold and score >= threshold:
            should_reveal = True
    else:
        should_reveal = score >= threshold
    if not should_reveal:
        return {
            "turn": state.therapist_turns,
            "field": field_name,
            "score": score,
            "threshold": threshold,
            "revealed": False,
            "already_revealed": False,
            "prerequisites_met": True,
            "enable_one_shot": state.enable_one_shot,
            "enable_two_hit": state.enable_two_hit,
            "one_shot_margin": state.one_shot_margin,
            "query_text": query_text,
        }

    state.revealed_fields[field_name] = True
    event = {
        "turn": state.therapist_turns,
        "field": field_name,
        "score": score,
        "threshold": threshold,
        "revealed": True,
        "already_revealed": False,
        "prerequisites_met": True,
        "enable_one_shot": state.enable_one_shot,
        "enable_two_hit": state.enable_two_hit,
        "one_shot_margin": state.one_shot_margin,
        "query_text": query_text,
    }
    state.reveal_events.append(event)
    state.events.append(
        f"Reveal unlocked {field_name} at turn {state.therapist_turns} "
        f"(score={score:.3f}, threshold={threshold:.3f})."
    )
    return event


def format_judge_status(turn: int, judge_status: dict[str, str]) -> str:
    return (
        f"Reveal status after turn {turn}: "
        f"unlocked={judge_status['unlocked']}; "
        f"visible={judge_status['visible']}"
    )


def maybe_update_state(state: SimulatorState, judge_model: str | None = None) -> dict[str, str]:
    del judge_model

    if not state.dialogue or state.dialogue[-1]["role"] != "user":
        return {
            "unlocked": "none",
            "visible": ", ".join(field for field, shown in state.revealed_fields.items() if shown) or "none",
        }

    latest_therapist_turn = state.dialogue[-1]["content"]
    therapist_turn_embedding = get_embedding(latest_therapist_turn, model=state.embedding_model)
    query_cache: dict[str, tuple[str, list[float]]] = {}
    unlocked: list[str] = []

    for field_name in ALL_FIELDS:
        event = _maybe_reveal_field(
            state=state,
            field_name=field_name,
            therapist_turn_embedding=therapist_turn_embedding,
            query_cache=query_cache,
        )
        event["therapist_text"] = latest_therapist_turn
        state.turn_similarity_events.append(event)
        if event["revealed"]:
            unlocked.append(field_name)

    visible = [field_name for field_name, shown in state.revealed_fields.items() if shown]
    return {
        "unlocked": ", ".join(unlocked) if unlocked else "none",
        "visible": ", ".join(visible) if visible else "none",
    }


def generate_client_reply(state: SimulatorState, model: str) -> str:
    return call_llm(
        system_prompt=render_client_system_prompt(state),
        user_prompt=render_client_user_prompt(state.dialogue, window_size=HISTORY_WINDOW_SIZE),
        temperature=0.3,
        model=model,
    ).strip()


def save_transcript(
    state: SimulatorState,
    path: Path,
    *,
    model: str,
    judge_model: str,
    max_turns: int,
) -> Path:
    del judge_model
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": state.case.case_id,
        "client_name": state.case.name,
        "difficulty": state.difficulty.name,
        "model": model,
        "embedding_model": state.embedding_model,
        "use_prerequisites": state.use_prerequisites,
        "enable_one_shot": state.enable_one_shot,
        "enable_two_hit": state.enable_two_hit,
        "one_shot_margin": state.one_shot_margin,
        "max_turns": max_turns,
        "therapist_turns": state.therapist_turns,
        "revealed_fields": state.revealed_fields,
        "best_similarity_by_field": state.best_similarity_by_field,
        "reveal_events": state.reveal_events,
        "turn_similarity_events": state.turn_similarity_events,
        "dialogue": state.dialogue,
        "events": state.events,
        "transcript_text": transcript_text(state.dialogue),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_interactive_session(
    state: SimulatorState,
    model: str,
    judge_model: str,
    max_turns: int,
    transcript_path: Path,
) -> None:
    ensure_api_key()
    print("Therapist intake")
    print("----------------")
    print(state.therapist_intake())
    print("\nType therapist messages. Type 'exit' to stop.\n")

    try:
        for _ in range(max_turns):
            therapist = input("Therapist> ").strip()
            if not therapist:
                continue
            if therapist.lower() in {"exit", "quit"}:
                break

            state.dialogue.append({"role": "user", "content": therapist})
            state.therapist_turns += 1
            reveal_status = maybe_update_state(state, judge_model)
            print(format_judge_status(state.therapist_turns, reveal_status))

            reply = generate_client_reply(state, model)
            state.dialogue.append({"role": "assistant", "content": reply})
            print(f"Client> {reply}\n")

    except (EOFError, KeyboardInterrupt):
        print()
        print("Session interrupted. Saving transcript.")
    finally:
        saved_path = save_transcript(
            state,
            transcript_path,
            model=model,
            judge_model=judge_model,
            max_turns=max_turns,
        )
        print(f"Transcript saved to {saved_path}")

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
        help="Difficulty used for per-field reveal thresholds",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Client model")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Deprecated compatibility argument; reveal is now deterministic.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model used for therapist-to-CCD similarity scoring.",
    )
    parser.add_argument(
        "--use-prerequisites",
        action="store_true",
        help="Enable prerequisite gating before a field is eligible for scoring.",
    )
    parser.add_argument(
        "--enable-one-shot",
        action="store_true",
        help="Require a single score to exceed threshold plus margin for one-shot reveal logic.",
    )
    parser.add_argument(
        "--enable-two-hit",
        action="store_true",
        help="Require two threshold hits across turns for two-hit reveal logic.",
    )
    parser.add_argument(
        "--one-shot-margin",
        type=float,
        default=DEFAULT_ONE_SHOT_MARGIN,
        help="Extra margin above threshold required for one-shot reveal when enabled.",
    )
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum therapist turns")
    parser.add_argument(
        "--transcript-path",
        help="Optional output path for saved interactive transcript JSON",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print intake and masked prompt only")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_path = Path(args.dataset)
    case = load_case(dataset_path, args.case_id)
    state = SimulatorState(
        case=case,
        difficulty=DIFFICULTIES[args.difficulty],
        embedding_model=args.embedding_model,
        use_prerequisites=args.use_prerequisites,
        enable_one_shot=args.enable_one_shot,
        enable_two_hit=args.enable_two_hit,
        one_shot_margin=args.one_shot_margin,
    )
    if args.dry_run:
        run_dry_run(state)
        return
    transcript_path = (
        Path(args.transcript_path)
        if args.transcript_path
        else default_transcript_path(state)
    )
    run_interactive_session(
        state=state,
        model=args.model,
        judge_model=args.judge_model,
        max_turns=args.max_turns,
        transcript_path=transcript_path,
    )


if __name__ == "__main__":
    main()
