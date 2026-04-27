from __future__ import annotations

from groq_req import call_groq_messages
from llm import call_llm
from mind_voyager.client_simulator import (
    ClientCase,
    HISTORY_WINDOW_SIZE,
    load_prompt,
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


def therapist_dialogue_history_text(
    transcript: list[dict[str, str]],
    window_size: int = HISTORY_WINDOW_SIZE,
) -> str:
    entries = transcript[-window_size:] if window_size > 0 else transcript
    if not entries:
        return "(No prior dialogue yet.)"

    lines = []
    for item in entries:
        speaker = "Client" if item["role"] == "user" else "Therapist"
        lines.append(f"{speaker}: {item['content']}")
    return "\n".join(lines)


def render_therapist_user_prompt(
    transcript: list[dict[str, str]],
    window_size: int = HISTORY_WINDOW_SIZE,
) -> str:
    template = load_prompt("therapist_user_prompt.txt")
    return template.format(
        dialogue_history=therapist_dialogue_history_text(transcript, window_size=window_size),
        history_window_size=window_size,
    )


def generate_therapist_reply(
    therapist_prompt: str,
    transcript: list[dict[str, str]],
    model: str,
    provider: str,
) -> str:
    user_prompt = render_therapist_user_prompt(
        transcript=transcript,
        window_size=HISTORY_WINDOW_SIZE,
    )
    messages = [
        {"role": "system", "content": therapist_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if provider == "groq":
        return call_groq_messages(messages=messages, temperature=0.3, model=model).strip()
    return call_llm(
        system_prompt=therapist_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        model=model,
    ).strip()


def latest_client_utterance(transcript: list[dict[str, str]]) -> str | None:
    for item in reversed(transcript):
        if item["role"] == "user":
            return item["content"]
    return None
