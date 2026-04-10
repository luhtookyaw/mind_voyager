# llm.py
import os
from math import sqrt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Thin wrapper around OpenAI chat completion.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.choices[0].message.content

def call_llm_messages(
    messages: list,
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Use this for multi-turn (history) calls.
    messages example:
    [{"role":"system","content":"..."}, {"role":"user","content":"..."}, ...]
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages
    )
    return response.choices[0].message.content


def get_embedding(
    text: str,
    model: str = "text-embedding-ada-002",
) -> list[float]:
    """
    Thin wrapper around OpenAI embeddings.
    """
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
