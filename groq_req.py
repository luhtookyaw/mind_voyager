import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def call_groq_messages(
    messages: list,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.3,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Check your .env file or shell environment.")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content

