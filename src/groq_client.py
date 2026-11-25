
import os
import time

from groq import Groq
from dotenv import load_dotenv

from src.metrics import calculate_cost

load_dotenv()

MODEL = "openai/gpt-oss-120b"

INPUT_PRICE = 0.15
OUTPUT_PRICE = 0.60


def call_groq(prompt: str) -> dict:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing")

    client = Groq(api_key=api_key)

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI engineering tutor. "
                    "Explain concepts clearly and concisely."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=300,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    usage = response.usage

    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens

    estimated_cost = calculate_cost(
        input_tokens,
        output_tokens,
        INPUT_PRICE,
        OUTPUT_PRICE,
    )

    return {
        "provider": "Groq",
        "model": MODEL,
        "response": response.choices[0].message.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": round(latency_ms, 2),
        "estimated_cost_usd": estimated_cost,
        "actual_billing": "Check Groq usage dashboard",
    }
