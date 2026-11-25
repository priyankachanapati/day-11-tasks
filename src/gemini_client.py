
import os
import time

from google import genai
from dotenv import load_dotenv

from src.metrics import calculate_cost
from google.genai import types

load_dotenv()

MODEL = "gemini-2.5-flash"

# Standard API pricing per 1 million tokens (USD)
INPUT_PRICE = 0.30
OUTPUT_PRICE = 2.50


def call_gemini(prompt: str) -> dict:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing")

    client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        api_version="v1beta"
    )
)

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "system_instruction": (
                "You are a helpful AI engineering tutor. "
                "Explain concepts clearly and concisely."
            ),
            "temperature": 0.2,
            "max_output_tokens": 300,
        },
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Extract token usage metadata
    usage = response.usage_metadata

    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    total_tokens = usage.total_token_count or 0

    # Extract thinking tokens separately
    thinking_tokens = getattr(
        usage, "thoughts_token_count", 0
    ) or 0

    # Include thinking tokens in estimated output cost
    billable_output_tokens = output_tokens + thinking_tokens

    estimated_cost = calculate_cost(
        input_tokens,
        billable_output_tokens,
        INPUT_PRICE,
        OUTPUT_PRICE,
    )

    return {
        "provider": "Google Gemini",
        "model": MODEL,
        "response": response.text,

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "billable_output_tokens": billable_output_tokens,
        "total_tokens": total_tokens,

        "latency_ms": round(latency_ms, 2),

        "input_price_per_million": INPUT_PRICE,
        "output_price_per_million": OUTPUT_PRICE,

        "estimated_cost_usd": round(estimated_cost, 10),

        "actual_billing": "Check Google AI Studio usage",
    }
