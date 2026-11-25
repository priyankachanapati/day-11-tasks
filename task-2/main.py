
import os
import json
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

# Load the Day 11 root .env file
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

load_dotenv(ENV_FILE)

print("Loading environment from:", ENV_FILE)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from the root .env file")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"

SYSTEM_MESSAGE = (
    "You are a helpful AI engineering tutor. "
    "Explain concepts clearly with practical examples."
)

USER_PROMPT = (
    "Explain what an API is and how an LLM API works. "
    "Give a simple real-world example."
)


def generate_response(temperature):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE
        },
        {
            "role": "user",
            "content": USER_PROMPT
        }
    ]

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=600
    )

    latency = (time.perf_counter() - start_time) * 1000

    answer = response.choices[0].message.content
    usage = response.usage

    result = {
        "provider": "Groq (OpenAI-compatible SDK)",
        "model": MODEL,
        "temperature": temperature,
        "response": answer,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": round(latency, 2)
    }

    print("\n" + "=" * 60)
    print(f"TEMPERATURE: {temperature}")
    print("=" * 60)

    print("\nGenerated Response:\n")
    print(answer)

    print("\nUsage Information:")
    print(f"Input Tokens: {usage.prompt_tokens}")
    print(f"Output Tokens: {usage.completion_tokens}")
    print(f"Total Tokens: {usage.total_tokens}")
    print(f"Latency: {latency:.2f} ms")

    return result


def main():

    print("LLM CHAT COMPLETION EXPERIMENT")
    print(f"Provider: Groq")
    print(f"Model: {MODEL}")

    results = []

    # Default temperature
    results.append(generate_response(None))

    # Temperature 0
    results.append(generate_response(0.0))

    # Temperature 1
    results.append(generate_response(1.0))

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / "chat_completion_results.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nExperiment completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
