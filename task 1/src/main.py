
import json
from pathlib import Path

from src.gemini_client import call_gemini
from src.groq_client import call_groq


PROMPT = """
Explain what a Large Language Model is.
Describe next-token prediction in simple terms.
Give one practical example.
Keep your answer concise.
"""


def main():

    results = []

    providers = [
        ("Gemini", call_gemini),
        ("Groq", call_groq),
    ]

    for name, api_function in providers:

        print(f"\nCalling {name} API...")

        try:
            result = api_function(PROMPT)
            results.append(result)

            print(f"Provider: {result['provider']}")
            print(f"Model: {result['model']}")
            print(f"Input tokens: {result['input_tokens']}")
            print(f"Output tokens: {result['output_tokens']}")
            print(f"Total tokens: {result['total_tokens']}")
            print(f"Latency: {result['latency_ms']} ms")
            print(
                f"Estimated cost: "
                f"${result['estimated_cost_usd']:.8f}"
            )

        except Exception as error:
            print(f"{name} API failed: {error}")

    results_folder = Path("results")
    results_folder.mkdir(exist_ok=True)

    output_file = results_folder / "api_comparison.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nComparison completed.")
    print(f"Evidence saved to: {output_file}")


if __name__ == "__main__":
    main()
