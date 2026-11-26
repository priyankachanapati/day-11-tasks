
import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

results_path = BASE_DIR / "results" / "llm_smoke_test.json"

results = json.loads(results_path.read_text())

print("\nTOKEN USAGE AND LATENCY SUMMARY")
print("=" * 55)

for mode in ["zero_shot", "few_shot"]:

    rows = [
        row for row in results
        if row["mode"] == mode
    ]

    input_tokens = sum(
        row["input_tokens"] for row in rows
    )

    output_tokens = sum(
        row["output_tokens"] for row in rows
    )

    latencies = sorted(
        row["latency_ms"]
        for row in rows
        if row["latency_ms"] is not None
    )

    p95_index = math.ceil(0.95 * len(latencies)) - 1
    p95 = latencies[p95_index]

    print(f"\nMode: {mode.upper()}")
    print("Tickets:", len(rows))
    print("Input tokens:", input_tokens)
    print("Output tokens:", output_tokens)
    print("Total tokens:", input_tokens + output_tokens)
    print("p95 latency (ms):", round(p95, 2))

    print("-" * 35)
