
import json
import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR.parent / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in Day 11 .env file.")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"

Priority = Literal["Low", "Medium", "High", "Critical"]


# --------------------------------------------------
# 2. STRUCTURED OUTPUT MODEL
# --------------------------------------------------

class TicketPrediction(BaseModel):
    priority: Priority


# --------------------------------------------------
# 3. SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an IT support ticket priority classifier.

Classify each ticket into exactly one priority:

- Critical: Major outage, severe business impact,
  or critical service unavailable.
- High: Significant functionality affected or
  urgent business impact.
- Medium: Normal issue affecting a user or workflow
  without major disruption.
- Low: Minor issue, cosmetic problem, or general request.

Return only a valid JSON object with this structure:
{"priority": "Low"}

The priority must be exactly one of:
Low, Medium, High, Critical.
"""


# --------------------------------------------------
# 4. FORMAT A TICKET
# --------------------------------------------------

def format_ticket(ticket):
    return (
        f"Title: {ticket['title']}\n"
        f"Description: {ticket['description']}\n"
        f"Channel: {ticket['channel']}"
    )


# --------------------------------------------------
# 5. BUILD FEW-SHOT EXAMPLES
# --------------------------------------------------

def build_few_shot_examples(train_df):
    """
    Select 3 examples for each priority class.
    Uses training data only.
    """

    examples = []

    priority_order = [
        "Critical",
        "High",
        "Medium",
        "Low"
    ]

    for priority in priority_order:

        class_df = train_df[
            train_df["priority"] == priority
        ].copy()

        # Calculate the combined title and description length.
        class_df["text_length"] = (
            class_df["title"].fillna("").str.len()
            + class_df["description"].fillna("").str.len()
        )

        # Select the 3 most informative examples by length.
        class_df = class_df.sort_values(
            by="text_length",
            ascending=False
        )

        selected = class_df.head(3)

        for _, row in selected.iterrows():

            examples.append({
                "ticket": format_ticket(row),
                "priority": row["priority"]
            })

    return examples


# --------------------------------------------------
# 6. CLASSIFY A TICKET
# --------------------------------------------------

def classify_ticket(ticket, mode, examples=None):

    user_prompt = ""

    if mode == "few_shot":

        user_prompt += (
            "Here are examples of correctly classified tickets:\n"
        )

        for example in examples or []:

            user_prompt += (
                f"\nTicket:\n{example['ticket']}\n"
                f"Correct priority: {example['priority']}\n"
            )

        user_prompt += "\nNow classify this new ticket:\n"

    user_prompt += format_ticket(ticket)

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    raw_output = response.choices[0].message.content

    try:
        parsed = TicketPrediction.model_validate_json(
            raw_output
        )

        prediction = parsed.priority
        validation_error = None

    except (ValidationError, ValueError, TypeError) as error:

        prediction = None
        validation_error = str(error)

    usage = response.usage

    return {
        "prediction": prediction,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "validation_error": validation_error
    }


# --------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------

def main():

    train_df = pd.read_csv(
        BASE_DIR / "data" / "train.csv"
    )

    test_df = pd.read_csv(
        BASE_DIR / "data" / "test.csv"
    )

    # Full test set: 152 tickets.
    sample_df = test_df

    # Build examples using training data only.
    examples = build_few_shot_examples(train_df)

    # --------------------------------------------------
    # VERIFY FEW-SHOT EXAMPLES
    # --------------------------------------------------

    print("\nFew-shot example verification:")

    for priority in [
        "Critical",
        "High",
        "Medium",
        "Low"
    ]:

        count = sum(
            1
            for example in examples
            if example["priority"] == priority
        )

        print(f"{priority}: {count} examples")

    print("Total examples:", len(examples))


    # --------------------------------------------------
    # RUN ZERO-SHOT AND FEW-SHOT EVALUATION
    # --------------------------------------------------

    print("\nRunning full evaluation for both modes...\n")

    print(
        "Total planned API calls:",
        len(sample_df) * 2
    )

    all_results = []

    for mode in ["zero_shot", "few_shot"]:

        print(f"\n--- {mode.upper()} ---")

        for _, ticket in sample_df.iterrows():

            try:

                result = classify_ticket(
                    ticket,
                    mode,
                    examples
                )

                record = {
                    "ticket_id": ticket["ticket_id"],
                    "mode": mode,
                    "actual_priority": ticket["priority"],
                    **result
                }

                all_results.append(record)

                print(
                    f"{ticket['ticket_id']}: "
                    f"Actual={ticket['priority']}, "
                    f"Predicted={result['prediction']}, "
                    f"Latency={result['latency_ms']} ms"
                )

                if result["validation_error"]:
                    print(
                        "Validation error:",
                        result["validation_error"]
                    )

            except Exception as error:

                print(
                    f"Error for ticket {ticket['ticket_id']}: {error}"
                )

                all_results.append({
                    "ticket_id": ticket["ticket_id"],
                    "mode": mode,
                    "actual_priority": ticket["priority"],
                    "prediction": None,
                    "latency_ms": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "validation_error": str(error)
                })

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    output_path = (
        BASE_DIR / "results" / "llm_smoke_test.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(all_results, indent=4)
    )

    successful = sum(
        result["prediction"] is not None
        for result in all_results
    )

    print("\nEvaluation completed.")
    print("Total attempts:", len(all_results))
    print("Successful predictions:", successful)
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
