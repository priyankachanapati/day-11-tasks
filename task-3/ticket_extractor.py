
import os
import csv
import json
import time

from pathlib import Path
from typing import Literal

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"

DATA_FILE = Path(__file__).parent / "data" / "tickets_sample.csv"
RESULTS_DIR = Path(__file__).parent / "results"


# --------------------------------------------------
# 2. Define Pydantic model
# --------------------------------------------------

class ExtractedTicket(BaseModel):

    ticket_id: str = Field(
        description="Original support ticket identifier"
    )

    category: Literal[
        "Technical",
        "Account",
        "Billing",
        "Access",
        "Other"
    ]

    priority: Literal[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]

    summary: str = Field(
        description="Concise summary of the ticket"
    )

    sentiment: Literal[
        "Positive",
        "Neutral",
        "Negative"
    ]

    requires_escalation: bool = Field(
        description="Whether the issue requires support escalation"
    )


# --------------------------------------------------
# 3. Make JSON schema compatible with strict mode
# --------------------------------------------------

def make_strict_schema(schema):

    if isinstance(schema, dict):

        if schema.get("type") == "object":
            schema["additionalProperties"] = False

        for value in schema.values():
            make_strict_schema(value)

    elif isinstance(schema, list):

        for item in schema:
            make_strict_schema(item)

    return schema


TICKET_SCHEMA = make_strict_schema(
    ExtractedTicket.model_json_schema()
)


# --------------------------------------------------
# 4. Extract one ticket with retry mechanism
# --------------------------------------------------

def extract_ticket(ticket, max_retries=3):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional support ticket analyst. "
                "Extract the ticket information accurately. "
                "Classify the category, priority and sentiment. "
                "Do not invent information. "
                "Return only information supported by the ticket."
            )
        },
        {
            "role": "user",
            "content": json.dumps(ticket)
        }
    ]

    for attempt in range(1, max_retries + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extracted_ticket",
                        "strict": True,
                        "schema": TICKET_SCHEMA
                    }
                },
                max_completion_tokens=500
            )

            raw_output = response.choices[0].message.content

            if not raw_output:
                raise ValueError("Empty model response")

            extracted = ExtractedTicket.model_validate_json(
                raw_output
            )

            # Verify that the original ticket ID is preserved
            if extracted.ticket_id != str(ticket["ticket_id"]):
                raise ValueError("Ticket ID mismatch")

            return {
                "valid": True,
                "retry_count": attempt - 1,
                "data": extracted.model_dump(),
                "error": None
            }

        except (ValidationError, ValueError, json.JSONDecodeError) as error:

            print(
                f"Validation attempt {attempt} failed "
                f"for ticket {ticket['ticket_id']}: {error}"
            )

            messages.append({
                "role": "assistant",
                "content": (
                    raw_output if "raw_output" in locals()
                    and raw_output else "{}"
                )
            })

            messages.append({
                "role": "user",
                "content": (
                    "Your previous response failed validation. "
                    "Correct the errors and return a valid object "
                    "matching the required schema. "
                    f"Error: {str(error)}"
                )
            })

            raw_output = None

        except Exception as error:

            print(
                f"API error for ticket {ticket['ticket_id']}: {error}"
            )

            return {
                "valid": False,
                "retry_count": attempt - 1,
                "data": None,
                "error": str(error)
            }

    return {
        "valid": False,
        "retry_count": max_retries,
        "data": None,
        "error": "Maximum validation retries exceeded"
    }


# --------------------------------------------------
# 5. Process 20 tickets
# --------------------------------------------------

def main():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    with DATA_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        reader = csv.DictReader(file)
        tickets = list(reader)[:20]

    if len(tickets) != 20:
        raise ValueError(
            f"Expected 20 tickets, found {len(tickets)}"
        )

    results = []

    print("\nSTRUCTURED TICKET EXTRACTION")
    print("=" * 55)

    for index, ticket in enumerate(tickets, start=1):

        print(
            f"\nProcessing ticket {index}/20: "
            f"{ticket['ticket_id']}"
        )

        result = extract_ticket(ticket)

        results.append(result)

        if result["valid"]:
            print("Pydantic validation: PASSED")
            print(f"Retries: {result['retry_count']}")
        else:
            print("Pydantic validation: FAILED")
            print(f"Error: {result['error']}")

        # Small delay between requests
        time.sleep(0.5)

    # --------------------------------------------------
    # 6. Calculate validation statistics
    # --------------------------------------------------

    total = len(results)

    valid_count = sum(
        1 for result in results if result["valid"]
    )

    failed_count = total - valid_count

    total_retries = sum(
        result["retry_count"] for result in results
    )

    success_rate = (valid_count / total) * 100

    report = {
        "model": MODEL,
        "total_tickets": total,
        "valid_tickets": valid_count,
        "failed_tickets": failed_count,
        "total_retries": total_retries,
        "validation_success_rate": round(success_rate, 2),
        "pass_threshold": 95,
        "passed": success_rate >= 95
    }

    # --------------------------------------------------
    # 7. Save evidence
    # --------------------------------------------------

    RESULTS_DIR.mkdir(exist_ok=True)

    extracted_data = [
        result["data"]
        for result in results
        if result["valid"]
    ]

    (RESULTS_DIR / "extracted_tickets.json").write_text(
        json.dumps(extracted_data, indent=4),
        encoding="utf-8"
    )

    (RESULTS_DIR / "validation_report.json").write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)

    print(f"Total tickets: {total}")
    print(f"Valid tickets: {valid_count}")
    print(f"Failed tickets: {failed_count}")
    print(f"Total retries: {total_retries}")
    print(f"Success rate: {success_rate:.2f}%")

    print(
        "Result:",
        "PASSED" if report["passed"] else "NOT PASSED"
    )

    print("\nEvidence saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
