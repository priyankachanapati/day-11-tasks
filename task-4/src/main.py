
import csv
import html
import json
import os
import time
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from example_selector import select_examples


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "v2.0"

PROMPT_FILE = (
    ROOT_DIR / "task-4" / "prompts" / "prompt_v2.txt"
)

DATA_FILE = (
    ROOT_DIR / "task-3" / "data" / "tickets_sample.csv"
)

RESULTS_DIR = ROOT_DIR / "task-4" / "results"


# --------------------------------------------------
# 2. Pydantic output model
# --------------------------------------------------

class ExtractedTicket(BaseModel):

    ticket_id: str

    category: Literal[
        "Technical", "Account", "Billing", "Access", "Other"
    ]

    priority: Literal[
        "Low", "Medium", "High", "Critical"
    ]

    summary: str = Field(
        description="Concise summary of the support ticket"
    )

    sentiment: Literal[
        "Positive", "Neutral", "Negative"
    ]

    requires_escalation: bool


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
# 3. Load versioned XML prompt
# --------------------------------------------------

def load_prompt():

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(PROMPT_FILE)

    root = ET.fromstring(
        PROMPT_FILE.read_text(encoding="utf-8")
    )

    system_element = root.find("system_instructions")
    user_element = root.find("user_context")

    if system_element is None or user_element is None:
        raise ValueError(
            "Prompt must contain system_instructions "
            "and user_context elements"
        )

    system_prompt = "".join(
        system_element.itertext()
    ).strip()

    user_template = ET.tostring(
        user_element,
        encoding="unicode"
    )

    return system_prompt, user_template


# --------------------------------------------------
# 4. Build XML-tagged few-shot context
# --------------------------------------------------

def build_prompt(ticket, user_template):

    selected_examples = select_examples(
        ticket,
        limit=2
    )

    examples_xml = ""

    for example in selected_examples:

        example_input = html.escape(
            json.dumps(
                example["ticket"],
                ensure_ascii=False
            )
        )

        example_output = html.escape(
            json.dumps(
                example["output"],
                ensure_ascii=False
            )
        )

        examples_xml += f"""
        <example id="{example['id']}">
            <input>{example_input}</input>
            <expected_output>{example_output}</expected_output>
        </example>
        """

    ticket_xml = html.escape(
        json.dumps(ticket, ensure_ascii=False)
    )

    user_prompt = (
        user_template
        .replace("{few_shot_examples}", examples_xml)
        .replace("{ticket_context}", ticket_xml)
    )

    return user_prompt, selected_examples


# --------------------------------------------------
# 5. Process one ticket
# --------------------------------------------------

def process_ticket(ticket, system_prompt, user_template):

    user_prompt, selected_examples = build_prompt(
        ticket,
        user_template
    )

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "extracted_ticket",
                "strict": True,
                "schema": TICKET_SCHEMA
            }
        },
        max_completion_tokens=500,
        temperature=0
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    raw_output = response.choices[0].message.content

    if not raw_output:
        raise ValueError("Empty model response")

    extracted = ExtractedTicket.model_validate_json(
        raw_output
    )

    if extracted.ticket_id != str(ticket["ticket_id"]):
        raise ValueError("Ticket ID mismatch")

    usage = response.usage

    return {
        "ticket_id": str(ticket["ticket_id"]),
        "prompt_version": PROMPT_VERSION,
        "selected_examples": [
            example["id"]
            for example in selected_examples
        ],
        "input": ticket,
        "output": extracted.model_dump(),
        "valid": True,
        "latency_ms": latency_ms,
        "token_usage": {
            "input_tokens": (
                usage.prompt_tokens if usage else None
            ),
            "output_tokens": (
                usage.completion_tokens if usage else None
            ),
            "total_tokens": (
                usage.total_tokens if usage else None
            )
        }
    }


# --------------------------------------------------
# 6. Run five test inputs
# --------------------------------------------------

def main():

    system_prompt, user_template = load_prompt()

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    with DATA_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        tickets = list(csv.DictReader(file))[:5]

    if len(tickets) != 5:
        raise ValueError(
            f"Expected 5 tickets, found {len(tickets)}"
        )

    results = []

    print("\nPROMPT AND CONTEXT ENGINEERING - V2")
    print("=" * 60)
    print("Prompt version:", PROMPT_VERSION)
    print("Model:", MODEL)
    print("Test inputs:", len(tickets))

    for index, ticket in enumerate(tickets, start=1):

        print(
            f"\nTest {index}/5: {ticket['ticket_id']}"
        )

        try:

            result = process_ticket(
                ticket,
                system_prompt,
                user_template
            )

            results.append(result)

            print(
                "Selected examples:",
                result["selected_examples"]
            )

            print(
                "Category:",
                result["output"]["category"]
            )

            print(
                "Priority:",
                result["output"]["priority"]
            )

            print("Validation: PASSED")

            print(
                "Latency:",
                result["latency_ms"],
                "ms"
            )

            print(
                "Tokens:",
                result["token_usage"]["total_tokens"]
            )

        except Exception as error:

            print("Test failed:", error)

            results.append({
                "ticket_id": str(ticket["ticket_id"]),
                "prompt_version": PROMPT_VERSION,
                "valid": False,
                "error": str(error)
            })

    # --------------------------------------------------
    # 7. Save v2 results separately
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    valid_count = sum(
        1 for result in results if result["valid"]
    )

    report = {
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "total_tests": len(results),
        "passed_tests": valid_count,
        "failed_tests": len(results) - valid_count,
        "success_rate": round(
            valid_count / len(results) * 100,
            2
        ),
        "results": results
    }

    output_file = (
        RESULTS_DIR / "prompt_v2_test_results.json"
    )

    output_file.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("V2 PROMPT TEST SUMMARY")
    print("=" * 60)

    print("Total tests:", len(results))
    print("Passed:", valid_count)
    print("Failed:", len(results) - valid_count)
    print("Success rate:", report["success_rate"], "%")

    print("Evidence saved:", output_file)


if __name__ == "__main__":
    main()
