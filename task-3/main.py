
import os
import json
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from invoice import ExtractedInvoice


# Load API key from Day 11 root folder
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


# Initialize Groq client using OpenAI SDK
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"


# Sample invoice text
invoice_text = """
Invoice Number: INV-1001
Vendor: ABC Technologies
Invoice Date: 2026-11-20
Due Date: 2026-12-20

Items:
Laptop - Quantity: 1 - Unit Price: 15000 INR
Mouse - Quantity: 2 - Unit Price: 500 INR

Total Amount: 16000 INR
Notes: Payment within 30 days.
"""


def make_strict_schema(schema):
    """
    Recursively update JSON Schema for strict structured outputs.
    Every object must explicitly disallow additional properties.
    """

    if isinstance(schema, dict):

        if schema.get("type") == "object":
            schema["additionalProperties"] = False

        for value in schema.values():
            make_strict_schema(value)

    elif isinstance(schema, list):

        for item in schema:
            make_strict_schema(item)

    return schema


def extract_invoice():

    # Generate strict JSON Schema from Pydantic model
    invoice_schema = make_strict_schema(
        ExtractedInvoice.model_json_schema()
    )

    # Make API request
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an invoice data extraction assistant. "
                    "Extract the information accurately. "
                    "Use empty strings for unavailable text fields "
                    "and zero for unavailable numeric fields. "
                    "Do not invent missing information."
                )
            },
            {
                "role": "user",
                "content": invoice_text
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "extracted_invoice",
                "strict": True,
                "schema": invoice_schema
            }
        }
    )

    # Extract response text
    raw_output = response.choices[0].message.content

    if not raw_output:
        raise ValueError("The model returned an empty response")

    # Validate API response using Pydantic
    invoice = ExtractedInvoice.model_validate_json(raw_output)

    print("\n" + "=" * 50)
    print("STRUCTURED INVOICE OUTPUT")
    print("=" * 50)

    print(json.dumps(invoice.model_dump(), indent=4))

    print("\nPydantic validation: PASSED")

    # Save result
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / "invoice_output.json"

    output_file.write_text(
        json.dumps(invoice.model_dump(), indent=4),
        encoding="utf-8"
    )

    print(f"\nEvidence saved to: {output_file}")


if __name__ == "__main__":
    extract_invoice()
