
import os
import json
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from tools import get_exchange_rate


# Load API key
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"


# Define the tool for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": (
                "Get an illustrative exchange rate "
                "for a currency against USD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "currency": {
                        "type": "string",
                        "description": "Currency code such as EUR, INR or GBP"
                    }
                },
                "required": ["currency"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


# Execute the requested Python function
def execute_tool(function_name, arguments):

    if function_name == "get_exchange_rate":

        return get_exchange_rate(
            currency=arguments["currency"]
        )

    return {
        "success": False,
        "error": f"Unknown function: {function_name}"
    }


def run_agent():

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful financial assistant. "
                "Use the get_exchange_rate tool whenever "
                "a user requests a currency conversion. "
                "Clearly mention that the supplied rates "
                "are illustrative."
            )
        },
        {
            "role": "user",
            "content": "Convert 100 EUR to USD using the exchange rate tool."
        }
    ]

    tool_calls_count = 0

    print("\nFUNCTION CALLING EXPERIMENT")
    print("=" * 50)

    # Tool calling loop
    for iteration in range(5):

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0
        )

        assistant_message = response.choices[0].message

        # If the model does not request a tool
        if not assistant_message.tool_calls:

            final_answer = assistant_message.content

            print("\nFINAL LLM RESPONSE:")
            print(final_answer)

            result = {
                "model": MODEL,
                "tool_calls": tool_calls_count,
                "final_response": final_answer,
                "status": "completed"
            }

            results_dir = Path(__file__).parent / "results"
            results_dir.mkdir(exist_ok=True)

            output_file = results_dir / "tool_call_result.json"

            output_file.write_text(
                json.dumps(result, indent=4),
                encoding="utf-8"
            )

            print(f"\nEvidence saved to: {output_file}")

            return

        # Add the assistant's tool request to conversation
        messages.append(assistant_message)

        # Execute every requested tool
        for tool_call in assistant_message.tool_calls:

            function_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(f"\nTool requested: {function_name}")
            print(f"Arguments: {arguments}")

            tool_result = execute_tool(
                function_name,
                arguments
            )

            tool_calls_count += 1

            print("Tool result:")
            print(json.dumps(tool_result, indent=4))

            # Return the function result to the LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result)
            })

    raise RuntimeError("Maximum tool-calling iterations exceeded")


if __name__ == "__main__":
    run_agent()
