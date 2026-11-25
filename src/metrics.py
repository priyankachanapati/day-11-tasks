
def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """
    Calculate estimated API cost in USD.
    """

    input_cost = (
        input_tokens * input_price_per_million / 1_000_000
    )

    output_cost = (
        output_tokens * output_price_per_million / 1_000_000
    )

    return input_cost + output_cost
