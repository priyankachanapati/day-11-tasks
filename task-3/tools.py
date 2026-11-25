
def get_exchange_rate(currency: str) -> dict:
    """
    Return an illustrative exchange rate against USD.
    """

    exchange_rates = {
        "EUR": 1.08,
        "INR": 0.012,
        "GBP": 1.27,
        "JPY": 0.0067,
        "CAD": 0.74,
    }

    currency = currency.upper()

    if currency not in exchange_rates:
        return {
            "success": False,
            "error": f"Exchange rate unavailable for {currency}"
        }

    return {
        "success": True,
        "currency": currency,
        "base_currency": "USD",
        "rate_to_usd": exchange_rates[currency],
        "description": f"1 {currency} = {exchange_rates[currency]} USD"
    }
