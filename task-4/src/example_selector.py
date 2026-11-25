
EXAMPLES = [
    {
        "id": "EX001",
        "ticket": {
            "ticket_id": "EX001",
            "title": "Search results not relevant",
            "description": "Searching for a product returns unrelated results."
        },
        "output": {
            "category": "Technical",
            "priority": "Medium",
            "summary": "Product search returns irrelevant results.",
            "sentiment": "Negative",
            "requires_escalation": False
        },
        "keywords": [
            "search", "search results", "irrelevant",
            "unrelated results", "wrong results"
        ]
    },
    {
        "id": "EX002",
        "ticket": {
            "ticket_id": "EX002",
            "title": "Change button colour",
            "description": "Please make the Save button blue to match the brand."
        },
        "output": {
            "category": "Technical",
            "priority": "Low",
            "summary": "User requests a change to the Save button color.",
            "sentiment": "Neutral",
            "requires_escalation": False
        },
        "keywords": [
            "button", "colour", "color", "ui", "interface",
            "design", "display", "css", "branding"
        ]
    },
    {
        "id": "EX003",
        "ticket": {
            "ticket_id": "EX003",
            "title": "Font too small on FAQ",
            "description": "FAQ answers are difficult to read on mobile."
        },
        "output": {
            "category": "Technical",
            "priority": "Low",
            "summary": "FAQ text is too small on mobile devices.",
            "sentiment": "Negative",
            "requires_escalation": False
        },
        "keywords": [
            "font", "text size", "small text", "read",
            "mobile", "faq", "layout"
        ]
    },
    {
        "id": "EX004",
        "ticket": {
            "ticket_id": "EX004",
            "title": "Production website unavailable",
            "description": "The main website returns a 500 error for all users."
        },
        "output": {
            "category": "Technical",
            "priority": "Critical",
            "summary": "Production website is unavailable due to a server error.",
            "sentiment": "Negative",
            "requires_escalation": True
        },
        "keywords": [
            "production", "500 error", "outage",
            "website down", "server error", "unavailable"
        ]
    },
    {
        "id": "EX005",
        "ticket": {
            "ticket_id": "EX005",
            "title": "Incorrect invoice",
            "description": "The latest invoice contains an incorrect charge."
        },
        "output": {
            "category": "Billing",
            "priority": "Medium",
            "summary": "User reports an incorrect invoice charge.",
            "sentiment": "Negative",
            "requires_escalation": False
        },
        "keywords": [
            "invoice", "payment", "billing",
            "charge", "refund", "amount"
        ]
    },
    {
        "id": "EX006",
        "ticket": {
            "ticket_id": "EX006",
            "title": "VPN access denied",
            "description": "I cannot access the company VPN because permission is denied."
        },
        "output": {
            "category": "Access",
            "priority": "High",
            "summary": "User cannot access the company VPN.",
            "sentiment": "Negative",
            "requires_escalation": False
        },
        "keywords": [
            "vpn", "permission", "access denied",
            "authorization", "access", "login access"
        ]
    },
    {
        "id": "EX007",
        "ticket": {
            "ticket_id": "EX007",
            "title": "Account locked",
            "description": "My account is locked and I cannot sign in."
        },
        "output": {
            "category": "Account",
            "priority": "Medium",
            "summary": "User cannot sign in because the account is locked.",
            "sentiment": "Negative",
            "requires_escalation": False
        },
        "keywords": [
            "account", "locked account", "password",
            "sign in", "login", "profile"
        ]
    },
    {
        "id": "EX008",
        "ticket": {
            "ticket_id": "EX008",
            "title": "General support question",
            "description": "I have a general question about the support process."
        },
        "output": {
            "category": "Other",
            "priority": "Low",
            "summary": "User has a general question about support.",
            "sentiment": "Neutral",
            "requires_escalation": False
        },
        "keywords": [
            "general question", "support process",
            "information request", "general inquiry"
        ]
    }
]


def select_examples(ticket, limit=2):
    """Select examples using keyword overlap with deterministic ranking."""

    text = (
        str(ticket.get("title", "")) + " " +
        str(ticket.get("description", ""))
    ).lower()

    scored = []

    for example in EXAMPLES:
        score = sum(
            1 for keyword in example["keywords"]
            if keyword.lower() in text
        )

        if score > 0:
            scored.append((score, example))

    scored.sort(
        key=lambda item: (-item[0], item[1]["id"])
    )

    selected = [
        example for _, example in scored[:limit]
    ]

    # Use general examples only when no specific match is found.
    if not selected:
        selected = [
            EXAMPLES[-1]
        ]

    return selected
