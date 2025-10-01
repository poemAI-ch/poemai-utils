#!/usr/bin/env python3
"""
Lean OpenAI Responses API demo using plain requests (no openai SDK).
- Defines a function tool (get_fx_rate)
- Lets the model call it
- Submits tool output via previous_response_id + function_call_output
- Prints the final text

Usage:
  export OPENAI_API_KEY=sk-...
  python try_openai_requests.py
"""
import json
import os

import requests

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-5"

if not API_KEY:
    raise SystemExit("Set OPENAI_API_KEY.")

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
)

# ------------------------- Tool schema ---------------------------------------
TOOLS = [
    {
        "type": "function",
        "name": "get_fx_rate",
        "description": "Get the spot exchange rate base->quote (e.g., CHF->EUR).",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "3-letter ISO code"},
                "quote": {"type": "string", "description": "3-letter ISO code"},
            },
            "required": ["base", "quote"],
            "additionalProperties": False,
        },
    }
]


# --------------------- Your local tool implementation ------------------------
def get_fx_rate_impl(base: str, quote: str) -> dict:
    stub = {
        ("CHF", "EUR"): 1.04,
        ("EUR", "CHF"): 0.96,
        ("USD", "CHF"): 0.89,
        ("CHF", "USD"): 1.12,
    }
    r = stub.get((base.upper(), quote.upper()))
    if r is None:
        return {"ok": False, "error": f"FX rate {base}->{quote} not found"}
    return {"ok": True, "rate": r, "base": base.upper(), "quote": quote.upper()}


# ------------------ Small helpers to work with Responses JSON ----------------
def responses_create(**payload) -> dict:
    print("REQUEST:\n", json.dumps(payload, indent=2, ensure_ascii=False), "\n")

    resp = SESSION.post(f"{BASE_URL}/responses", data=json.dumps(payload))
    if not resp.ok:
        raise RuntimeError(f"responses.create HTTP {resp.status_code}: {resp.text}")
    response_json = resp.json()
    print("RESPONSE:\n", json.dumps(response_json, indent=2, ensure_ascii=False), "\n")
    return response_json


def extract_function_calls(resp_json: dict):
    """
    Extract [{'call_id', 'name', 'arguments'}] from a Responses API JSON.
    The Responses API emits 'output' -> list[...] items; function calls usually
    appear as objects with type == 'function_call'. We also allow for nested shapes.
    """
    out = []

    def walk(x):
        if isinstance(x, dict):
            # Primary: objects with type == "function_call"
            if x.get("type") == "function_call":
                fn_name = x.get("name") or x.get("function", {}).get("name")
                args = (
                    x.get("arguments") or x.get("function", {}).get("arguments") or "{}"
                )
                out.append(
                    {
                        "call_id": x.get("call_id") or x.get("id"),
                        "name": fn_name,
                        "arguments": args,
                    }
                )
            # Back-compat: nested "tool_calls" as seen in some variants
            if "tool_calls" in x and isinstance(x["tool_calls"], list):
                for tc in x["tool_calls"]:
                    fn = tc.get("function", {})
                    out.append(
                        {
                            "call_id": tc.get("id"),
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments") or "{}",
                        }
                    )
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(resp_json.get("output", []))
    return out


def extract_text(resp_json: dict) -> str:
    """
    Collect model-generated text; the Responses API often places user-facing text
    under items with type 'output_text' or 'text'.
    """
    chunks = []

    def walk(x):
        if isinstance(x, dict):
            t = x.get("type")
            if t in ("output_text", "text"):
                val = x.get("text") or x.get("content")
                if isinstance(val, str):
                    chunks.append(val)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(resp_json.get("output", []))
    return "\n".join(chunks).strip()


# ------------------------------- Turn 1 --------------------------------------
first = responses_create(
    model=MODEL,
    instructions=(
        "You’re a precise finance assistant. "
        "If currency conversion is requested, you MUST call get_fx_rate and then compute the result."
    ),
    tools=TOOLS,
    input=[
        {"role": "user", "content": "Convert 120 CHF to EUR and show your calculation."}
    ],
)

calls = extract_function_calls(first)
if not calls:
    print(extract_text(first) or json.dumps(first, indent=2))
    raise SystemExit(0)

# Run tools locally
tool_outputs = []
for c in calls:
    args = json.loads(c["arguments"] or "{}")
    if c["name"] == "get_fx_rate":
        result = get_fx_rate_impl(args.get("base", "CHF"), args.get("quote", "EUR"))
    else:
        result = {"ok": False, "error": f"Unknown tool {c['name']}"}
    tool_outputs.append(
        {
            "type": "function_call_output",
            "call_id": c["call_id"],
            "output": json.dumps(result, ensure_ascii=False),
        }
    )

# ------------------------------- Turn 2 --------------------------------------
second = responses_create(
    model=MODEL,
    previous_response_id=first["id"],
    input=tool_outputs,  # one or more function_call_output items
)

print(extract_text(second) or json.dumps(second, indent=2))
