import json
import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def get_fx_rate_impl(base: str, quote: str) -> dict:
    stub = {
        ("CHF", "EUR"): 1.04,
        ("EUR", "CHF"): 0.96,
        ("USD", "CHF"): 0.89,
        ("CHF", "USD"): 1.12,
    }
    r = stub.get((base.upper(), quote.upper()))
    return (
        {"ok": bool(r), "rate": r, "base": base.upper(), "quote": quote.upper()}
        if r
        else {"ok": False, "error": f"FX rate {base}->{quote} not found"}
    )


def extract_function_calls(resp):
    """
    Return a list of {'call_id', 'name', 'arguments'} from Responses API output.
    Works against the structured 'output' list.
    """
    out = []
    # Prefer reading the pydantic model directly if available
    data = json.loads(resp.model_dump_json())

    def walk(x):
        if isinstance(x, dict):
            # New Responses API emits items with type == "function_call"
            if x.get("type") == "function_call":
                fn = x.get("name") or x.get("function", {}).get("name")
                args = x.get("arguments")
                if args is None:
                    args = x.get("function", {}).get("arguments")
                out.append(
                    {
                        "call_id": x.get("call_id") or x.get("id"),
                        "name": fn,
                        "arguments": args or "{}",
                    }
                )
            # Back-compat for nested tool_calls shapes
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

    walk(data.get("output", []))
    return out


def output_text(resp) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t
    try:
        data = json.loads(resp.model_dump_json())
        buf = []

        def walk(x):
            if isinstance(x, dict):
                if x.get("type") in ("output_text", "text"):
                    val = x.get("text") or x.get("content")
                    if isinstance(val, str):
                        buf.append(val)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(data.get("output", []))
        return "\n".join(buf).strip()
    except Exception:
        return ""


# --- Turn 1: let the model call the tool ------------------------------------
resp1 = client.responses.create(
    model="gpt-5",
    instructions=(
        "You’re a precise finance assistant. "
        "If currency conversion is requested, you MUST call get_fx_rate."
    ),
    tools=TOOLS,
    input=[
        {"role": "user", "content": "Convert 120 CHF to EUR and show your calculation."}
    ],
)

calls = extract_function_calls(resp1)
if not calls:
    print(output_text(resp1))
    raise SystemExit(0)

# --- Execute your tools ------------------------------------------------------
results_by_call_id = {}
for c in calls:
    args = json.loads(c["arguments"] or "{}")
    if c["name"] == "get_fx_rate":
        results_by_call_id[c["call_id"]] = get_fx_rate_impl(
            args.get("base", "CHF"), args.get("quote", "EUR")
        )
    else:
        results_by_call_id[c["call_id"]] = {
            "ok": False,
            "error": f"Unknown tool {c['name']}",
        }

# --- Turn 2: return tool output to the model --------------------------------
if hasattr(client.responses, "submit_tool_outputs"):
    # Path A: if your SDK has the helper (some versions do)
    resp2 = client.responses.submit_tool_outputs(
        response_id=resp1.id,
        tool_outputs=[
            {"tool_call_id": call_id, "output": json.dumps(result, ensure_ascii=False)}
            for call_id, result in results_by_call_id.items()
        ],
    )
else:
    # Path B: portable approach per docs — new create() with previous_response_id
    # Build function_call_output items (one per tool call)
    followup_input = [
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result, ensure_ascii=False),
        }
        for call_id, result in results_by_call_id.items()
    ]
    resp2 = client.responses.create(
        model="gpt-5",
        previous_response_id=resp1.id,
        input=followup_input,
    )

print(output_text(resp2) or resp2)
