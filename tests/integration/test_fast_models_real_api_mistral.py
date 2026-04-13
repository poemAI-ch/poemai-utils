import json
import logging
import os
import time

import pytest
from poemai_utils.openai.ask_lean import AskLean

_logger = logging.getLogger(__name__)


MISTRAL_CHAT_COMPLETIONS_URL = os.getenv(
    "MISTRAL_CHAT_COMPLETIONS_URL", "https://api.mistral.ai/v1/chat/completions"
)

MODEL_CANDIDATES = [
    "mistral-large-latest",
    "mistral-small-latest",
    "ministral-8b-latest",
]


@pytest.fixture(scope="module", autouse=True)
def mistral_api_key():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise Exception(
            "Set MISTRAL_API_KEY to run Mistral integration tests. "
            'Example: export MISTRAL_API_KEY="$(cat ~/mistral_api_key.txt )"'
        )
    return api_key


def _extract_content(response) -> str:
    try:
        content = response.choices[0].message.content
        if content is None:
            return ""
        return str(content).strip()
    except Exception:
        return ""


def _extract_tool_calls(response):
    try:
        raw_calls = response.choices[0].message.tool_calls
    except Exception:
        return []

    calls = []
    for call in raw_calls or []:
        if hasattr(call, "to_dict"):
            call = call.to_dict()
        elif not isinstance(call, dict):
            continue

        function_payload = call.get("function") or {}
        name = function_payload.get("name")
        arguments_raw = function_payload.get("arguments", "{}")
        try:
            arguments = (
                arguments_raw
                if isinstance(arguments_raw, dict)
                else json.loads(arguments_raw or "{}")
            )
        except Exception:
            arguments = {}

        calls.append(
            {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "name": name,
                "arguments": arguments,
            }
        )
    return calls


@pytest.mark.integration
@pytest.mark.external
def test_mistral_api_works_with_fast_model_candidates(mistral_api_key: str):
    prompts = [
        "Reply with one short word: pong",
        "What is 2 + 2? Reply with only the number.",
    ]
    max_token_candidates = [64, 256, 1024]
    failures = {}

    for model_key in MODEL_CANDIDATES:
        ask = AskLean(
            openai_api_key=mistral_api_key,
            model=model_key,
            base_url=MISTRAL_CHAT_COMPLETIONS_URL,
            timeout=60,
            max_retries=2,
        )

        _logger.info("Testing Mistral model %s", model_key)
        started = time.time()
        last_output = ""
        last_exception = None

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            for max_tokens in max_token_candidates:
                try:
                    response = ask.ask(
                        messages=messages,
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                    last_output = _extract_content(response)
                except Exception as exc:
                    last_exception = exc
                    _logger.warning(
                        "Mistral model %s failed for prompt %r with max_tokens=%s: %s",
                        model_key,
                        prompt,
                        max_tokens,
                        exc,
                        exc_info=True,
                    )
                    break

                duration_ms = int((time.time() - started) * 1000)
                _logger.info(
                    "Mistral model %s responded in %sms with output: %r",
                    model_key,
                    duration_ms,
                    last_output,
                )
                if last_output:
                    break

            if last_output:
                break

        if last_output:
            _logger.info(
                "Mistral API check succeeded with model %s on endpoint %s",
                model_key,
                MISTRAL_CHAT_COMPLETIONS_URL,
            )
            return

        if last_exception:
            failures[model_key] = f"request failed: {last_exception}"
        else:
            failures[model_key] = "empty output for all simple prompts"

    details = "; ".join(f"{model}: {reason}" for model, reason in failures.items())
    pytest.fail(
        "Mistral API check failed for all candidate models "
        f"at {MISTRAL_CHAT_COMPLETIONS_URL}: {details}"
    )


@pytest.mark.integration
@pytest.mark.external
def test_mistral_chat_completions_tool_call_round_trip(mistral_api_key: str):
    tools = [
        {
            "type": "function",
            "function": {
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
        key = (base.upper(), quote.upper())
        rate = stub.get(key)
        if rate is None:
            return {"ok": False, "error": f"FX rate {base}->{quote} not found"}
        return {"ok": True, "rate": rate, "base": key[0], "quote": key[1]}

    failures = {}

    for model_key in MODEL_CANDIDATES:
        ask = AskLean(
            openai_api_key=mistral_api_key,
            model=model_key,
            base_url=MISTRAL_CHAT_COMPLETIONS_URL,
            timeout=60,
            max_retries=2,
        )
        try:
            first_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise finance assistant. "
                        "If currency conversion is requested, you MUST call get_fx_rate "
                        "and then compute the result."
                    ),
                },
                {
                    "role": "user",
                    "content": "Convert 120 CHF to EUR and show your calculation.",
                },
            ]

            first_response = ask.ask(
                messages=first_messages,
                temperature=0,
                tools=tools,
                tool_choice="auto",
            )
            _logger.info(
                "Mistral first tool-call response for %s: %s",
                model_key,
                json.dumps(first_response.to_dict(), indent=2, ensure_ascii=False),
            )

            tool_calls = _extract_tool_calls(first_response)
            assert tool_calls, f"Model {model_key} did not return tool calls."

            first_call = tool_calls[0]
            assert first_call["name"] == "get_fx_rate"
            arguments = first_call["arguments"]
            assert arguments.get("base") == "CHF"
            assert arguments.get("quote") == "EUR"

            tool_result = get_fx_rate_impl(arguments["base"], arguments["quote"])

            assistant_message = first_response.choices[0].message
            if hasattr(assistant_message, "to_dict"):
                assistant_message = assistant_message.to_dict()

            second_messages = list(first_messages)
            second_messages.append(assistant_message)
            second_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": first_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

            second_response = ask.ask(
                messages=second_messages,
                temperature=0,
                tools=tools,
                tool_choice="auto",
            )
            _logger.info(
                "Mistral second tool-call response for %s: %s",
                model_key,
                json.dumps(second_response.to_dict(), indent=2, ensure_ascii=False),
            )

            second_output = _extract_content(second_response)
            assert (
                second_output
            ), f"Model {model_key} returned empty post-tool response."
            normalized = second_output.replace(",", ".")
            assert "124.8" in normalized or "124" in normalized, (
                f"Model {model_key} returned unexpected conversion output: "
                f"{second_output}"
            )
            return
        except Exception as exc:
            _logger.warning(
                "Mistral tool-call round trip failed for model %s: %s",
                model_key,
                exc,
                exc_info=True,
            )
            failures[model_key] = str(exc)

    details = "; ".join(f"{model}: {reason}" for model, reason in failures.items())
    pytest.fail(
        "Mistral chat completions tool-call test failed for all model candidates: "
        + details
    )
