import json
import logging
import os
from typing import Callable, Iterable, Optional

import pytest
from poemai_utils.openai.ask_responses import AskResponses

_logger = logging.getLogger(__name__)


MODEL_CANDIDATES = [
    "gpt-5",
]

REASONING_MODEL_CANDIDATES = [
    "gpt-5",
]


@pytest.fixture(scope="module", autouse=True)
def api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("Set OPENAI_API_KEY to run OpenAI integration tests.")
    return api_key


def _execute_with_models(
    candidates: Iterable[Optional[str]],
    runner: Callable[[AskResponses, str], None],
    api_key: str,
) -> str:

    errors = {}
    for candidate in candidates:
        if not candidate:
            continue

        ask = AskResponses(openai_api_key=api_key, model=candidate)
        try:
            runner(ask, candidate)
            return candidate
        except (RuntimeError, AssertionError) as exc:
            _logger.warning(f"Model {candidate} failed: {exc}", exc_info=True)
            errors[candidate] = str(exc)[:200]

    formatted = "; ".join(f"{model}: {message}" for model, message in errors.items())
    pytest.fail(
        "OpenAI Responses integration failed for all candidate models: "
        + (formatted or "no models attempted")
    )


@pytest.mark.integration
@pytest.mark.external
def test_openai_responses_tool_call_round_trip(api_key: str):
    tools = [
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
        key = (base.upper(), quote.upper())
        rate = stub.get(key)
        if rate is None:
            return {"ok": False, "error": f"FX rate {base}->{quote} not found"}
        return {"ok": True, "rate": rate, "base": key[0], "quote": key[1]}

    def _runner(ask: AskResponses, model: str) -> None:

        ask_arguments = {
            "input_data": {
                "role": "user",
                "content": "Convert 120 CHF to EUR and show your calculation.",
            },
            "instructions": (
                "You’re a precise finance assistant. If currency conversion is requested, "
                "you MUST call get_fx_rate and then compute the result."
            ),
            "tools": tools,
            "temperature": 1,
        }

        _logger.info(
            f"Calling model {model} with arguments: {json.dumps(ask_arguments, indent=2, ensure_ascii=False)}"
        )
        first_llm_response = ask.ask(**ask_arguments)

        _logger.info(
            f"First LLM response:\n{json.dumps(first_llm_response.model_dump(), indent=2, ensure_ascii=False)}\n****************\n"
        )

        tool_calls = AskResponses.extract_tool_calls(first_llm_response)
        _logger.info(
            f"Extracted tool calls: {json.dumps([call.model_dump() for call in tool_calls], indent=2, ensure_ascii=False)}"
        )
        assert tool_calls, f"Model {model} did not produce a tool call."

        call = tool_calls[0]
        assert call.name == "get_fx_rate"
        arguments = json.loads(call.arguments)
        assert arguments.get("base") == "CHF"
        assert arguments.get("quote") == "EUR"

        tool_result = get_fx_rate_impl(arguments["base"], arguments["quote"])

        second_llm_arguments = {
            "input_data": [
                {
                    "type": "function_call_output",
                    "call_id": getattr(call, "call_id", getattr(call, "id", None)),
                    "output": json.dumps(tool_result, ensure_ascii=False),
                }
            ],
            "previous_response_id": first_llm_response.id,
        }

        _logger.info(
            f"Calling model {model} with second arguments:{second_llm_arguments}"
        )
        second_llm_response = ask.ask(**second_llm_arguments)
        _logger.info(
            f"Second LLM response:\n {json.dumps(second_llm_response.model_dump(), indent=2, ensure_ascii=False)}\n****************\n"
        )

        assert "124.80" in second_llm_response.output_text

        assert "124.80" in second_llm_response.output_text

    _execute_with_models(MODEL_CANDIDATES, _runner, api_key)


@pytest.mark.integration
@pytest.mark.external
def test_openai_responses_structured_output_and_reasoning(api_key: str):
    def _runner(ask: AskResponses, model: str) -> None:
        response = ask.ask(
            input_data="Add 17 and 25 and return the result in JSON format, including steps.",
            instructions="Return JSON with 'result' as an integer and 'steps' as an array of strings.",
            response_format={"type": "json_object"},
            reasoning={"effort": "medium"},
        )

        payload = json.loads(response.output_text)
        assert int(payload["result"]) == 42
        assert isinstance(payload.get("steps"), list)
        assert payload["steps"], "Expected non-empty steps list"

        reasoning = getattr(response, "reasoning", None)
        if reasoning and getattr(reasoning, "content", None):
            content = reasoning.content
            assert any(
                isinstance(entry, dict) and entry.get("type") == "text"
                for entry in content
            ), "Expected textual reasoning entries"

    _execute_with_models(REASONING_MODEL_CANDIDATES, _runner, api_key)
