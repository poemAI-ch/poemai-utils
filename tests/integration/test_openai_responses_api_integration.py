import json
import logging
import os
import time
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


def _collect_stream_text(stream, *, model: str, chunk_label: str) -> str:
    final_text_parts: list[str] = []

    for idx, chunk in enumerate(stream):
        chunk_dict = chunk.model_dump()
        _logger.info(
            "%s chunk %s from %s: %s",
            chunk_label,
            idx,
            model,
            json.dumps(chunk_dict, indent=2, ensure_ascii=False),
        )

        delta_text = chunk_dict.get("delta")
        if isinstance(delta_text, str) and delta_text:
            final_text_parts.append(delta_text)
            continue

        output_text = getattr(chunk, "output_text", None)
        if isinstance(output_text, str) and output_text:
            final_text_parts.append(output_text)
            continue

        direct_text = chunk_dict.get("text")
        if isinstance(direct_text, str) and direct_text:
            if not final_text_parts:
                final_text_parts.append(direct_text)
            continue

        response_payload = chunk_dict.get("response") or {}
        if response_payload:
            for item in response_payload.get("output", []) or []:
                for content in item.get("content", []) or []:
                    text_value = content.get("text")
                    if isinstance(text_value, str) and text_value:
                        if not final_text_parts:
                            final_text_parts.append(text_value)

    final_text = "".join(final_text_parts).strip()
    return final_text


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
def test_openai_responses_streaming_completion_primes(api_key: str):
    prompt = (
        "List the first five prime numbers as separate bullet points prefixed with STREAM, "
        "and end with STREAMING-OK."
    )

    def _runner(ask: AskResponses, model: str) -> None:
        ask_arguments = {
            "input_data": prompt,
            "instructions": (
                "Respond using bullet points. Each line should begin with STREAM followed by the "
                "prime number, and finish with a final line STREAMING-OK."
            ),
            "temperature": 0,
            "stream": True,
        }

        _logger.info(
            "Starting long streaming call to %s with arguments: %s",
            model,
            json.dumps(ask_arguments, indent=2, ensure_ascii=False),
        )

        stream = ask.ask(**ask_arguments)
        final_text = _collect_stream_text(
            stream,
            model=model,
            chunk_label="Long stream",
        )

        _logger.info("Final prime streamed text from %s: %s", model, final_text)

        assert final_text, "Expected non-empty streamed output"
        normalized = final_text.upper()
        assert normalized.count("STREAM") >= 5, (
            "Expected at least five STREAM bullet markers, got: " + final_text
        )
        assert "STREAMING-OK" in normalized, (
            "Expected long streamed output to include STREAMING-OK, got: " + final_text
        )

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


@pytest.mark.integration
@pytest.mark.external
def test_openai_responses_streaming_completion(api_key: str):
    prompt = "Reply with STREAMING-OK and nothing else."

    def _runner(ask: AskResponses, model: str) -> None:
        ask_arguments = {
            "input_data": prompt,
            "instructions": "You are a precise assistant. Follow the user instructions exactly.",
            "temperature": 0,
            "stream": True,
        }

        _logger.info(
            "Starting streaming call to %s with arguments: %s",
            model,
            json.dumps(ask_arguments, indent=2, ensure_ascii=False),
        )

        stream = ask.ask(**ask_arguments)
        final_text = _collect_stream_text(
            stream,
            model=model,
            chunk_label="Stream",
        )

        _logger.info("Final streamed text from %s: %s", model, final_text)

        assert final_text, "Expected non-empty streamed output"
        assert "STREAMING-OK" in final_text.upper(), (
            "Expected streamed output to include STREAMING-OK, got: " + final_text
        )

    _execute_with_models(MODEL_CANDIDATES, _runner, api_key)


@pytest.mark.integration
@pytest.mark.external
def test_openai_responses_reasoning_effort_levels(api_key: str):
    prompt = (
        "Brain teaser: A snail climbs a 25-meter pole. Each day it climbs 4 meters during the "
        "day but slips down 1 meter at night. How many days will it take the snail to reach the top? "
        "Show structured reasoning, then end with a final line in the exact format 'Answer: <number> days'."
    )
    instructions = (
        "Explain your reasoning clearly. After the reasoning, place the final result on its own line "
        "formatted exactly as 'Answer: <number> days'."
    )

    def _runner(ask: AskResponses, model: str) -> None:
        efforts = ["minimal", "low", "high"]
        outputs = {}
        durations = {}

        for effort in efforts:
            _logger.info(
                "Starting reasoning effort '%s' for model %s",
                effort,
                model,
            )
            start = time.perf_counter()
            response = ask.ask(
                input_data=prompt,
                instructions=instructions,
                reasoning={"effort": effort},
                store=False,
            )
            duration = time.perf_counter() - start
            durations[effort] = duration

            output_text = getattr(response, "output_text", "").strip()
            outputs[effort] = output_text

            _logger.info(
                "Reasoning effort '%s' response (%.3fs): %s",
                effort,
                duration,
                output_text,
            )

            if getattr(response, "reasoning", None):
                reasoning_meta = (
                    response.reasoning.to_dict()
                    if hasattr(response.reasoning, "to_dict")
                    else str(response.reasoning)
                )
                _logger.info(
                    "Reasoning effort '%s' metadata: %s",
                    effort,
                    reasoning_meta,
                )

            assert output_text, f"Expected non-empty output for effort '{effort}'"
            assert (
                "Answer:" in output_text
            ), f"Expected final answer marker in output for effort '{effort}', got: {output_text}"

        for effort, text in outputs.items():
            _logger.info(
                "Collected output for %s effort (%.3fs): %s",
                effort,
                durations[effort],
                text,
            )

        _logger.info(
            "Completed reasoning effort sweep for model %s",
            model,
        )

    _execute_with_models(REASONING_MODEL_CANDIDATES, _runner, api_key)


@pytest.mark.integration
@pytest.mark.external
def test_openai_responses_max_output_tokens(api_key: str):
    prompt = "Write a short haiku about unit testing in software engineering."

    def _runner(ask: AskResponses, model: str) -> None:
        max_output_tokens = 40
        _logger.info(
            "Calling model %s with max_output_tokens=%s",
            model,
            max_output_tokens,
        )
        start = time.perf_counter()
        response = ask.ask(
            input_data=prompt,
            instructions="Keep the poem concise and respectful of the token limit.",
            max_output_tokens=max_output_tokens,
            temperature=0.2,
        )
        duration = time.perf_counter() - start

        output_text = getattr(response, "output_text", "").strip()
        _logger.info(
            "Model %s returned in %.3fs with output: %s",
            model,
            duration,
            output_text,
        )

        usage = getattr(response, "usage", None)
        if usage:
            output_tokens = usage.get("output_tokens")
            _logger.info(
                "Model %s usage stats: %s",
                model,
                usage,
            )
            if output_tokens is not None:
                assert (
                    output_tokens <= max_output_tokens
                ), f"Expected output tokens <= {max_output_tokens}, got {output_tokens}"

        assert output_text, "Expected non-empty output when using max_output_tokens"

    _execute_with_models(MODEL_CANDIDATES, _runner, api_key)


@pytest.mark.integration
@pytest.mark.external
def test_openai_responses_conversation_manager_comparison(api_key: str):
    instructions = "You are a succinct assistant focused on integration testing. Follow user prompts exactly."
    first_prompt = "Turn 1: Provide a two-sentence status update about the OpenAI Responses integration tests."
    second_prompt = "Turn 2: Summarize your previous reply in exactly three uppercase words separated by spaces."

    def _runner(ask: AskResponses, model: str) -> None:
        scenarios = (
            ("manual", False),
            ("conversation_manager", True),
        )

        durations = {}

        for label, use_manager in scenarios:
            _logger.info(
                "Starting %s conversation run with model %s",
                label,
                model,
            )
            start = time.perf_counter()

            if use_manager:
                manager = ask.start_conversation()
                first_response = manager.send(
                    input_data={"role": "user", "content": first_prompt},
                    instructions=instructions,
                    store=True,
                )
                second_response = manager.send(
                    input_data={"role": "user", "content": second_prompt},
                    store=True,
                )
            else:
                first_response = ask.ask(
                    input_data={"role": "user", "content": first_prompt},
                    instructions=instructions,
                    store=True,
                )
                second_response = ask.ask(
                    input_data={"role": "user", "content": second_prompt},
                    previous_response_id=first_response.id,
                    store=True,
                )

            duration = time.perf_counter() - start
            durations[label] = duration

            first_text = getattr(first_response, "output_text", "").strip()
            second_text = getattr(second_response, "output_text", "").strip()

            _logger.info(
                "%s mode (model %s) first response: %s",
                label,
                model,
                first_text,
            )
            _logger.info(
                "%s mode (model %s) second response: %s",
                label,
                model,
                second_text,
            )
            _logger.info(
                "%s mode (model %s) conversation duration: %.3fs",
                label,
                model,
                duration,
            )
            _logger.info(
                "%s mode (model %s) run completed in %.3fs",
                label,
                model,
                duration,
            )

            assert first_text, f"{label} mode produced empty first response"
            assert second_text, f"{label} mode produced empty second response"
            words = [word for word in second_text.split() if word]
            cleaned = ["".join(ch for ch in word if ch.isalpha()) for word in words]
            cleaned = [word for word in cleaned if word]
            assert (
                len(cleaned) >= 3
            ), f"{label} mode summary should contain at least three words, got: {second_text}"
            assert all(
                word.isupper() for word in cleaned[:3]
            ), f"{label} mode summary words should be uppercase, got: {second_text}"

        _logger.info(
            "Timing comparison for model %s: manual=%.3fs conversation_manager=%.3fs",
            model,
            durations["manual"],
            durations["conversation_manager"],
        )
        _logger.info(
            "Completed conversation manager comparison for model %s",
            model,
        )

    _execute_with_models(MODEL_CANDIDATES, _runner, api_key)
