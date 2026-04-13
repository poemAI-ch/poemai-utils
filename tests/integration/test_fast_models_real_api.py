import logging
import os
import time

import pytest
from poemai_utils.openai.ask import Ask

_logger = logging.getLogger(__name__)


MODEL_CANDIDATES = [
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-mini",
    "gpt-5-mini-2025-08-07",
    "gpt-5-mini",
    "gpt-5.4-mini-2026-03-17",
]


@pytest.fixture(scope="module", autouse=True)
def api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("Set OPENAI_API_KEY to run OpenAI integration tests.")
    return api_key


@pytest.mark.integration
@pytest.mark.external
def test_fast_models(api_key: str):
    prompts = [
        "Reply with one short word: pong",
        "What is 2 + 2? Reply with only the number.",
    ]
    failures = {}
    max_token_candidates = [64, 256, 1024]

    for model_key in MODEL_CANDIDATES:
        model = Ask.OPENAI_MODEL.by_model_key(model_key)
        ask = Ask(openai_api_key=api_key, model=model)

        _logger.info("Testing model %s", model.model_key)
        started = time.time()

        last_output = ""
        last_exception = None

        for prompt in prompts:
            for max_tokens in max_token_candidates:
                try:
                    response = ask.ask(
                        prompt=prompt,
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                    last_output = (response or "").strip()
                except Exception as exc:
                    last_exception = exc
                    error_text = str(exc)
                    if "max_tokens or model output limit was reached" in error_text:
                        _logger.info(
                            "Model %s hit output limit for prompt %r with max_tokens=%s; retrying with higher limit",
                            model.model_key,
                            prompt,
                            max_tokens,
                        )
                        continue

                    _logger.warning(
                        "Model %s failed for prompt %r with max_tokens=%s: %s",
                        model.model_key,
                        prompt,
                        max_tokens,
                        exc,
                        exc_info=True,
                    )
                    break

                duration_ms = int((time.time() - started) * 1000)
                _logger.info(
                    "Model %s responded in %sms with output: %r",
                    model.model_key,
                    duration_ms,
                    last_output,
                )

                if last_output:
                    break

            if last_output:
                break

        if not last_output:
            if last_exception:
                failures[model.model_key] = f"request failed: {last_exception}"
            else:
                failures[model.model_key] = "empty output for all simple prompts"

    if failures:
        details = "; ".join(f"{model}: {reason}" for model, reason in failures.items())
        pytest.fail(f"One or more fast model checks failed: {details}")
