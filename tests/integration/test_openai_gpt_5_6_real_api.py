import logging
import os
import time

import pytest
from poemai_utils.openai.ask import Ask
from poemai_utils.openai.ask_responses import AskResponses
from poemai_utils.openai.openai_model import OPENAI_MODEL, OPENAI_TEXT_VERBOSITY_LEVELS

_logger = logging.getLogger(__name__)


GPT_5_6_MODELS = (
    OPENAI_MODEL.GPT_5_6_LUNA,
    OPENAI_MODEL.GPT_5_6_TERRA,
    OPENAI_MODEL.GPT_5_6_SOL,
)


@pytest.fixture(scope="module")
def api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Set OPENAI_API_KEY to run GPT-5.6 integration tests.")
    return api_key


@pytest.mark.integration
@pytest.mark.external
@pytest.mark.parametrize("model", GPT_5_6_MODELS)
def test_openai_gpt_5_6_models_real_api(model, api_key):
    assert "low" in OPENAI_TEXT_VERBOSITY_LEVELS

    prompt = "Reply with exactly GPT56_OK and nothing else."

    chat_started = time.perf_counter()
    chat_response = Ask(model=model, openai_api_key=api_key).ask(
        prompt,
        max_tokens=32,
        reasoning_effort="none",
        verbosity="low",
    )
    chat_duration = time.perf_counter() - chat_started
    _logger.info(
        "GPT-5.6 Chat Completions model=%s duration=%.3fs response_id=%s usage=%s output=%r",
        model.model_key,
        chat_duration,
        None,
        None,
        chat_response,
    )
    assert "GPT56_OK" in (chat_response or "")

    responses_started = time.perf_counter()
    responses_response = AskResponses(openai_api_key=api_key, model=model).ask(
        input=prompt,
        max_output_tokens=32,
        reasoning_effort="none",
        verbosity="low",
        store=False,
    )
    responses_duration = time.perf_counter() - responses_started
    responses_output = getattr(responses_response, "output_text", "")
    _logger.info(
        "GPT-5.6 Responses model=%s duration=%.3fs response_id=%s usage=%s output=%r",
        model.model_key,
        responses_duration,
        getattr(responses_response, "id", None),
        getattr(responses_response, "usage", None),
        responses_output,
    )
    assert "GPT56_OK" in responses_output
