import json
from unittest.mock import MagicMock, patch

import pytest
from poemai_utils.ai_model import AIApiType
from poemai_utils.openai.ask import Ask
from poemai_utils.openai.ask_responses import AskResponses
from poemai_utils.openai.openai_model import OPENAI_MODEL

GPT_5_6_MODELS = (
    OPENAI_MODEL.GPT_5_6,
    OPENAI_MODEL.GPT_5_6_LUNA,
    OPENAI_MODEL.GPT_5_6_TERRA,
    OPENAI_MODEL.GPT_5_6_SOL,
)


@pytest.mark.parametrize("model", GPT_5_6_MODELS)
def test_gpt_5_6_models_are_registered_with_capabilities(model):
    resolved = OPENAI_MODEL.by_model_key(model.model_key)

    assert resolved is model
    assert OPENAI_MODEL.by_model_key(model.calc_model_key()) is model
    assert set(model.api_types) == {
        AIApiType.CHAT_COMPLETIONS,
        AIApiType.RESPONSES,
    }
    assert model.supports_vision is True
    assert model.supports_reasoning is True
    assert model.supports_temperature is False
    assert model.requires_max_completion_tokens is True


@pytest.mark.parametrize("model", GPT_5_6_MODELS)
def test_chat_completions_for_gpt_5_6_forwards_core_controls_without_temperature(
    model,
):
    with patch("openai.OpenAI") as openai_class:
        client = MagicMock()
        openai_class.return_value = client
        response = MagicMock()
        response.choices[0].message.content = "GPT56_OK"
        client.chat.completions.create.return_value = response

        answer = Ask(model=model, openai_api_key="test-key").ask(
            "Reply with GPT56_OK.",
            reasoning_effort="none",
            verbosity="low",
        )

    assert answer == "GPT56_OK"
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == model.model_key
    assert request["reasoning_effort"] == "none"
    assert request["verbosity"] == "low"
    assert request["max_completion_tokens"] == 600
    assert "temperature" not in request


def test_chat_completions_named_controls_override_raw_controls(caplog):
    with patch("openai.OpenAI") as openai_class:
        client = MagicMock()
        openai_class.return_value = client
        response = MagicMock()
        response.choices[0].message.content = "GPT56_OK"
        client.chat.completions.create.return_value = response

        Ask(model=OPENAI_MODEL.GPT_5_6_SOL, openai_api_key="test-key").ask(
            "Reply with GPT56_OK.",
            additional_args={"reasoning_effort": "high", "verbosity": "high"},
            reasoning_effort="none",
            verbosity="low",
        )

    request = client.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "none"
    assert request["verbosity"] == "low"
    assert "overrides" in caplog.text


def test_responses_merges_text_config_and_forwards_core_controls(caplog):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "resp_gpt56",
        "model": "gpt-5.6-sol",
        "output_text": "GPT56_OK",
    }

    with patch(
        "poemai_utils.openai.ask_responses.requests.post", return_value=response
    ) as post:
        result = AskResponses(
            openai_api_key="test-key", model=OPENAI_MODEL.GPT_5_6_SOL
        ).ask(
            input="Reply with GPT56_OK.",
            response_format={"type": "text"},
            additional_args={"text": {"verbosity": "high", "custom": True}},
            reasoning={"effort": "high", "summary": "auto"},
            reasoning_effort="none",
            verbosity="low",
        )

    request = json.loads(post.call_args.kwargs["data"])
    assert request["model"] == "gpt-5.6-sol"
    assert request["reasoning"] == {"effort": "none", "summary": "auto"}
    assert request["text"] == {
        "format": {"type": "text"},
        "verbosity": "low",
        "custom": True,
    }
    assert "temperature" not in request
    assert result.output_text == "GPT56_OK"
    assert "overrides" in caplog.text


@pytest.mark.parametrize(
    ("argument", "value"),
    [("reasoning_effort", "minimal"), ("verbosity", "verbose")],
)
def test_invalid_gpt_5_6_core_controls_are_rejected(argument, value):
    ask = AskResponses(openai_api_key="test-key", model="gpt-5.6-sol")

    with pytest.raises(ValueError):
        ask.ask(input="test", **{argument: value})
