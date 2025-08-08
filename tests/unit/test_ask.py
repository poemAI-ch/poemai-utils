import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from poemai_utils.openai.ask import Ask
from poemai_utils.openai.openai_model import OPENAI_MODEL

_logger = logging.getLogger(__name__)


def test_count_tokens():
    ask = Ask()
    assert ask.count_tokens("hello world") == 2


@pytest.mark.asyncio
async def test_async_ask():
    # Mock the HTTPX client and its context manager behavior
    async_client_mock = MagicMock()
    # Properly simulate entering the context manager
    async_client_mock.__aenter__.return_value = async_client_mock
    # Simulate exiting the context manager
    async_client_mock.__aexit__.return_value = None

    # Create a mock for the response object that will be used in the context manager
    response_mock = AsyncMock()
    response_mock.__aenter__.return_value = (
        response_mock  # Return itself when entering the context
    )
    response_mock.__aexit__.return_value = None  # Simple exit, no exception handling

    response_mock.status_code = 200

    # Create async iterator function

    return_texts = ["chunk1", "chunk2", "chunk3"]
    return_chunks = []
    for text in return_texts:
        message_text = json.dumps({"choices": [{"delta": {"content": text}}]})
        chunk = f"data:{message_text}\n\n"
        return_chunks.append(chunk)

    async def async_iter():
        for chunk in return_chunks:
            yield chunk

    response_mock.aiter_text = lambda: async_iter()

    got_one = False
    async for chunk_text in response_mock.aiter_text():
        got_one = True

    assert got_one

    # Mock the entire AsyncClient class to return our client mock
    httpx_mock = MagicMock()

    httpx_mock.AsyncClient.return_value = async_client_mock

    async_client_mock.stream = MagicMock(return_value=response_mock)

    # Set the mocked httpx as the httpx_override in AsyncOpenai
    openai = MagicMock()
    BASE_URL = "https://example.com"
    FULL_BASE_URL = f"{BASE_URL}/v1/chat/completions"
    ask = Ask(openai=openai, base_url=BASE_URL)
    ask.async_openai.httpx = httpx_mock

    # Check if the base_url is correctly overridden
    assert ask.base_url == BASE_URL
    assert ask.async_openai.base_url == FULL_BASE_URL

    # Collect results from the async generator
    results = []
    async for response in ask.ask_async("hello world"):
        results.append(response)
        _logger.info(f"response: {response}")

    # Assuming your implementation processes results as such
    assert results == [
        {"content": "chunk1"},
        {"content": "chunk2"},
        {"content": "chunk3"},
    ]

    assert async_client_mock.stream.call_count == 1

    assert async_client_mock.stream.call_args[0][0] == "POST"
    assert async_client_mock.stream.call_args[0][1] == FULL_BASE_URL


def test_temperature_handling_for_models_without_support():
    """Test that temperature parameter is skipped for models that don't support it."""
    # Mock the OpenAI client
    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the response from OpenAI API
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Paris is the capital of France."
        mock_client.chat.completions.create.return_value = mock_response

        # Test with a GPT-5 model that doesn't support temperature
        ask = Ask(model=OPENAI_MODEL.GPT_5, openai_api_key="test_key")

        # Call ask with temperature=0 (which should be ignored)
        response = ask.ask("What is the capital of France?", temperature=0)

        # Verify the response
        assert response == "Paris is the capital of France."

        # Verify that the OpenAI client was called without temperature parameter
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args

        # Check that temperature is NOT in the kwargs
        assert "temperature" not in call_args.kwargs

        # Verify model and messages are correct
        assert call_args.kwargs["model"] == "gpt-5"
        assert call_args.kwargs["messages"] == [
            {"role": "user", "content": "What is the capital of France?"}
        ]


def test_temperature_handling_for_models_with_support():
    """Test that temperature parameter is passed for models that support it."""
    # Mock the OpenAI client
    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the response from OpenAI API
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Paris is the capital of France."
        mock_client.chat.completions.create.return_value = mock_response

        # Test with a GPT-4o model that supports temperature
        ask = Ask(model=OPENAI_MODEL.GPT_4_o, openai_api_key="test_key")

        # Call ask with temperature=0.5
        response = ask.ask("What is the capital of France?", temperature=0.5)

        # Verify the response
        assert response == "Paris is the capital of France."

        # Verify that the OpenAI client was called with temperature parameter
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args

        # Check that temperature IS in the kwargs
        assert call_args.kwargs["temperature"] == 0.5

        # Verify model and messages are correct
        assert call_args.kwargs["model"] == "gpt-4o"
        assert call_args.kwargs["messages"] == [
            {"role": "user", "content": "What is the capital of France?"}
        ]


def test_temperature_handling_in_async_mode():
    """Test that temperature handling works correctly in async mode for models without support."""
    # Test with GPT-5 model
    with patch("openai.OpenAI"):
        ask = Ask(model=OPENAI_MODEL.GPT_5, openai_api_key="test_key")

        # Check that the model doesn't support temperature
        assert hasattr(ask.model, "supports_temperature")
        assert ask.model.supports_temperature == False

        # Check that the AsyncOpenai instance also has the correct model
        assert ask.async_openai.model == ask.model
        assert hasattr(ask.async_openai.model, "supports_temperature")
        assert ask.async_openai.model.supports_temperature == False
