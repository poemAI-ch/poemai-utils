import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from poemai_utils.openai.ask import Ask

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
    ask = Ask(openai=openai, base_url=BASE_URL)
    ask.async_openai.httpx = httpx_mock

    # Check if the base_url is correctly overridden
    assert ask.base_url == BASE_URL
    assert ask.async_openai.base_url == BASE_URL

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
    assert async_client_mock.stream.call_args[0][1] == BASE_URL
