import json
from unittest.mock import MagicMock, patch

import pytest
from poemai_utils.openai.ask_lean import AskLean


@pytest.fixture
def ask_lean_client():
    return AskLean(openai_api_key="fake_api_key")


def test_ask_success(ask_lean_client):
    """Test a successful API call with no retries needed."""
    messages = [{"role": "user", "content": "Hello"}]
    mock_response = {
        "id": "123",
        "object": "chat.completion",
        "choices": [{"message": {"content": "Hi!"}}],
    }

    with patch("requests.post") as mock_post:
        mock_requests_response = MagicMock()
        mock_requests_response.status_code = 200
        mock_requests_response.json.return_value = mock_response
        mock_post.return_value = mock_requests_response

        response = ask_lean_client.ask(messages=messages)
        assert (
            response.dict() == mock_response
        ), "Response should match the mocked return value"

        # Check the requests.post call arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer fake_api_key"
        data_sent = json.loads(kwargs["data"])
        assert data_sent["messages"] == messages
        assert data_sent["model"] == "gpt-4"


def test_ask_with_properties(ask_lean_client):
    """Test a successful API call with no retries needed."""
    messages = [{"role": "user", "content": "Hello"}]
    mock_response = {
        "id": "123",
        "object": "chat.completion",
        "choices": [{"message": {"content": "Hi!"}}],
    }

    with patch("requests.post") as mock_post:
        mock_requests_response = MagicMock()
        mock_requests_response.status_code = 200
        mock_requests_response.json.return_value = mock_response
        mock_post.return_value = mock_requests_response

        response = ask_lean_client.ask(messages=messages)
        assert (
            response.dict() == mock_response
        ), "Response should match the mocked return value"

        # Check the requests.post call arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer fake_api_key"
        data_sent = json.loads(kwargs["data"])
        assert data_sent["messages"] == messages
        assert data_sent["model"] == "gpt-4"

        assert response.choices[0].message.content == "Hi!"


def test_ask_with_retry(ask_lean_client):
    """Test that the class retries on server errors and eventually succeeds."""
    messages = [{"role": "user", "content": "Hello"}]
    mock_response = {
        "id": "123",
        "object": "chat.completion",
        "choices": [{"message": {"content": "Hi!"}}],
    }

    with patch("requests.post") as mock_post:
        # First call -> 500 error, second call -> 200 success
        first_attempt = MagicMock()
        first_attempt.status_code = 500
        first_attempt.text = "Server Error"

        second_attempt = MagicMock()
        second_attempt.status_code = 200
        second_attempt.json.return_value = mock_response

        mock_post.side_effect = [first_attempt, second_attempt]

        response = ask_lean_client.ask(messages=messages)
        assert response.dict() == mock_response

        assert (
            mock_post.call_count == 2
        ), "Should have retried once after first 500 error"


def test_ask_all_retries_fail(ask_lean_client):
    """Test that a RuntimeError is raised if all retries fail."""
    messages = [{"role": "user", "content": "Hello"}]

    with patch("requests.post") as mock_post:
        # Simulate that all attempts return a 500 error
        mock_attempt = MagicMock()
        mock_attempt.status_code = 500
        mock_attempt.text = "Server Error"
        mock_post.side_effect = [mock_attempt, mock_attempt, mock_attempt]

        with pytest.raises(RuntimeError) as exc_info:
            ask_lean_client.ask(messages=messages)
        assert "OpenAI API call failed" in str(exc_info.value)


def test_ask_with_response_format(ask_lean_client):
    """Test that the response_format is included in the request when specified."""
    messages = [{"role": "user", "content": "Solve 8x + 7 = -23"}]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "math_reasoning",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string"},
                                "output": {"type": "string"},
                            },
                            "required": ["explanation", "output"],
                            "additionalProperties": False,
                        },
                    },
                    "final_answer": {"type": "string"},
                },
                "required": ["steps", "final_answer"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    mock_response = {"id": "123", "choices": []}

    with patch("requests.post") as mock_post:
        mock_requests_response = MagicMock()
        mock_requests_response.status_code = 200
        mock_requests_response.json.return_value = mock_response
        mock_post.return_value = mock_requests_response

        ask_lean_client.ask(messages=messages, response_format=response_format)

        # Check if response_format is included
        args, kwargs = mock_post.call_args
        data_sent = json.loads(kwargs["data"])
        assert "response_format" in data_sent
        assert data_sent["response_format"] == response_format
