import json
import unittest
from unittest.mock import Mock, patch

from poemai_utils.openai.ask_lean import AskLean


class TestAskLeanResponses(unittest.TestCase):
    """Test the Responses API integration in AskLean."""

    def setUp(self):
        self.api_key = "test-api-key"
        self.ask_lean = AskLean(openai_api_key=self.api_key, model="gpt-4o")

    def test_to_responses_api(self):
        """Test creating an AskResponses instance from AskLean."""
        responses_instance = self.ask_lean.to_responses_api()

        # Check that configuration is transferred
        self.assertEqual(responses_instance.openai_api_key, self.api_key)
        self.assertEqual(responses_instance.model, "gpt-4o")
        self.assertEqual(responses_instance.timeout, self.ask_lean.timeout)
        self.assertEqual(responses_instance.max_retries, self.ask_lean.max_retries)
        self.assertEqual(responses_instance.base_delay, self.ask_lean.base_delay)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_responses_api(self, mock_post):
        """Test using the Responses API through AskLean."""
        # Mock the Responses API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4o",
            "output_text": "This is a response from the Responses API.",
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            "finish_reason": "stop",
            "created": 1234567890,
        }
        mock_post.return_value = mock_response

        # Prepare traditional chat messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        # Call the migration method
        response = self.ask_lean.ask_with_responses_api(
            messages=messages, temperature=0.7, max_tokens=150
        )

        # Verify the request was made to the Responses API
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://api.openai.com/v1/responses")

        # Verify the request format conversion
        request_data = json.loads(kwargs["data"])
        self.assertEqual(request_data["model"], "gpt-4o")
        self.assertEqual(request_data["input"], "What is the capital of France?")
        self.assertEqual(request_data["instructions"], "You are a helpful assistant.")
        self.assertEqual(request_data["temperature"], 0.7)
        # max_tokens is no longer sent to the Responses API (by design - it's not supported)
        self.assertNotIn("max_tokens", request_data)

        # Verify the response is converted back to Chat Completions format
        self.assertEqual(
            response.choices[0].message.content,
            "This is a response from the Responses API.",
        )
        self.assertEqual(response.choices[0].message.role, "assistant")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(response.model, "gpt-4o")
        self.assertEqual(response.usage["total_tokens"], 25)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_responses_api_complex_messages(self, mock_post):
        """Test migration with complex message structures."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "Complex conversation response",
            "model": "gpt-4o",
        }
        mock_post.return_value = mock_response

        # Complex message structure with multiple exchanges
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
            {"role": "user", "content": "What's 2+2?"},
        ]

        response = self.ask_lean.ask_with_responses_api(messages=messages)

        # Verify the conversion handled multiple messages correctly
        request_data = json.loads(mock_post.call_args[1]["data"])

        expected_input = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
            {"role": "user", "content": "What's 2+2?"},
        ]

        self.assertEqual(request_data["input"], expected_input)
        self.assertEqual(request_data["instructions"], "You are a helpful assistant.")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_responses_api_with_tools(self, mock_post):
        """Test migration with tools/function calling."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "I'll use the calculator function.",
            "model": "gpt-4o",
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Calculate 15 * 7"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform basic calculations",
                },
            }
        ]

        response = self.ask_lean.ask_with_responses_api(
            messages=messages, tools=tools, tool_choice="auto"
        )

        # Verify tools were passed correctly
        request_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(request_data["tools"], tools)
        self.assertEqual(request_data["tool_choice"], "auto")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_responses_api_response_format(self, mock_post):
        """Test migration with response format specification."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": '{"answer": "Paris"}',
            "model": "gpt-4o",
        }
        mock_post.return_value = mock_response

        messages = [
            {
                "role": "user",
                "content": "What is the capital of France? Respond in JSON.",
            }
        ]
        response_format = {"type": "json_object"}

        response = self.ask_lean.ask_with_responses_api(
            messages=messages, response_format=response_format
        )

        # Verify response_format was passed correctly
        request_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(request_data["response_format"], response_format)
        self.assertEqual(response.choices[0].message.content, '{"answer": "Paris"}')

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_responses_api_fallback_response(self, mock_post):
        """Test fallback when response doesn't have output_text."""
        # Mock response without output_text
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "model": "gpt-4o",
            "custom_field": "some value",
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Test"}]
        response = self.ask_lean.ask_with_responses_api(messages=messages)

        # Should return the raw response if no output_text
        self.assertEqual(response.custom_field, "some value")
        self.assertEqual(response.model, "gpt-4o")


if __name__ == "__main__":
    unittest.main()
