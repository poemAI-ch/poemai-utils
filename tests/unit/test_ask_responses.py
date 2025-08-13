import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from poemai_utils.openai.ask_responses import AskResponses, PydanticLikeBox


class TestAskResponses(unittest.TestCase):
    def setUp(self):
        self.api_key = "test-api-key"
        self.ask_responses = AskResponses(
            openai_api_key=self.api_key, model="gpt-4o-mini"
        )

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_simple_text(self, mock_post):
        """Test simple text request to Responses API."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4o-mini",
            "output_text": "This is a test response.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_post.return_value = mock_response

        # Test simple text input
        response = self.ask_responses.ask(
            input_data="Hello, world!", instructions="Be helpful.", max_tokens=100
        )

        # Verify request was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args

        self.assertEqual(args[0], "https://api.openai.com/v1/responses")
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.api_key}")

        # Parse the request data
        request_data = json.loads(kwargs["data"])
        self.assertEqual(request_data["model"], "gpt-4o-mini")
        self.assertEqual(request_data["input"], "Hello, world!")
        self.assertEqual(request_data["instructions"], "Be helpful.")
        self.assertEqual(request_data["max_tokens"], 100)

        # Verify response
        self.assertIsInstance(response, PydanticLikeBox)
        self.assertEqual(response.output_text, "This is a test response.")
        self.assertEqual(response.model, "gpt-4o-mini")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_simple_method(self, mock_post):
        """Test the ask_simple convenience method."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output_text": "Simple response"}
        mock_post.return_value = mock_response

        result = self.ask_responses.ask_simple(
            prompt="Test prompt", instructions="Test instructions", temperature=0.5
        )

        self.assertEqual(result, "Simple response")

        # Verify request parameters
        args, kwargs = mock_post.call_args
        request_data = json.loads(kwargs["data"])
        self.assertEqual(request_data["input"], "Test prompt")
        self.assertEqual(request_data["instructions"], "Test instructions")
        self.assertEqual(request_data["temperature"], 0.5)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_vision(self, mock_post):
        """Test vision request with image."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "I can see an image of a cat."
        }
        mock_post.return_value = mock_response

        result = self.ask_responses.ask_vision(
            text="What's in this image?",
            image_url="https://example.com/cat.jpg",
            instructions="Describe the image",
        )

        self.assertEqual(result, "I can see an image of a cat.")

        # Verify the input structure for vision
        args, kwargs = mock_post.call_args
        request_data = json.loads(kwargs["data"])

        expected_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's in this image?"},
                    {"type": "input_image", "image_url": "https://example.com/cat.jpg"},
                ],
            }
        ]
        self.assertEqual(request_data["input"], expected_input)
        self.assertEqual(request_data["instructions"], "Describe the image")

    def test_convert_messages_to_input(self):
        """Test conversion from Chat Completions messages to Responses input."""
        # Test simple case with system and user message
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        instructions, input_data = AskResponses.convert_messages_to_input(messages)

        self.assertEqual(instructions, "You are helpful.")
        self.assertEqual(input_data, "Hello!")

        # Test multiple messages
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        instructions, input_data = AskResponses.convert_messages_to_input(messages)

        self.assertEqual(instructions, "You are helpful.")
        expected_input = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        self.assertEqual(input_data, expected_input)

        # Test multiple system messages
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello!"},
        ]

        instructions, input_data = AskResponses.convert_messages_to_input(messages)

        self.assertEqual(instructions, "You are helpful.\n\nBe concise.")
        self.assertEqual(input_data, "Hello!")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_retry_on_server_error(self, mock_post):
        """Test retry logic on server errors."""
        # Mock server error then success
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"

        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"output_text": "Success!"}

        mock_post.side_effect = [error_response, success_response]

        with patch("time.sleep"):  # Don't actually sleep in tests
            response = self.ask_responses.ask(input_data="Test")

        self.assertEqual(response.output_text, "Success!")
        self.assertEqual(mock_post.call_count, 2)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_failure_after_max_retries(self, mock_post):
        """Test failure after exhausting max retries."""
        # Mock consistent server errors
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"
        mock_post.return_value = error_response

        with patch("time.sleep"):  # Don't actually sleep in tests
            with self.assertRaises(RuntimeError) as context:
                self.ask_responses.ask(input_data="Test")

            self.assertIn("OpenAI Responses API call failed", str(context.exception))

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_client_error_no_retry(self, mock_post):
        """Test that client errors (4xx) don't trigger retries."""
        error_response = Mock()
        error_response.status_code = 400
        error_response.text = "Bad Request"
        mock_post.return_value = error_response

        with self.assertRaises(RuntimeError):
            self.ask_responses.ask(input_data="Test")

        # Should only be called once (no retries for client errors)
        self.assertEqual(mock_post.call_count, 1)

    def test_from_chat_messages(self):
        """Test the from_chat_messages class method."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]

        instance = AskResponses.from_chat_messages(
            messages=messages, openai_api_key="test-key", model="gpt-4o"
        )

        self.assertIsInstance(instance, AskResponses)
        self.assertEqual(instance.openai_api_key, "test-key")
        self.assertEqual(instance.model, "gpt-4o")


if __name__ == "__main__":
    unittest.main()
