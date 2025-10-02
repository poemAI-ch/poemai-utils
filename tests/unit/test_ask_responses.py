import json
import unittest
from unittest.mock import Mock, patch

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
        # max_tokens is no longer sent to the Responses API (by design - it's not supported)
        self.assertNotIn("max_tokens", request_data)
        self.assertEqual(request_data["temperature"], 0)

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
    def test_ask_with_tools_includes_payload(self, mock_post):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_tool_1",
            "model": "gpt-4o-mini",
            "output_text": "",
            "output": [
                {
                    "id": "toolcall_1",
                    "type": "tool_call",
                    "tool_call": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris", "unit": "celsius"}',
                    },
                }
            ],
        }
        mock_post.return_value = mock_response

        response = self.ask_responses.ask(
            input_data="What's the weather in Paris?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        args, kwargs = mock_post.call_args
        request_data = json.loads(kwargs["data"])
        normalized_tools = request_data["tools"]
        self.assertEqual(len(normalized_tools), 1)
        self.assertEqual(normalized_tools[0]["type"], "function")
        self.assertEqual(normalized_tools[0]["name"], "get_weather")
        self.assertEqual(normalized_tools[0]["description"], "Get current weather")
        self.assertEqual(normalized_tools[0]["parameters"]["type"], "object")
        self.assertFalse(normalized_tools[0]["parameters"]["additionalProperties"])
        self.assertEqual(normalized_tools[0]["parameters"]["required"], ["location"])

        self.assertEqual(
            request_data["tool_choice"],
            {"type": "function", "name": "get_weather"},
        )

        tool_calls = AskResponses.extract_tool_calls(response)
        self.assertEqual(len(tool_calls), 1)
        tool_call = tool_calls[0]
        self.assertEqual(tool_call.name, "get_weather")
        self.assertEqual(json.loads(tool_call.arguments)["location"], "Paris")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_json_response_format(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_json_1",
            "model": "gpt-4o-mini",
            "output_text": json.dumps({"name": "John", "age": 25}),
        }
        mock_post.return_value = mock_response

        response = self.ask_responses.ask(
            input_data="Extract name and age from 'John is 25 years old'.",
            instructions="Return a JSON object with fields name and age.",
            response_format={"type": "json_object"},
        )

        args, kwargs = mock_post.call_args
        request_data = json.loads(kwargs["data"])
        self.assertEqual(
            request_data["text"],
            {"format": {"type": "json_object"}},
        )

        payload = json.loads(response.output_text)
        self.assertEqual(payload["name"], "John")
        self.assertEqual(payload["age"], 25)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_reasoning_controls(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_reasoning_1",
            "model": "gpt-4o-mini",
            "output_text": "Answer",
            "reasoning": {
                "content": [
                    {"type": "text", "text": "Step 1"},
                    {"type": "text", "text": "Step 2"},
                ]
            },
        }
        mock_post.return_value = mock_response

        response = self.ask_responses.ask(
            input_data="Add 2 and 2.",
            reasoning={"effort": "medium"},
            include=["message.output_text.logprobs"],
            model="gpt-5",
        )

        args, kwargs = mock_post.call_args
        request_data = json.loads(kwargs["data"])
        self.assertEqual(request_data["model"], "gpt-5")
        self.assertEqual(request_data["reasoning"], {"effort": "medium"})
        self.assertEqual(request_data["include"], ["message.output_text.logprobs"])

        self.assertEqual(response.output_text, "Answer")
        self.assertEqual(len(response.reasoning.content), 2)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_max_output_tokens(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_max_output",
            "model": "gpt-4o-mini",
            "output_text": "Hello there!",
        }
        mock_post.return_value = mock_response

        response = self.ask_responses.ask(
            input_data="Say hello",
            max_output_tokens=42,
        )

        args, kwargs = mock_post.call_args
        request_data = json.loads(kwargs["data"])
        self.assertEqual(request_data["max_output_tokens"], 42)
        self.assertEqual(response.output_text, "Hello there!")

    def test_extract_tool_calls_from_message_content(self):
        response_payload = PydanticLikeBox(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "id": "call_1",
                                "name": "search",
                                "arguments": '{"query": "weather"}',
                            }
                        ],
                    }
                ]
            }
        )

        tool_calls = AskResponses.extract_tool_calls(response_payload)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "search")
        self.assertIn("query", json.loads(tool_calls[0].arguments))

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_fx_tool_call_flow_matches_reference(self, mock_post):
        ask = AskResponses(openai_api_key="test-key", model="gpt-5")

        first_response_json = {
            "id": "resp_first",
            "object": "response",
            "created_at": 1759337773,
            "status": "completed",
            "instructions": (
                "You’re a precise finance assistant. If currency conversion is requested, "
                "you MUST call get_fx_rate and then compute the result."
            ),
            "model": "gpt-5-2025-08-07",
            "output": [
                {"id": "rs_reasoning", "type": "reasoning", "summary": []},
                {
                    "id": "fc_tool",
                    "type": "function_call",
                    "status": "completed",
                    "arguments": '{"base":"CHF","quote":"EUR"}',
                    "call_id": "call_fx",
                    "name": "get_fx_rate",
                },
            ],
            "parallel_tool_calls": True,
            "store": True,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "tool_choice": "auto",
            "tools": [],
        }

        second_response_text = (
            "- Rate used: 1 CHF = 1.04 EUR\n"
            "- Calculation: 120 CHF × 1.04 EUR/CHF = 124.80 EUR\n"
            "- Result: 124.80 EUR\n\n"
            "Note: This uses the current mid-market rate and excludes any fees or bank markups."
        )

        second_response_json = {
            "id": "resp_second",
            "object": "response",
            "created_at": 1759337776,
            "status": "completed",
            "model": "gpt-5-2025-08-07",
            "output": [
                {"id": "rs_reasoning_2", "type": "reasoning", "summary": []},
                {
                    "id": "msg_final",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [],
                            "logprobs": [],
                            "text": second_response_text,
                        }
                    ],
                },
            ],
            "previous_response_id": "resp_first",
            "parallel_tool_calls": True,
            "store": True,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
        }

        first_mock = Mock()
        first_mock.status_code = 200
        first_mock.json.return_value = first_response_json

        second_mock = Mock()
        second_mock.status_code = 200
        second_mock.json.return_value = second_response_json

        mock_post.side_effect = [first_mock, second_mock]

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

        first = ask.ask(
            input_data=[
                {
                    "role": "user",
                    "content": "Convert 120 CHF to EUR and show your calculation.",
                }
            ],
            instructions=(
                "You’re a precise finance assistant. If currency conversion is requested, "
                "you MUST call get_fx_rate and then compute the result."
            ),
            tools=tools,
        )

        first_request = json.loads(mock_post.call_args_list[0][1]["data"])
        expected_first_request = {
            "model": "gpt-5",
            "input": [
                {
                    "role": "user",
                    "content": "Convert 120 CHF to EUR and show your calculation.",
                }
            ],
            "instructions": (
                "You’re a precise finance assistant. If currency conversion is requested, "
                "you MUST call get_fx_rate and then compute the result."
            ),
            "tools": tools,
        }
        self.assertEqual(first_request, expected_first_request)

        calls = AskResponses.extract_tool_calls(first)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call.name, "get_fx_rate")
        call_args = json.loads(call.arguments)
        self.assertEqual(call_args, {"base": "CHF", "quote": "EUR"})

        tool_result = {"ok": True, "rate": 1.04, "base": "CHF", "quote": "EUR"}

        second = ask.ask(
            input_data=[
                {
                    "type": "function_call_output",
                    "call_id": getattr(call, "call_id", getattr(call, "id", None)),
                    "output": json.dumps(tool_result, ensure_ascii=False),
                }
            ],
            previous_response_id=first_response_json["id"],
        )

        second_request = json.loads(mock_post.call_args_list[1][1]["data"])
        expected_second_request = {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_fx",
                    "output": json.dumps(tool_result, ensure_ascii=False),
                }
            ],
            "previous_response_id": "resp_first",
        }
        self.assertEqual(second_request, expected_second_request)
        self.assertEqual(second.output_text, second_response_text)

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
