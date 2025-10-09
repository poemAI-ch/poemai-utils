import json
import unittest
from unittest.mock import Mock, patch

from poemai_utils.openai.ask_responses import AskResponses, ConversationManager


class TestStatefulConversations(unittest.TestCase):
    """Test stateful conversation functionality in the Responses API."""

    def setUp(self):
        self.api_key = "test-api-key"
        self.ask_responses = AskResponses(openai_api_key=self.api_key, model="gpt-4o")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_with_stateful_parameters(self, mock_post):
        """Test that stateful parameters are correctly passed to the API."""
        # Mock response with an ID
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123abc",
            "object": "response",
            "model": "gpt-4o",
            "output_text": "This is a stateful response.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_post.return_value = mock_response

        # Test request with stateful parameters
        response = self.ask_responses.ask(
            input="Hello",
            instructions="Be helpful",
            store=True,
            previous_response_id="resp_previous_123",
            include=["reasoning.encrypted_content"],
        )

        # Verify request parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        request_data = json.loads(kwargs["data"])

        self.assertEqual(request_data["store"], True)
        self.assertEqual(request_data["previous_response_id"], "resp_previous_123")
        self.assertEqual(request_data["include"], ["reasoning.encrypted_content"])

        # Verify response
        self.assertEqual(response.id, "resp_123abc")
        self.assertEqual(response.output_text, "This is a stateful response.")

    def test_conversation_manager_creation(self):
        """Test creating a conversation manager."""
        conversation = self.ask_responses.start_conversation()

        self.assertIsInstance(conversation, ConversationManager)
        self.assertEqual(conversation.ask_responses, self.ask_responses)
        self.assertIsNone(conversation.last_response_id)
        self.assertEqual(len(conversation.conversation_history), 0)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_conversation_manager_first_message(self, mock_post):
        """Test sending the first message in a stateful conversation."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_first_123",
            "output_text": "Hello! How can I help you?",
        }
        mock_post.return_value = mock_response

        conversation = self.ask_responses.start_conversation()
        response = conversation.send(input="Hello", instructions="Be helpful")

        # Verify the first message sets up stateful conversation
        request_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(request_data["store"], True)  # Should default to True
        self.assertNotIn(
            "previous_response_id", request_data
        )  # First message has no previous ID

        # Verify conversation state is updated
        self.assertEqual(conversation.last_response_id, "resp_first_123")
        self.assertEqual(len(conversation.conversation_history), 1)
        self.assertEqual(response.output_text, "Hello! How can I help you?")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_conversation_manager_follow_up_message(self, mock_post):
        """Test sending follow-up messages that use previous response ID."""
        # Mock responses for two messages
        first_response = Mock()
        first_response.status_code = 200
        first_response.json.return_value = {
            "id": "resp_first_123",
            "output_text": "Hello! How can I help you?",
        }

        second_response = Mock()
        second_response.status_code = 200
        second_response.json.return_value = {
            "id": "resp_second_456",
            "output_text": "The capital of France is Paris.",
        }

        mock_post.side_effect = [first_response, second_response]

        conversation = self.ask_responses.start_conversation()

        # First message
        response1 = conversation.send("Hello")

        # Second message should use the ID from the first
        response2 = conversation.send("What's the capital of France?")

        # Verify the second call used the previous response ID
        self.assertEqual(mock_post.call_count, 2)

        # Check first call
        first_call_data = json.loads(mock_post.call_args_list[0][1]["data"])
        self.assertEqual(first_call_data["store"], True)
        self.assertNotIn("previous_response_id", first_call_data)

        # Check second call
        second_call_data = json.loads(mock_post.call_args_list[1][1]["data"])
        self.assertEqual(second_call_data["store"], True)
        self.assertEqual(second_call_data["previous_response_id"], "resp_first_123")

        # Verify conversation state
        self.assertEqual(conversation.last_response_id, "resp_second_456")
        self.assertEqual(len(conversation.conversation_history), 2)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_conversation_manager_multiple_turns(self, mock_post):
        """Test a multi-turn conversation with state tracking."""
        # Mock multiple responses
        responses = [
            {"id": "resp_1", "output_text": "Response 1"},
            {"id": "resp_2", "output_text": "Response 2"},
            {"id": "resp_3", "output_text": "Response 3"},
        ]

        mock_response_objects = []
        for resp_data in responses:
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = resp_data
            mock_response_objects.append(mock_resp)

        mock_post.side_effect = mock_response_objects

        conversation = self.ask_responses.start_conversation()

        # Send multiple messages
        conversation.send("First message")
        conversation.send("Second message")
        conversation.send("Third message")

        # Verify all calls were made with correct previous_response_id
        self.assertEqual(mock_post.call_count, 3)

        # First call: no previous ID
        first_call = json.loads(mock_post.call_args_list[0][1]["data"])
        self.assertNotIn("previous_response_id", first_call)

        # Second call: uses first response ID
        second_call = json.loads(mock_post.call_args_list[1][1]["data"])
        self.assertEqual(second_call["previous_response_id"], "resp_1")

        # Third call: uses second response ID
        third_call = json.loads(mock_post.call_args_list[2][1]["data"])
        self.assertEqual(third_call["previous_response_id"], "resp_2")

        # Final state should have the last response ID
        self.assertEqual(conversation.last_response_id, "resp_3")
        self.assertEqual(conversation.get_conversation_id(), "resp_3")
        self.assertEqual(len(conversation.conversation_history), 3)

    def test_conversation_manager_reset(self):
        """Test resetting conversation state."""
        conversation = self.ask_responses.start_conversation()

        # Manually set some state
        conversation.last_response_id = "resp_123"
        conversation.conversation_history = [{"test": "data"}]

        # Reset and verify clean state
        conversation.reset()

        self.assertIsNone(conversation.last_response_id)
        self.assertEqual(len(conversation.conversation_history), 0)
        self.assertIsNone(conversation.get_conversation_id())

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_conversation_manager_with_tools(self, mock_post):
        """Test stateful conversation with tools."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_with_tools",
            "output_text": "I'll search for that information.",
        }
        mock_post.return_value = mock_response

        conversation = self.ask_responses.start_conversation()
        response = conversation.send(
            input="Search for Python tutorials",
            tools=[{"type": "web_search_preview"}],
        )

        # Verify tools were passed
        request_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(request_data["tools"], [{"type": "web_search_preview"}])
        self.assertEqual(request_data["store"], True)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_conversation_manager_store_override(self, mock_post):
        """Test that store parameter can be overridden."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_no_store",
            "output_text": "Response without storage",
        }
        mock_post.return_value = mock_response

        conversation = self.ask_responses.start_conversation()
        conversation.send("Hello", store=False)

        # Verify store was set to False
        request_data = json.loads(mock_post.call_args[1]["data"])
        self.assertEqual(request_data["store"], False)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_conversation_history_tracking(self, mock_post):
        """Test that conversation history is properly tracked."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_history_test",
            "output_text": "Test response",
        }
        mock_post.return_value = mock_response

        conversation = self.ask_responses.start_conversation()
        conversation.send(input="Test message", instructions="Test instructions")

        # Verify history entry
        self.assertEqual(len(conversation.conversation_history), 1)
        history_entry = conversation.conversation_history[0]

        self.assertEqual(history_entry["input"], "Test message")
        self.assertEqual(history_entry["instructions"], "Test instructions")
        self.assertEqual(history_entry["response_id"], "resp_history_test")
        self.assertEqual(history_entry["output_text"], "Test response")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_conversation_without_response_id(self, mock_post):
        """Test handling of responses that don't have an ID."""
        # Mock response without ID
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output_text": "Response without ID"}
        mock_post.return_value = mock_response

        conversation = self.ask_responses.start_conversation()
        conversation.send("Hello")

        # Verify conversation state when no ID is returned
        self.assertIsNone(conversation.last_response_id)
        self.assertEqual(len(conversation.conversation_history), 1)
        self.assertIsNone(conversation.conversation_history[0]["response_id"])


if __name__ == "__main__":
    unittest.main()
