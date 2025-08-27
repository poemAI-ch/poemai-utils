import json
import unittest
from unittest.mock import Mock, patch

from poemai_utils.openai.ask_responses import AskResponses


class TestMaxTokensRequestInspection(unittest.TestCase):
    """Inspect exactly what's being sent to the API to confirm the bug."""

    def setUp(self):
        self.ask_responses = AskResponses(openai_api_key="test-key", model="gpt-4o")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_inspect_actual_request_data(self, mock_post):
        """Print the actual request data to see exactly what's being sent."""

        # Mock successful response to avoid error
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output_text": "test", "model": "gpt-4o"}
        mock_post.return_value = mock_response

        # Make the API call
        self.ask_responses.ask(
            input_data="Test message",
            instructions="You are helpful",
            # Note: max_tokens not specified, so it uses default of 600
        )

        # Inspect what was actually sent
        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])

        print("\n=== ACTUAL REQUEST DATA SENT TO API ===")
        print(json.dumps(request_data, indent=2))
        print("========================================\n")

        # Verify the fix is working
        self.assertNotIn(
            "max_tokens",
            request_data,
            "FIX CONFIRMED: max_tokens is no longer being sent!",
        )

        # Verify the clean API request contains expected parameters
        expected_keys = {"model", "input", "temperature", "instructions"}
        self.assertEqual(
            set(request_data.keys()),
            expected_keys,
            "API request should only contain supported parameters",
        )


if __name__ == "__main__":
    unittest.main()
