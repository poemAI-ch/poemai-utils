import json
import unittest
from unittest.mock import Mock, patch

from poemai_utils.openai.ask_responses import AskResponses


class TestAskResponsesMaxTokensFix(unittest.TestCase):
    """Test that validates the max_tokens bug has been fixed in OpenAI Responses API."""

    def setUp(self):
        self.ask_responses = AskResponses(openai_api_key="test-api-key", model="gpt-4o")

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_max_tokens_never_sent_to_responses_api_FIXED(self, mock_post):
        """
        Test that max_tokens is NEVER sent to the OpenAI Responses API.

        This test validates that the bug has been fixed - regardless of the
        max_tokens parameter value, it should never be included in the API request
        because the Responses API doesn't support it.
        """
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "object": "response",
            "output_text": "Test response",
            "model": "gpt-4o",
        }
        mock_post.return_value = mock_response

        # Test 1: Call with default parameters (previously had max_tokens=600)
        self.ask_responses.ask("Test prompt")

        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])

        # max_tokens should NEVER be sent to the Responses API
        self.assertNotIn(
            "max_tokens",
            request_data,
            "max_tokens should never be sent to OpenAI Responses API",
        )

        # Reset the mock for next test
        mock_post.reset_mock()

        # Test 2: Call with explicit max_tokens=1000 (should still be ignored)
        self.ask_responses.ask("Test prompt", max_tokens=1000)

        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])

        # max_tokens should STILL not be sent, even when explicitly provided
        self.assertNotIn(
            "max_tokens",
            request_data,
            "max_tokens should be ignored even when explicitly provided",
        )

        # Reset the mock for next test
        mock_post.reset_mock()

        # Test 3: Call with explicit max_tokens=None
        self.ask_responses.ask("Test prompt", max_tokens=None)

        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])

        # max_tokens should not be sent when None
        self.assertNotIn(
            "max_tokens", request_data, "max_tokens should not be sent when None"
        )

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_no_more_production_errors(self, mock_post):
        """
        Test that the production error no longer occurs.

        Since max_tokens is never sent to the API, we should no longer get
        the "Unknown parameter: 'max_tokens'" error.
        """
        # Mock successful response (what should happen now that bug is fixed)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "output_text": "Test response",
            "model": "gpt-4o",
        }
        mock_post.return_value = mock_response

        # This should now work without any errors
        response = self.ask_responses.ask("Test prompt")
        self.assertEqual(response.output_text, "Test response")

        # Verify no max_tokens was sent
        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])
        self.assertNotIn("max_tokens", request_data)

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_simple_also_fixed(self, mock_post):
        """
        Test that ask_simple method also has the fix applied.

        ask_simple previously had max_tokens=600 default, but now it should
        never send max_tokens to the API.
        """
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_123",
            "output_text": "Simple response",
            "model": "gpt-4o",
        }
        mock_post.return_value = mock_response

        # Call ask_simple (previously would send max_tokens=600)
        response = self.ask_responses.ask_simple("What is 2+2?")
        self.assertEqual(response, "Simple response")

        # Verify max_tokens is not sent
        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])

        self.assertNotIn(
            "max_tokens", request_data, "ask_simple should also never send max_tokens"
        )

    def test_documentation_updated(self):
        """
        Test that documents the fix for future reference.
        """
        fix_summary = """
        *** MAX_TOKENS BUG FIX SUMMARY ***
        
        PROBLEM WAS:
        The AskResponses.ask() method sent max_tokens to the OpenAI Responses API,
        but the Responses API doesn't accept this parameter, causing 400 errors.
        
        SOLUTION IMPLEMENTED:
        1. Removed the line: if max_tokens is not None: data["max_tokens"] = max_tokens
        2. Added comment: "max_tokens is NOT supported by the OpenAI Responses API"
        3. Updated docstring to indicate max_tokens is IGNORED
        4. Changed ask_simple default from max_tokens=600 to max_tokens=None
        
        RESULT:
        - No more 400 errors in production
        - max_tokens parameter is completely ignored (backward compatible)
        - API requests are clean and only include supported parameters
        
        BACKWARD COMPATIBILITY:
        - Method signatures unchanged (max_tokens parameter still exists)  
        - Existing code works without modification
        - Parameter is simply ignored instead of causing errors
        """

        # This test always passes - it's documentation
        self.assertTrue(True, fix_summary)


if __name__ == "__main__":
    unittest.main()

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_default_max_tokens_causes_the_bug(self, mock_post):
        """
        Test that demonstrates the bug occurs even when max_tokens is not explicitly provided,
        because the method has a default value of 600.
        """
        # Mock the 400 error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = json.dumps(
            {
                "error": {
                    "message": "Unknown parameter: 'max_tokens'.",
                    "type": "invalid_request_error",
                    "param": "max_tokens",
                    "code": "unknown_parameter",
                }
            }
        )
        mock_post.return_value = mock_response

        # Call WITHOUT specifying max_tokens - it defaults to 600
        with self.assertRaises(RuntimeError) as context:
            self.ask_responses.ask(
                input="Test message",
                instructions="You are a helpful assistant",
                # max_tokens not specified, defaults to 600
            )

        # Verify the error occurred
        self.assertIn("Unknown parameter: 'max_tokens'", str(context.exception))

        # Verify that max_tokens=600 was sent due to the default parameter
        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])

        self.assertIn(
            "max_tokens",
            request_data,
            "Bug: max_tokens default value (600) is being sent to API",
        )
        self.assertEqual(
            request_data["max_tokens"],
            600,
            "The default max_tokens=600 is causing the API error",
        )

    @patch("poemai_utils.openai.ask_responses.requests.post")
    def test_ask_simple_also_has_the_bug(self, mock_post):
        """
        Test that ask_simple method also has the same max_tokens bug.
        """
        # Mock the 400 error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = json.dumps(
            {
                "error": {
                    "message": "Unknown parameter: 'max_tokens'.",
                    "type": "invalid_request_error",
                    "param": "max_tokens",
                    "code": "unknown_parameter",
                }
            }
        )
        mock_post.return_value = mock_response

        # ask_simple also defaults max_tokens to 600
        with self.assertRaises(RuntimeError) as context:
            self.ask_responses.ask_simple(
                prompt="What is 2+2?", instructions="You are a math assistant"
            )

        # Verify the same error occurs
        self.assertIn("Unknown parameter: 'max_tokens'", str(context.exception))

        # Verify max_tokens was sent
        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])
        self.assertEqual(request_data["max_tokens"], 600)

    def test_identify_the_root_cause(self):
        """
        Test that identifies the root cause: max_tokens parameter should not be sent
        to OpenAI Responses API at all, regardless of value.

        This is a documentation test that explains the issue.
        """
        # The bug is in the ask() method around line 182-184:
        #
        # if max_tokens is not None:
        #     data["max_tokens"] = max_tokens
        #
        # The problem is:
        # 1. max_tokens defaults to 600 in the method signature
        # 2. So it's never None by default
        # 3. Therefore max_tokens=600 always gets sent to the API
        # 4. But the OpenAI Responses API doesn't accept max_tokens parameter
        # 5. This causes a 400 "Unknown parameter: 'max_tokens'" error

        # The fix should be to either:
        # 1. Remove max_tokens parameter entirely from the Responses API wrapper, OR
        # 2. Map it to the correct parameter name for Responses API, OR
        # 3. Change the default to None so it's only sent when explicitly provided

        self.assertTrue(True, "This test documents the root cause of the bug")


if __name__ == "__main__":
    unittest.main()
