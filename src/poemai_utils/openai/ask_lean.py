import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
from box import Box
from poemai_utils.openai.openai_model import OPENAI_MODEL

_logger = logging.getLogger(__name__)


class PydanticLikeBox(Box):
    def dict(self):
        return self.to_dict()


class AskLean:
    OPENAI_MODEL = (
        OPENAI_MODEL  # to make it easier to import / access, just use Ask.OPENAI_MODEL
    )

    def __init__(
        self,
        openai_api_key,
        model="gpt-4",
        base_url="https://api.openai.com/v1/chat/completions",
        timeout=60,
        max_retries=3,
        base_delay=1.0,  # seconds
    ):
        self.openai_api_key = openai_api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay

    def ask(
        self,
        messages,
        model=None,
        temperature=0,
        max_tokens=600,
        stop=None,
        tools=None,
        tool_choice=None,
        json_mode=False,  # still just a placeholder
        response_format=None,
        additional_args=None,
    ):
        use_model = model if model is not None else self.model

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        data = {"model": use_model, "messages": messages, "temperature": temperature}

        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        if stop is not None:
            data["stop"] = stop

        if tools is not None:
            data["tools"] = tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice

        # Add response_format if provided
        if response_format is not None:
            data["response_format"] = response_format

        if additional_args is not None:
            data.update(additional_args)

        for attempt in range(self.max_retries):
            try:
                _logger.debug(
                    f"Sending request to OpenAI API: url={self.base_url} data={data}"
                )
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    response_json = response.json()
                    _logger.debug(f"Received response from OpenAI API: {response_json}")
                    retval = PydanticLikeBox(response_json)
                    return retval

                else:
                    # Non-200 response. Retry if it's a server error.
                    if (
                        500 <= response.status_code < 600
                        and attempt < self.max_retries - 1
                    ):
                        sleep_time = self.base_delay * (2**attempt)
                        time.sleep(sleep_time)
                        continue
                    else:
                        # Non-retryable error or last attempt
                        raise RuntimeError(
                            f"OpenAI API call failed with status {response.status_code}: {response.text}"
                        )
            except requests.exceptions.RequestException as e:
                # Network or connection error - retry if possible
                if attempt < self.max_retries - 1:
                    sleep_time = self.base_delay * (2**attempt)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"OpenAI API request failed: {e}")

        # If we got here, it means we exhausted all retries
        raise RuntimeError("Failed to get a successful response after all retries.")

    def to_responses_api(self):
        """
        Create an AskResponses instance with the same configuration.

        This method helps migrate from the Chat Completions API to the new Responses API.

        Returns:
            AskResponses instance with equivalent configuration
        """
        from poemai_utils.openai.ask_responses import AskResponses

        return AskResponses(
            openai_api_key=self.openai_api_key,
            model=self.model,
            timeout=self.timeout,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
        )

    def ask_with_responses_api(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = 600,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        additional_args: Optional[Dict[str, Any]] = None,
    ) -> PydanticLikeBox:
        """
        Send a request using the Responses API instead of Chat Completions API.

        This method converts the traditional messages format to the new Responses API format
        and provides the same interface as the ask() method.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (overrides instance default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            tools: Available tools/functions
            tool_choice: Tool choice strategy
            response_format: Response format specification
            additional_args: Additional arguments to pass to the API

        Returns:
            Response object (compatible with Chat Completions format)
        """
        responses_api = self.to_responses_api()

        # Convert messages to Responses API format
        from poemai_utils.openai.ask_responses import AskResponses

        instructions, input_data = AskResponses.convert_messages_to_input(messages)

        # Make the call using Responses API
        response = responses_api.ask(
            input_data=input_data,
            instructions=instructions,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            additional_args=additional_args,
        )

        # Convert response back to Chat Completions format for compatibility
        if hasattr(response, "output_text"):
            # Create a compatible response structure
            compatible_response = {
                "choices": [
                    {
                        "message": {
                            "content": response.output_text,
                            "role": "assistant",
                        },
                        "finish_reason": getattr(response, "finish_reason", "stop"),
                        "index": 0,
                    }
                ],
                "usage": getattr(response, "usage", {}),
                "id": getattr(response, "id", ""),
                "object": "chat.completion",
                "created": getattr(response, "created", int(time.time())),
                "model": getattr(response, "model", model or self.model),
            }
            return PydanticLikeBox(compatible_response)

        return response
