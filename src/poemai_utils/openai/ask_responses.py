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


class ConversationManager:
    """
    Manages stateful conversations using the Responses API.

    This class automatically handles conversation state by storing response IDs
    and using them for subsequent requests, eliminating the need to manually
    manage message history.
    """

    def __init__(self, ask_responses: "AskResponses"):
        self.ask_responses = ask_responses
        self.last_response_id: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = []

    def send(
        self,
        input_data: Union[str, List[Dict[str, Any]]],
        instructions: Optional[str] = None,
        **kwargs,
    ) -> PydanticLikeBox:
        """
        Send a message in the stateful conversation.

        Args:
            input_data: The input message
            instructions: System instructions (only used for first message typically)
            **kwargs: Additional arguments passed to ask()

        Returns:
            Response object with conversation state maintained
        """
        # Ensure store is enabled for stateful conversations
        kwargs.setdefault("store", True)

        # Use previous response ID if available
        if self.last_response_id is not None:
            kwargs["previous_response_id"] = self.last_response_id

        response = self.ask_responses.ask(
            input_data=input_data, instructions=instructions, **kwargs
        )

        # Store the response ID for next message
        if hasattr(response, "id"):
            self.last_response_id = response.id

        # Keep a local history for debugging/reference (optional)
        self.conversation_history.append(
            {
                "input": input_data,
                "instructions": instructions,
                "response_id": getattr(response, "id", None),
                "output_text": getattr(response, "output_text", None),
            }
        )

        return response

    def reset(self):
        """Reset the conversation state."""
        self.last_response_id = None
        self.conversation_history = []

    def get_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID (last response ID)."""
        return self.last_response_id


class AskResponses:
    """
    A lightweight wrapper around OpenAI's new Responses API.

    The Responses API is OpenAI's recommended approach for new applications,
    providing a simpler interface than the Chat Completions API while supporting
    the same underlying models and capabilities.

    Key differences from Chat Completions API:
    - Uses `input` parameter instead of `messages` array
    - Uses `instructions` parameter for system prompts
    - Returns `output_text` directly instead of nested choice structure
    - Supports the same models and features (vision, function calling, etc.)
    """

    OPENAI_MODEL = OPENAI_MODEL  # to make it easier to import / access, just use AskResponses.OPENAI_MODEL

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1/responses",
        timeout: int = 60,
        max_retries: int = 3,
        base_delay: float = 1.0,  # seconds
    ):
        self.openai_api_key = openai_api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay

    def ask(
        self,
        input_data: Union[str, List[Dict[str, Any]]],
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        store: Optional[bool] = None,
        previous_response_id: Optional[str] = None,
        include: Optional[List[str]] = None,
        additional_args: Optional[Dict[str, Any]] = None,
    ) -> Union[PydanticLikeBox, Any]:
        """
        Send a request to OpenAI's Responses API.

        Args:
            input_data: The input to the model. Can be:
                - A string for simple text input
                - A list of content objects for complex inputs (vision, etc.)
                - For multi-turn conversations, use a list of message objects
            instructions: System instructions for the model (replaces system messages)
            model: Model to use (overrides instance default)
            temperature: Sampling temperature (0-2)
            max_tokens: IGNORED - The Responses API does not support max_tokens parameter
            stop: Stop sequences
            tools: Available tools/functions
            tool_choice: Tool choice strategy
            response_format: Response format specification
            stream: Whether to stream the response
            store: Whether to store the conversation (default: True for stateful conversations)
            previous_response_id: ID of previous response for stateful conversations
            include: Additional data to include in response (e.g., ["reasoning.encrypted_content"])
            additional_args: Additional arguments to pass to the API

        Returns:
            Response object with output_text attribute and id for stateful conversations
        """
        use_model = model if model is not None else self.model

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        data = {
            "model": use_model,
            "input": input_data,
            "temperature": temperature,
        }

        if instructions is not None:
            data["instructions"] = instructions

        # Note: max_tokens is NOT supported by the OpenAI Responses API
        # Removed: if max_tokens is not None: data["max_tokens"] = max_tokens

        if stop is not None:
            data["stop"] = stop

        if tools is not None:
            data["tools"] = tools

        if tool_choice is not None:
            data["tool_choice"] = tool_choice

        if response_format is not None:
            data["response_format"] = response_format

        if stream:
            data["stream"] = stream

        if store is not None:
            data["store"] = store

        if previous_response_id is not None:
            data["previous_response_id"] = previous_response_id

        if include is not None:
            data["include"] = include

        if additional_args is not None:
            data.update(additional_args)

        for attempt in range(self.max_retries):
            try:
                _logger.debug(
                    f"Sending request to OpenAI Responses API: url={self.base_url} data={data}"
                )

                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout,
                    stream=stream,
                )

                if response.status_code == 200:
                    if stream:
                        return self._handle_streaming_response(response)
                    else:
                        response_json = response.json()
                        _logger.debug(
                            f"Received response from OpenAI Responses API: {response_json}"
                        )
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
                            f"OpenAI Responses API call failed with status {response.status_code}: {response.text}"
                        )
            except requests.exceptions.RequestException as e:
                # Network or connection error - retry if possible
                if attempt < self.max_retries - 1:
                    sleep_time = self.base_delay * (2**attempt)
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"OpenAI Responses API request failed: {e}")

        # If we got here, it means we exhausted all retries
        raise RuntimeError("Failed to get a successful response after all retries.")

    def _handle_streaming_response(self, response):
        """Handle streaming response from the API."""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield PydanticLikeBox(data)
                    except json.JSONDecodeError:
                        continue

    def ask_simple(
        self,
        prompt: str,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Simplified interface for basic text generation.

        Args:
            prompt: The text prompt
            instructions: System instructions (optional)
            model: Model to use (overrides instance default)
            temperature: Sampling temperature
            max_tokens: IGNORED - The Responses API does not support max_tokens parameter

        Returns:
            Generated text as a string
        """
        response = self.ask(
            input_data=prompt,
            instructions=instructions,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.output_text

    def ask_vision(
        self,
        text: str,
        image_url: str,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = 600,
    ) -> str:
        """
        Simplified interface for vision tasks.

        Args:
            text: The text prompt
            image_url: URL or base64 data URL of the image
            instructions: System instructions (optional)
            model: Model to use (overrides instance default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text as a string
        """
        input_data = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ]

        response = self.ask(
            input_data=input_data,
            instructions=instructions,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.output_text

    @classmethod
    def from_chat_messages(
        cls,
        messages: List[Dict[str, str]],
        openai_api_key: str,
        model: str = "gpt-4o",
        **kwargs,
    ) -> "AskResponses":
        """
        Create an AskResponses instance and convert chat messages to responses format.

        This helper method allows migration from the old Chat Completions format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            openai_api_key: OpenAI API key
            model: Model to use
            **kwargs: Additional arguments for AskResponses constructor

        Returns:
            AskResponses instance
        """
        instance = cls(openai_api_key=openai_api_key, model=model, **kwargs)
        return instance

    @staticmethod
    def convert_messages_to_input(
        messages: List[Dict[str, str]]
    ) -> tuple[Optional[str], Union[str, List[Dict[str, Any]]]]:
        """
        Convert Chat Completions messages format to Responses API format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Tuple of (instructions, input_data)
        """
        instructions = None
        input_messages = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role == "system":
                # Convert system message to instructions
                if instructions is None:
                    instructions = content
                else:
                    instructions += "\n\n" + content
            elif role in ["user", "assistant"]:
                # Keep user and assistant messages as input
                input_messages.append({"role": role, "content": content})

        # If only one user message, return as simple string
        if len(input_messages) == 1 and input_messages[0]["role"] == "user":
            return instructions, input_messages[0]["content"]

        # Multiple messages or complex conversation
        return instructions, input_messages

    def start_conversation(self) -> ConversationManager:
        """
        Create a new stateful conversation manager.

        Returns:
            ConversationManager instance for handling stateful conversations
        """
        return ConversationManager(self)


# Backward compatibility alias
AskResponsesLean = AskResponses
