import json
import logging
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import requests
from box import Box
from poemai_utils.openai.openai_model import OPENAI_MODEL

_logger = logging.getLogger(__name__)


class PydanticLikeBox(Box):

    __init__ = Box.__init__

    def dict(self):
        return self.to_dict()

    def model_dump(self, *_, **__):
        """Mimic pydantic's model_dump for compatibility with tests."""
        return self.to_dict()

    def model_dump_json(self, *_, **__):
        """Provide json dump helper similar to pydantic."""
        return json.dumps(self.to_dict())


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
        input,
        instructions: Optional[str] = None,
        **kwargs,
    ) -> PydanticLikeBox:
        """
        Send a message in the stateful conversation.

        Args:
            input: The input data
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
            input=input, instructions=instructions, **kwargs
        )

        # Store the response ID for next message
        if hasattr(response, "id"):
            self.last_response_id = response.id

        # Keep a local history for debugging/reference (optional)
        self.conversation_history.append(
            {
                "input": input,
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
        input: Union[str, List[Dict[str, Any]]],
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        poemai_max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        store: Optional[bool] = None,
        previous_response_id: Optional[str] = None,
        include: Optional[List[str]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        additional_args: Optional[Dict[str, Any]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Union[PydanticLikeBox, Any]:
        """
        Send a request to OpenAI's Responses API.

        Args:
            input: The input to the model. Can be:
                - A string for simple text input
                - A dict representing a single message (will be wrapped automatically)
                - A list of content/message objects for complex or multi-turn inputs
            instructions: System instructions for the model (replaces system messages)
            model: Model to use (overrides instance default)
            temperature: Sampling temperature (0-2)
            max_tokens: DEPRECATED - Use poemai_max_tokens instead
            max_output_tokens: Direct max_output_tokens for Responses API (use poemai_max_tokens for auto-mapping)
            poemai_max_tokens: Platform-agnostic token limit (auto-maps to max_output_tokens for Responses API)
            stop: Stop sequences
            tools: Available tools/functions
            tool_choice: Tool choice strategy
            response_format: Response format specification (mapped to Responses `text.format`)
            stream: Whether to stream the response
            store: Whether to store the conversation (default: True for stateful conversations)
            previous_response_id: ID of previous response for stateful conversations
            include: Additional data to include in response (e.g., ["reasoning.encrypted_content"])
            reasoning: Reasoning configuration for reasoning-capable models (e.g., {"effort": "medium"})
            additional_args: Additional arguments to pass to the API

        Returns:
            Response object with output_text attribute and id for stateful conversations
        """
        use_model = model if model is not None else self.model

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        if not isinstance(input, (str, list)):
            input_type = type(input).__name__
            raise ValueError(
                f"Input must be a string or a list of message dicts, got {input_type}"
            )

        if isinstance(input, list):
            if not all(isinstance(msg, Mapping) for msg in input):
                raise ValueError("ask: If input is a list, each item must be a dict")
            if not all(
                isinstance(msg.get("content"), (str, list))
                for msg in input
                if "content" in msg
            ):
                raise ValueError(
                    "ask: If input is a list, each message's content must be a string or a list if present"
                )

        data = {
            "model": use_model,
            "input": input,
        }

        if instructions is not None:
            data["instructions"] = instructions

        # Note: max_tokens is NOT supported by the OpenAI Responses API
        # Removed: if max_tokens is not None: data["max_tokens"] = max_tokens

        if stop is not None:
            data["stop"] = stop

        include_temperature = temperature
        if self._model_supports_temperature(use_model):
            data["temperature"] = include_temperature
        elif include_temperature not in (None, 0):
            _logger.warning(
                "Model %s does not support temperature parameter; omitting it.",
                use_model,
            )

        # Handle poemai_max_tokens - platform-agnostic token limiting
        # For Responses API, map to max_output_tokens
        if poemai_max_tokens is not None:
            if max_output_tokens is not None:
                _logger.warning(
                    "Both poemai_max_tokens and max_output_tokens specified; using max_output_tokens"
                )
            else:
                max_output_tokens = poemai_max_tokens
                _logger.debug(
                    f"Mapping poemai_max_tokens={poemai_max_tokens} to max_output_tokens for Responses API"
                )

        if max_tokens is not None:
            _logger.warning(
                "The Responses API does not support max_tokens parameter; ignoring it. Use poemai_max_tokens instead."
            )

        if parallel_tool_calls is not None:
            data["parallel_tool_calls"] = parallel_tool_calls
        if max_output_tokens is not None:
            data["max_output_tokens"] = max_output_tokens

        if tools is not None:
            data["tools"] = [self._normalize_tool_definition(tool) for tool in tools]

        if tool_choice is not None:
            data["tool_choice"] = self._normalize_tool_choice(tool_choice)

        if response_format is not None:
            text_config = data.get("text", {})
            if not isinstance(text_config, dict):
                text_config = {}
            text_config = dict(text_config)
            text_config["format"] = response_format
            data["text"] = text_config

        if stream:
            data["stream"] = stream

        if store is not None:
            data["store"] = store

        if previous_response_id is not None:
            data["previous_response_id"] = previous_response_id

        if include is not None:
            data["include"] = include

        if reasoning is not None:
            if self._model_supports_reasoning(use_model):
                data["reasoning"] = reasoning
            else:
                _logger.warning(
                    "Model %s does not support reasoning parameter; omitting it.",
                    use_model,
                )

        if additional_args is not None:
            data.update(additional_args)

        _logger.debug(
            f"Sending data to api:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
        )

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
                        self._ensure_output_text_field(response_json)
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
            input=prompt,
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
        input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ]

        response = self.ask(
            input=input,
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
        messages: List[Dict[str, str]],
    ) -> tuple[Optional[str], Union[str, List[Dict[str, Any]]]]:
        """
        Convert Chat Completions messages format to Responses API format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Tuple of (instructions, input)
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

    @staticmethod
    def _normalize_tool_definition(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy Responses tool definitions to the current schema."""

        if not isinstance(tool, dict):
            return tool

        if tool.get("type") != "function" or "function" not in tool:
            return tool

        function_block = tool.get("function") or {}
        normalized = {k: v for k, v in tool.items() if k != "function"}
        normalized.setdefault("type", "function")

        if function_block.get("name"):
            normalized["name"] = function_block["name"]

        if "description" in function_block:
            normalized["description"] = function_block["description"]

        parameters = function_block.get("parameters")
        normalized["parameters"] = AskResponses._ensure_function_schema(parameters)

        if "strict" in function_block:
            normalized["strict"] = function_block["strict"]

        # Preserve any additional keys on the legacy function definition
        for key, value in function_block.items():
            if key in {"name", "description", "parameters", "strict"}:
                continue
            normalized[key] = value

        return normalized

    @staticmethod
    def _normalize_tool_choice(
        tool_choice: Union[str, Dict[str, Any]],
    ) -> Union[str, Dict[str, Any]]:
        """Normalize legacy tool_choice structures to the current schema."""

        if not isinstance(tool_choice, dict):
            return tool_choice

        if tool_choice.get("type") != "function":
            return tool_choice

        if "name" in tool_choice:
            return tool_choice

        if "function" in tool_choice:
            function_block = tool_choice.get("function") or {}
            normalized = {k: v for k, v in tool_choice.items() if k != "function"}
            if "name" not in normalized:
                normalized["name"] = function_block.get("name")
            return normalized

        return tool_choice

    @staticmethod
    def _ensure_function_schema(
        parameters: Optional[Union[Dict[str, Any], List[Any]]],
    ) -> Dict[str, Any]:
        """Ensure function tool schemas comply with Responses requirements."""

        if not isinstance(parameters, dict):
            return {}

        schema = dict(parameters)

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            schema["properties"] = properties

        required = schema.get("required")
        if required is None:
            required = [
                name
                for name, details in properties.items()
                if isinstance(details, dict) and details.get("required") is True
            ]
            if required:
                schema["required"] = required
        elif not isinstance(required, list):
            schema["required"] = list(required)

        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        return schema

    @staticmethod
    def _model_supports_temperature(model: str) -> bool:
        try:
            model_enum = OPENAI_MODEL(model)
        except ValueError:
            return True

        supports = getattr(model_enum, "supports_temperature", True)
        return bool(supports)

    @staticmethod
    def _model_supports_reasoning(model: str) -> bool:
        try:
            model_enum = OPENAI_MODEL(model)
        except ValueError:
            return True

        return bool(getattr(model_enum, "supports_reasoning", False))

    @staticmethod
    def _normalize_input_payload(
        input: Union[str, List[Dict[str, Any]], Mapping[str, Any]],
    ) -> Union[str, List[Dict[str, Any]]]:
        """Coerce dict-style inputs into a list so Responses API accepts them."""

        if isinstance(input, Mapping):
            if isinstance(input, Box):
                payload = input.to_dict()
            else:
                payload = dict(input)

            return [payload]

        return input

    @staticmethod
    def extract_tool_calls(
        response: Union[PydanticLikeBox, Dict[str, Any]],
    ) -> List[PydanticLikeBox]:
        """Extract tool call payloads from a Responses API response."""

        response_dict = response.to_dict() if isinstance(response, Box) else response
        tool_calls: List[PydanticLikeBox] = []

        for block in response_dict.get("output", []) or []:
            block_type = block.get("type")

            if block_type in {"tool_call", "function_call"}:
                payload = block.get("tool_call") or block
                if payload:
                    normalized = dict(payload)
                    if "tool_name" in normalized and "name" not in normalized:
                        normalized["name"] = normalized["tool_name"]
                    if "arguments" in normalized and isinstance(
                        normalized["arguments"], dict
                    ):
                        normalized["arguments"] = json.dumps(normalized["arguments"])
                    tool_calls.append(PydanticLikeBox(normalized))
                continue

            if block_type == "message":
                for content in block.get("content", []) or []:
                    content_type = content.get("type")
                    if content_type in {"tool_call", "tool_use", "function_call"}:
                        payload = content.get("tool_call") or content
                        if payload:
                            normalized = dict(payload)
                            if "tool_name" in normalized and "name" not in normalized:
                                normalized["name"] = normalized["tool_name"]
                            if "arguments" in normalized and isinstance(
                                normalized["arguments"], dict
                            ):
                                normalized["arguments"] = json.dumps(
                                    normalized["arguments"]
                                )
                            tool_calls.append(PydanticLikeBox(normalized))

        return tool_calls

    @staticmethod
    def _ensure_output_text_field(response_json: Dict[str, Any]) -> None:
        """Populate output_text by aggregating message content when missing."""

        if response_json.get("output_text"):
            return

        outputs = response_json.get("output") or []
        chunks: List[str] = []

        for block in outputs:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "message":
                for content in block.get("content", []) or []:
                    if not isinstance(content, dict):
                        continue
                    content_type = content.get("type")
                    if content_type in {"output_text", "text"}:
                        text_value = content.get("text")
                        if isinstance(text_value, str):
                            chunks.append(text_value)

            elif block.get("type") == "output_text":
                text_value = block.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)

        # Set output_text field if we found content
        # Always set it to prevent BoxKeyError, but empty string allows
        # compatibility code to detect "no real output" case
        output_text = "".join(chunks)
        response_json["output_text"] = output_text

        if not output_text and outputs:
            _logger.warning(
                f"No text content found in response with output blocks. Response structure: {json.dumps(response_json, indent=2)}"
            )


# Backward compatibility alias
AskResponsesLean = AskResponses
