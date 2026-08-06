from poemai_utils.audio.contracts import TextToSpeechRequest, TextToSpeechResult
from poemai_utils.audio.errors import (
    TextToSpeechAuthenticationError,
    TextToSpeechConnectionError,
    TextToSpeechError,
    TextToSpeechInvalidRequestError,
    TextToSpeechProviderError,
    TextToSpeechProviderServerError,
    TextToSpeechRateLimitError,
)
from poemai_utils.audio.openai_speech_model import OPENAI_SPEECH_MODEL


def _error_for_provider_exception(exc):
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name in {"AuthenticationError", "PermissionDeniedError"} or status in {401, 403}:
        cls = TextToSpeechAuthenticationError
    elif name == "BadRequestError" or status == 400:
        cls = TextToSpeechInvalidRequestError
    elif name == "RateLimitError" or status == 429:
        cls = TextToSpeechRateLimitError
    elif name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "TimeoutException",
    }:
        cls = TextToSpeechConnectionError
    elif isinstance(status, int) and status >= 500:
        cls = TextToSpeechProviderServerError
    else:
        cls = TextToSpeechProviderError
    return cls(f"OpenAI text-to-speech failed ({cls.code})")


class OpenAITextToSpeech:
    OPENAI_SPEECH_MODEL = OPENAI_SPEECH_MODEL

    def __init__(
        self,
        api_key=None,
        base_url=None,
        client=None,
        openai_api_key=None,
        timeout=None,
        max_retries=None,
    ):
        if api_key is not None and openai_api_key is not None:
            raise ValueError("Specify only one of api_key and openai_api_key")
        if client is not None and any(
            value is not None
            for value in (api_key, openai_api_key, base_url, timeout, max_retries)
        ):
            raise ValueError(
                "client cannot be combined with API key, base URL, timeout, or retries"
            )
        if api_key is None:
            api_key = openai_api_key

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "You must install openai to use this function. Try: pip install openai"
                ) from exc
            openai_args = {}
            if api_key is not None:
                openai_args["api_key"] = api_key
            if base_url is not None:
                openai_args["base_url"] = base_url
            if timeout is not None:
                openai_args["timeout"] = timeout
            if max_retries is not None:
                openai_args["max_retries"] = max_retries
            client = OpenAI(**openai_args)
        self.client = client

    def synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResult:
        if not isinstance(request, TextToSpeechRequest):
            raise TypeError("request must be a TextToSpeechRequest")
        if len(request.text) > 4096:
            raise ValueError("OpenAI speech input must not exceed 4096 characters")

        args = {
            "model": request.model,
            "voice": request.voice,
            "input": request.text,
            "response_format": request.output_format.response_format,
        }
        if request.instructions is not None:
            args["instructions"] = request.instructions
        if request.speed is not None:
            args["speed"] = request.speed

        try:
            response = self.client.audio.speech.create(**args)
        except TextToSpeechError:
            raise
        except Exception as exc:
            raise _error_for_provider_exception(exc) from exc

        try:
            try:
                audio = response.read()
                request_id = getattr(response, "request_id", None)
                http_response = getattr(response, "response", None)
                headers = getattr(http_response, "headers", None)
                if request_id is None:
                    headers = headers or getattr(response, "headers", None)
                    if headers is not None:
                        request_id = headers.get("x-request-id") or headers.get(
                            "X-Request-Id"
                        )
                result = TextToSpeechResult(
                    audio_bytes=audio,
                    provider="openai",
                    model_key=request.model,
                    voice=request.voice,
                    output_format=request.output_format,
                    provider_request_id=request_id,
                )
            except Exception as exc:
                if isinstance(exc, TextToSpeechError):
                    raise
                raise _error_for_provider_exception(exc) from exc
            return result
        finally:
            response.close()
