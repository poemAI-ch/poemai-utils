"""Requests-based OpenAI text-to-speech adapter.

This adapter intentionally does not import the OpenAI SDK.  It is used by
small runtimes, such as the narration worker, where the SDK's HTTP dependency
would otherwise need to be packaged as a Lambda layer.
"""

import requests
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


def _error_for_status(status_code):
    if status_code in {401, 403}:
        error_type = TextToSpeechAuthenticationError
    elif status_code == 400:
        error_type = TextToSpeechInvalidRequestError
    elif status_code == 429:
        error_type = TextToSpeechRateLimitError
    elif isinstance(status_code, int) and status_code >= 500:
        error_type = TextToSpeechProviderServerError
    else:
        error_type = TextToSpeechProviderError
    return error_type(f"OpenAI text-to-speech failed ({error_type.code})")


def _request_id(response):
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return headers.get("x-request-id") or headers.get("X-Request-Id")


class OpenAITextToSpeechLean:
    """OpenAI speech adapter using only the ``requests`` transport."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT_SECONDS = 120
    DEFAULT_MAX_RETRIES = 1
    MAX_TIMEOUT_SECONDS = 150
    MAX_RETRIES = 2

    def __init__(
        self,
        api_key=None,
        base_url=None,
        openai_api_key=None,
        timeout=None,
        max_retries=None,
    ):
        if api_key is not None and openai_api_key is not None:
            raise ValueError("Specify only one of api_key and openai_api_key")
        self.api_key = api_key if api_key is not None else openai_api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = self._bounded_timeout(timeout)
        self.max_retries = self._bounded_retries(max_retries)

    @classmethod
    def _bounded_timeout(cls, timeout):
        value = cls.DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timeout must be between 1 and 150 seconds")
        if not 1 <= value <= cls.MAX_TIMEOUT_SECONDS:
            raise ValueError("timeout must be between 1 and 150 seconds")
        return value

    @classmethod
    def _bounded_retries(cls, max_retries):
        value = cls.DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= cls.MAX_RETRIES
        ):
            raise ValueError("max_retries must be between 0 and 2")
        return value

    def synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResult:
        if not isinstance(request, TextToSpeechRequest):
            raise TypeError("request must be a TextToSpeechRequest")
        if len(request.text) > 4096:
            raise ValueError("OpenAI speech input must not exceed 4096 characters")

        payload = {
            "model": request.model,
            "voice": request.voice,
            "input": request.text,
            "response_format": request.output_format.response_format,
        }
        if request.instructions is not None:
            payload["instructions"] = request.instructions
        if request.speed is not None:
            payload["speed"] = request.speed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/audio/speech"

        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                if attempt < self.max_retries:
                    continue
                raise TextToSpeechConnectionError(
                    "OpenAI text-to-speech failed (connection_failed)"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise TextToSpeechProviderError(
                    "OpenAI text-to-speech failed (provider_error)"
                ) from exc

            try:
                status_code = getattr(response, "status_code", None)
                if not isinstance(status_code, int) or not 200 <= status_code < 300:
                    error = _error_for_status(status_code)
                    if error.retryable and attempt < self.max_retries:
                        continue
                    raise error

                try:
                    audio = response.content
                    if not isinstance(audio, bytes) or not audio:
                        raise ValueError("provider returned no audio bytes")
                    return TextToSpeechResult(
                        audio_bytes=audio,
                        provider="openai",
                        model_key=request.model,
                        voice=request.voice,
                        output_format=request.output_format,
                        provider_request_id=_request_id(response),
                    )
                except TextToSpeechError:
                    raise
                except Exception as exc:
                    raise TextToSpeechProviderError(
                        "OpenAI text-to-speech failed (provider_error)"
                    ) from exc
            finally:
                if response is not None:
                    response.close()

        raise TextToSpeechProviderError("OpenAI text-to-speech failed (provider_error)")
