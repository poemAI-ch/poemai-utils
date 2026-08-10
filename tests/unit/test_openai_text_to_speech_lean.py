from unittest.mock import Mock, call

import pytest
import requests
from poemai_utils.audio import (
    OpenAITextToSpeechLean,
    SpeechOutputFormat,
    TextToSpeechAuthenticationError,
    TextToSpeechConnectionError,
    TextToSpeechInvalidRequestError,
    TextToSpeechProviderError,
    TextToSpeechProviderServerError,
    TextToSpeechRateLimitError,
    TextToSpeechRequest,
)


class FakeResponse:
    def __init__(self, status_code=200, content=b"audio", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.closed = False

    def close(self):
        self.closed = True


def test_lean_payload_maps_format_instructions_and_speed(monkeypatch):
    response = FakeResponse(headers={"x-request-id": "request-1"})
    post = Mock(return_value=response)
    monkeypatch.setattr(
        "poemai_utils.audio.openai_text_to_speech_lean.requests.post", post
    )
    request = TextToSpeechRequest(
        "hello",
        model="gpt-4o-mini-tts",
        voice="nova",
        output_format=SpeechOutputFormat.OPUS,
        instructions="Speak calmly.",
        speed=1.25,
    )

    result = OpenAITextToSpeechLean(
        api_key="secret", base_url="https://example.test/v1", timeout=75, max_retries=1
    ).synthesize(request)

    post.assert_called_once_with(
        "https://example.test/v1/audio/speech",
        headers={
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini-tts",
            "voice": "nova",
            "input": "hello",
            "response_format": "opus",
            "instructions": "Speak calmly.",
            "speed": 1.25,
        },
        timeout=75,
    )
    assert result.audio == b"audio"
    assert result.content_type == "audio/ogg"
    assert result.request_id == "request-1"
    assert response.closed is True


@pytest.mark.parametrize("output_format", list(SpeechOutputFormat))
def test_lean_payload_supports_all_formats(monkeypatch, output_format):
    response = FakeResponse()
    post = Mock(return_value=response)
    monkeypatch.setattr(
        "poemai_utils.audio.openai_text_to_speech_lean.requests.post", post
    )

    OpenAITextToSpeechLean(api_key="secret").synthesize(
        TextToSpeechRequest("hello", output_format=output_format)
    )

    assert post.call_args.kwargs["json"]["response_format"] == output_format.value


def test_lean_omits_optional_payload_fields(monkeypatch):
    post = Mock(return_value=FakeResponse())
    monkeypatch.setattr(
        "poemai_utils.audio.openai_text_to_speech_lean.requests.post", post
    )

    OpenAITextToSpeechLean(api_key="secret").synthesize(TextToSpeechRequest("hello"))

    assert post.call_args.kwargs["json"] == {
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": "hello",
        "response_format": "mp3",
    }


@pytest.mark.parametrize(
    "status_code,error_type",
    [
        (401, TextToSpeechAuthenticationError),
        (403, TextToSpeechAuthenticationError),
        (400, TextToSpeechInvalidRequestError),
        (429, TextToSpeechRateLimitError),
        (500, TextToSpeechProviderServerError),
        (503, TextToSpeechProviderServerError),
        (404, TextToSpeechProviderError),
    ],
)
def test_lean_maps_http_failures_without_response_body(
    monkeypatch, status_code, error_type
):
    post = Mock(return_value=FakeResponse(status_code=status_code))
    monkeypatch.setattr(
        "poemai_utils.audio.openai_text_to_speech_lean.requests.post", post
    )

    with pytest.raises(error_type) as raised:
        OpenAITextToSpeechLean(api_key="secret", max_retries=0).synthesize(
            TextToSpeechRequest("private narration")
        )

    assert raised.value.code == error_type.code
    assert "private narration" not in str(raised.value)
    assert "secret" not in str(raised.value)


def test_lean_retries_rate_limit_and_server_failures(monkeypatch):
    first = FakeResponse(status_code=429)
    second = FakeResponse(status_code=200, content=b"audio")
    post = Mock(side_effect=[first, second])
    monkeypatch.setattr(
        "poemai_utils.audio.openai_text_to_speech_lean.requests.post", post
    )

    result = OpenAITextToSpeechLean(api_key="secret", max_retries=1).synthesize(
        TextToSpeechRequest("hello")
    )

    assert result.audio == b"audio"
    assert post.call_count == 2
    assert first.closed is True
    assert second.closed is True


def test_lean_retries_connection_failures_and_maps_exhaustion(monkeypatch):
    post = Mock(
        side_effect=[
            requests.exceptions.Timeout("timeout"),
            requests.exceptions.ConnectionError("connection"),
        ]
    )
    monkeypatch.setattr(
        "poemai_utils.audio.openai_text_to_speech_lean.requests.post", post
    )

    with pytest.raises(TextToSpeechConnectionError) as raised:
        OpenAITextToSpeechLean(api_key="secret", max_retries=1).synthesize(
            TextToSpeechRequest("hello")
        )

    assert raised.value.code == "connection_failed"
    assert post.call_args_list == [
        call(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini-tts",
                "voice": "alloy",
                "input": "hello",
                "response_format": "mp3",
            },
            timeout=120,
        ),
        call(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini-tts",
                "voice": "alloy",
                "input": "hello",
                "response_format": "mp3",
            },
            timeout=120,
        ),
    ]


@pytest.mark.parametrize("content", [b"", "audio", None])
def test_lean_rejects_malformed_audio_response(monkeypatch, content):
    response = FakeResponse(content=content)
    monkeypatch.setattr(
        "poemai_utils.audio.openai_text_to_speech_lean.requests.post",
        Mock(return_value=response),
    )

    with pytest.raises(TextToSpeechProviderError) as raised:
        OpenAITextToSpeechLean(api_key="secret").synthesize(
            TextToSpeechRequest("hello")
        )

    assert raised.value.code == "provider_error"
    assert response.closed is True


def test_lean_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        OpenAITextToSpeechLean(timeout=151)
    with pytest.raises(ValueError):
        OpenAITextToSpeechLean(max_retries=3)
