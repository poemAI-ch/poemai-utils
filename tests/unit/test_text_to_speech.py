import pytest
from poemai_utils.audio import (
    DEFAULT_OPENAI_SPEECH_MODEL,
    DEFAULT_SPEECH_OUTPUT_FORMAT,
    OPENAI_SPEECH_MODEL,
    OpenAITextToSpeech,
    SpeechOutputFormat,
    TextToSpeechProviderError,
    TextToSpeechRateLimitError,
    TextToSpeechRequest,
    TextToSpeechResult,
)


class FakeResponse:
    def __init__(self, audio=b"audio", request_id="request-1"):
        self.audio = audio
        self.request_id = request_id
        self.closed = False

    def read(self):
        return self.audio

    def close(self):
        self.closed = True


class FakeSpeech:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, speech):
        self.audio = type("Audio", (), {"speech": speech})()


class NestedResponse(FakeResponse):
    def __init__(self, audio=b"audio", headers=None):
        super().__init__(audio=audio, request_id=None)
        self.response = type("HTTPResponse", (), {"headers": headers or {}})()


def test_request_defaults_and_normalization():
    request = TextToSpeechRequest("hello")
    assert request.model == DEFAULT_OPENAI_SPEECH_MODEL.model_key
    assert request.voice == "alloy"
    assert request.output_format == DEFAULT_SPEECH_OUTPUT_FORMAT
    assert request.instructions is None
    assert request.speed is None


def test_request_accepts_enum_and_raw_model():
    assert TextToSpeechRequest("x", model=OPENAI_SPEECH_MODEL.TTS_1).model == "tts-1"
    assert (
        TextToSpeechRequest("x", model="future-tts-model").model == "future-tts-model"
    )


@pytest.mark.parametrize("value", [None, "", "  "])
def test_request_rejects_empty_text(value):
    with pytest.raises((TypeError, ValueError)):
        TextToSpeechRequest(value)


@pytest.mark.parametrize("speed", [0.25, 4.0])
def test_speed_boundaries_are_inclusive(speed):
    assert TextToSpeechRequest("x", speed=speed).speed == speed


@pytest.mark.parametrize("speed", [0.24, 4.01, True, "1", None])
def test_invalid_speed_values(speed):
    if speed is None:
        return
    with pytest.raises((TypeError, ValueError)):
        TextToSpeechRequest("x", speed=speed)


def test_legacy_models_reject_instructions():
    with pytest.raises(ValueError):
        TextToSpeechRequest("x", model="tts-1", instructions="warmly")


def test_output_format_metadata_and_result_properties():
    for output_format in SpeechOutputFormat:
        result = TextToSpeechResult(b"abc", "openai", "model", "alloy", output_format)
        assert result.byte_length == 3
        assert result.content_type == output_format.content_type
        assert result.file_extension == output_format.file_extension


def test_basic_openai_arguments_and_response_lifecycle():
    response = FakeResponse()
    speech = FakeSpeech(response)
    OpenAITextToSpeech(client=FakeClient(speech)).synthesize(
        TextToSpeechRequest("hello")
    )
    assert speech.kwargs == {
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": "hello",
        "response_format": "mp3",
    }
    assert response.closed is True


def test_optional_arguments_are_sent_only_when_supplied():
    speech = FakeSpeech()
    request = TextToSpeechRequest("hello", instructions="calm", speed=1.5)
    OpenAITextToSpeech(client=FakeClient(speech)).synthesize(request)
    assert speech.kwargs["instructions"] == "calm"
    assert speech.kwargs["speed"] == 1.5


@pytest.mark.parametrize("output_format", list(SpeechOutputFormat))
def test_all_supported_output_formats_are_sent(output_format):
    speech = FakeSpeech()
    request = TextToSpeechRequest("hello", output_format=output_format)
    OpenAITextToSpeech(client=FakeClient(speech)).synthesize(request)
    assert speech.kwargs["response_format"] == output_format.response_format


def test_result_captures_provider_request_id():
    result = OpenAITextToSpeech(
        client=FakeClient(FakeSpeech(FakeResponse(request_id="r1")))
    ).synthesize(TextToSpeechRequest("hello"))
    assert result.audio == b"audio"
    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini-tts"
    assert result.request_id == "r1"


def test_result_has_canonical_fields_and_compatibility_aliases():
    result = TextToSpeechResult(
        audio_bytes=b"abc",
        provider="openai",
        model_key="model",
        voice="alloy",
        output_format=SpeechOutputFormat.MP3,
        provider_request_id="r1",
    )
    assert result.audio == result.audio_bytes == b"abc"
    assert result.model == result.model_key == "model"
    assert result.request_id == result.provider_request_id == "r1"


def test_nested_http_response_request_id_is_captured():
    response = NestedResponse(headers={"x-request-id": "nested-r1"})
    result = OpenAITextToSpeech(client=FakeClient(FakeSpeech(response))).synthesize(
        TextToSpeechRequest("hello")
    )
    assert result.provider_request_id == "nested-r1"


@pytest.mark.parametrize("audio", [ValueError("bad audio"), RuntimeError("read")])
def test_response_is_closed_when_result_validation_fails(audio):
    response = FakeResponse(audio=audio)
    with pytest.raises(TextToSpeechProviderError):
        OpenAITextToSpeech(client=FakeClient(FakeSpeech(response))).synthesize(
            TextToSpeechRequest("hello")
        )
    assert response.closed is True


def test_unexpected_provider_exception_is_translated_and_chained():
    error = RuntimeError("provider failure")
    with pytest.raises(TextToSpeechProviderError) as raised:
        OpenAITextToSpeech(client=FakeClient(FakeSpeech(error=error))).synthesize(
            TextToSpeechRequest("hello")
        )
    assert raised.value.code == "provider_error"
    assert raised.value.retryable is False
    assert raised.value.__cause__ is error


def test_rate_limit_is_retryable_and_chained():
    error = type("RateLimitError", (Exception,), {})()
    with pytest.raises(TextToSpeechRateLimitError) as raised:
        OpenAITextToSpeech(client=FakeClient(FakeSpeech(error=error))).synthesize(
            TextToSpeechRequest("hello")
        )
    assert raised.value.code == "rate_limited"
    assert raised.value.retryable is True
    assert raised.value.__cause__ is error


def test_input_longer_than_openai_limit_is_rejected_before_call():
    speech = FakeSpeech()
    with pytest.raises(ValueError, match="4096"):
        OpenAITextToSpeech(client=FakeClient(speech)).synthesize(
            TextToSpeechRequest("x" * 4097)
        )
    assert speech.kwargs is None


def test_adapter_enum_is_public():
    assert OpenAITextToSpeech.OPENAI_SPEECH_MODEL is OPENAI_SPEECH_MODEL


def test_client_is_constructed_with_explicit_api_key_and_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    adapter = OpenAITextToSpeech(api_key="secret", base_url="https://example.test")
    assert captured == {"api_key": "secret", "base_url": "https://example.test"}
    assert isinstance(adapter.client, FakeOpenAI)


def test_client_construction_omits_unspecified_configuration(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    OpenAITextToSpeech()
    assert captured == {}


def test_client_construction_passes_timeout_and_retries(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    OpenAITextToSpeech(api_key="secret", timeout=12.5, max_retries=2)
    assert captured == {"api_key": "secret", "timeout": 12.5, "max_retries": 2}


def test_injected_client_rejects_explicit_configuration():
    with pytest.raises(ValueError, match="client"):
        OpenAITextToSpeech(client=FakeClient(FakeSpeech()), api_key="secret")
