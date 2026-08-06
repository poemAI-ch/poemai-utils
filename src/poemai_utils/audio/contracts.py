from dataclasses import dataclass
from math import isfinite
from numbers import Number

from poemai_utils.audio.openai_speech_model import (
    DEFAULT_OPENAI_SPEECH_MODEL,
    OPENAI_SPEECH_MODEL,
    resolve_openai_speech_model_key,
)
from poemai_utils.audio.speech_output_format import (
    DEFAULT_SPEECH_OUTPUT_FORMAT,
    SpeechOutputFormat,
    resolve_speech_output_format,
)


def _require_non_empty_string(value, name):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty or whitespace-only")
    return value


@dataclass(frozen=True)
class TextToSpeechRequest:
    text: str
    model: object = DEFAULT_OPENAI_SPEECH_MODEL
    voice: str = "alloy"
    output_format: object = DEFAULT_SPEECH_OUTPUT_FORMAT
    instructions: str | None = None
    speed: Number | None = None

    def __post_init__(self):
        _require_non_empty_string(self.text, "text")
        model_key = _require_non_empty_string(
            resolve_openai_speech_model_key(self.model), "model"
        )
        _require_non_empty_string(self.voice, "voice")
        output_format = resolve_speech_output_format(self.output_format)

        known_model = None
        try:
            known_model = OPENAI_SPEECH_MODEL.by_model_key(model_key)
        except ValueError:
            pass

        if self.instructions is not None:
            _require_non_empty_string(self.instructions, "instructions")
            if known_model is not None and not known_model.supports_instructions:
                raise ValueError(f"Model {model_key} does not support instructions")

        if self.speed is not None:
            if isinstance(self.speed, bool) or not isinstance(self.speed, Number):
                raise TypeError("speed must be numeric")
            try:
                speed = float(self.speed)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("speed must be numeric") from exc
            if not isfinite(speed):
                raise ValueError("speed must be finite")
            if not 0.25 <= speed <= 4.0:
                raise ValueError("speed must be between 0.25 and 4.0")
            if known_model is not None and not known_model.supports_speed:
                raise ValueError(f"Model {model_key} does not support speed")

        object.__setattr__(self, "model", model_key)
        object.__setattr__(self, "output_format", output_format)


@dataclass(frozen=True)
class TextToSpeechResult:
    audio_bytes: bytes
    provider: str
    model_key: str
    voice: str
    output_format: SpeechOutputFormat
    provider_request_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.audio_bytes, bytes) or not self.audio_bytes:
            raise ValueError("audio_bytes must contain non-empty bytes")
        _require_non_empty_string(self.provider, "provider")
        _require_non_empty_string(self.model_key, "model_key")
        _require_non_empty_string(self.voice, "voice")
        if self.provider_request_id is not None:
            _require_non_empty_string(self.provider_request_id, "provider_request_id")
        object.__setattr__(
            self, "output_format", resolve_speech_output_format(self.output_format)
        )

    @property
    def audio(self):
        return self.audio_bytes

    @property
    def model(self):
        return self.model_key

    @property
    def request_id(self):
        return self.provider_request_id

    @property
    def provider_name(self):
        return self.provider

    @property
    def byte_length(self):
        return len(self.audio_bytes)

    @property
    def content_type(self):
        return self.output_format.content_type

    @property
    def file_extension(self):
        return self.output_format.file_extension
