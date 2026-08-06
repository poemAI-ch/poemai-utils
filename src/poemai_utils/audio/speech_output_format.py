from enum import Enum

from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr_attr


class SpeechOutputFormat(str, Enum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"

    @classmethod
    def by_response_format(cls, response_format):
        for fmt in cls:
            if fmt.response_format == response_format:
                return fmt
        raise ValueError(f"Unknown response_format: {response_format}")


add_enum_repr_attr(SpeechOutputFormat)


add_enum_attrs(
    {
        SpeechOutputFormat.MP3: {
            "response_format": "mp3",
            "content_type": "audio/mpeg",
            "file_extension": ".mp3",
        },
        SpeechOutputFormat.OPUS: {
            "response_format": "opus",
            "content_type": "audio/ogg",
            "file_extension": ".opus",
        },
        SpeechOutputFormat.AAC: {
            "response_format": "aac",
            "content_type": "audio/aac",
            "file_extension": ".aac",
        },
        SpeechOutputFormat.FLAC: {
            "response_format": "flac",
            "content_type": "audio/flac",
            "file_extension": ".flac",
        },
        SpeechOutputFormat.WAV: {
            "response_format": "wav",
            "content_type": "audio/wav",
            "file_extension": ".wav",
        },
        SpeechOutputFormat.PCM: {
            "response_format": "pcm",
            "content_type": "application/octet-stream",
            "file_extension": ".pcm",
        },
    }
)


DEFAULT_SPEECH_OUTPUT_FORMAT = SpeechOutputFormat.MP3


def resolve_speech_output_format(value):
    if isinstance(value, SpeechOutputFormat):
        return value
    if isinstance(value, str):
        value_lower = value.lower()
        for fmt in SpeechOutputFormat:
            if fmt.response_format == value_lower:
                return fmt
        raise ValueError(f"Unknown speech output format: {value}")
    raise TypeError(f"Expected SpeechOutputFormat or str, got {type(value).__name__}")
