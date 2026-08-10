from poemai_utils.audio.errors import (
    AuthenticationError,
    ConnectionError,
    InvalidRequestError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
    TextToSpeechAuthenticationError,
    TextToSpeechConnectionError,
    TextToSpeechError,
    TextToSpeechInvalidRequestError,
    TextToSpeechProviderError,
    TextToSpeechProviderServerError,
    TextToSpeechRateLimitError,
)
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
from poemai_utils.audio.text_to_speech import (
    OpenAITextToSpeech,
    OpenAITextToSpeechLean,
    TextToSpeechRequest,
    TextToSpeechResult,
)

__all__ = [
    "DEFAULT_OPENAI_SPEECH_MODEL",
    "OPENAI_SPEECH_MODEL",
    "resolve_openai_speech_model_key",
    "DEFAULT_SPEECH_OUTPUT_FORMAT",
    "SpeechOutputFormat",
    "resolve_speech_output_format",
    "TextToSpeechRequest",
    "TextToSpeechResult",
    "OpenAITextToSpeech",
    "OpenAITextToSpeechLean",
    "TextToSpeechError",
    "TextToSpeechInvalidRequestError",
    "TextToSpeechAuthenticationError",
    "TextToSpeechRateLimitError",
    "TextToSpeechConnectionError",
    "TextToSpeechProviderServerError",
    "TextToSpeechProviderError",
    "InvalidRequestError",
    "AuthenticationError",
    "RateLimitError",
    "ConnectionError",
    "ProviderServerError",
    "ProviderError",
]
