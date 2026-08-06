class TextToSpeechError(Exception):
    """Base error for provider-neutral text-to-speech failures."""

    code = "text_to_speech_error"
    retryable = False

    def __init__(self, message=None):
        super().__init__(message or self.code)


class TextToSpeechInvalidRequestError(TextToSpeechError, ValueError):
    code = "invalid_request"


class TextToSpeechAuthenticationError(TextToSpeechError):
    code = "authentication_failed"


class TextToSpeechRateLimitError(TextToSpeechError):
    code = "rate_limited"
    retryable = True


class TextToSpeechConnectionError(TextToSpeechError):
    code = "connection_failed"
    retryable = True


class TextToSpeechProviderServerError(TextToSpeechError):
    code = "provider_server_error"
    retryable = True


class TextToSpeechProviderError(TextToSpeechError):
    """Unexpected provider failure; conservatively treated as terminal."""

    code = "provider_error"


# Short aliases make the public contract convenient without exposing SDK types.
InvalidRequestError = TextToSpeechInvalidRequestError
AuthenticationError = TextToSpeechAuthenticationError
RateLimitError = TextToSpeechRateLimitError
ConnectionError = TextToSpeechConnectionError
ProviderServerError = TextToSpeechProviderServerError
ProviderError = TextToSpeechProviderError
