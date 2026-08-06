# poemai-utils

This package is a collection of utilities for AI projects.

## Text-to-speech

The audio API resolves a recommended OpenAI model and MP3 output format by
default, while still accepting enum values or raw model keys:

```python
from poemai_utils.audio import (
    DEFAULT_OPENAI_SPEECH_MODEL,
    OpenAITextToSpeech,
    SpeechOutputFormat,
    TextToSpeechRequest,
)

request = TextToSpeechRequest(
    "Welcome to the workshop.",
    model=DEFAULT_OPENAI_SPEECH_MODEL,
    output_format=SpeechOutputFormat.WAV,
)
request = TextToSpeechRequest("A preview.", model="future-openai-tts-model")
result = OpenAITextToSpeech(api_key="...").synthesize(request)

print(result.audio_bytes, result.content_type, result.file_extension)
print(result.provider_request_id)
```

Pass an API key explicitly when constructing the adapter; otherwise the OpenAI
SDK uses its normal environment configuration. Synthesis errors expose a
stable `code` and `retryable` flag.
