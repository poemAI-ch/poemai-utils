import logging
import os
from pathlib import Path

import pytest
from poemai_utils.audio import (
    DEFAULT_OPENAI_SPEECH_MODEL,
    OpenAITextToSpeech,
    SpeechOutputFormat,
    TextToSpeechRequest,
)

_logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.external
def test_openai_text_to_speech_real_api_generates_audio():
    """Verify that the OpenAI text-to-speech adapter generates real audio."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available for integration test")

    request = TextToSpeechRequest(
        "This is a real text-to-speech integration test.",
        model=DEFAULT_OPENAI_SPEECH_MODEL,
        voice="alloy",
        output_format=SpeechOutputFormat.WAV,
    )

    result = OpenAITextToSpeech(api_key=api_key).synthesize(request)

    audio_path = Path(__file__).with_name("artifacts") / "openai_tts_integration.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(result.audio_bytes)
    _logger.info(
        "Generated OpenAI TTS audio; open this file to listen: %s (%d bytes)",
        audio_path.resolve(),
        len(result.audio_bytes),
    )

    assert result.provider == "openai"
    assert result.model_key == DEFAULT_OPENAI_SPEECH_MODEL.model_key
    assert result.voice == request.voice
    assert result.output_format is SpeechOutputFormat.WAV
    assert result.content_type == "audio/wav"
    assert result.file_extension == ".wav"
    assert result.provider_request_id
    assert len(result.audio_bytes) > 0
    assert result.audio_bytes[:4] == b"RIFF"
    assert result.audio_bytes[8:12] == b"WAVE"
