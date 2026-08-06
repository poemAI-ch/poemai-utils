import pytest

from poemai_utils.audio import (
    DEFAULT_OPENAI_SPEECH_MODEL,
    DEFAULT_SPEECH_OUTPUT_FORMAT,
    OPENAI_SPEECH_MODEL,
    resolve_openai_speech_model_key,
    resolve_speech_output_format,
    SpeechOutputFormat,
)


def test_mp3_is_default():
    assert DEFAULT_SPEECH_OUTPUT_FORMAT == SpeechOutputFormat.MP3


def test_mp3_metadata():
    assert SpeechOutputFormat.MP3.response_format == "mp3"
    assert SpeechOutputFormat.MP3.content_type == "audio/mpeg"
    assert SpeechOutputFormat.MP3.file_extension == ".mp3"


def test_opus_metadata():
    assert SpeechOutputFormat.OPUS.response_format == "opus"
    assert SpeechOutputFormat.OPUS.content_type == "audio/ogg"
    assert SpeechOutputFormat.OPUS.file_extension == ".opus"


def test_aac_metadata():
    assert SpeechOutputFormat.AAC.response_format == "aac"
    assert SpeechOutputFormat.AAC.content_type == "audio/aac"
    assert SpeechOutputFormat.AAC.file_extension == ".aac"


def test_flac_metadata():
    assert SpeechOutputFormat.FLAC.response_format == "flac"
    assert SpeechOutputFormat.FLAC.content_type == "audio/flac"
    assert SpeechOutputFormat.FLAC.file_extension == ".flac"


def test_wav_metadata():
    assert SpeechOutputFormat.WAV.response_format == "wav"
    assert SpeechOutputFormat.WAV.content_type == "audio/wav"
    assert SpeechOutputFormat.WAV.file_extension == ".wav"


def test_pcm_metadata():
    assert SpeechOutputFormat.PCM.response_format == "pcm"
    assert SpeechOutputFormat.PCM.content_type == "application/octet-stream"
    assert SpeechOutputFormat.PCM.file_extension == ".pcm"


def test_by_response_format_all_values():
    assert SpeechOutputFormat.by_response_format("mp3") == SpeechOutputFormat.MP3
    assert SpeechOutputFormat.by_response_format("opus") == SpeechOutputFormat.OPUS
    assert SpeechOutputFormat.by_response_format("aac") == SpeechOutputFormat.AAC
    assert SpeechOutputFormat.by_response_format("flac") == SpeechOutputFormat.FLAC
    assert SpeechOutputFormat.by_response_format("wav") == SpeechOutputFormat.WAV
    assert SpeechOutputFormat.by_response_format("pcm") == SpeechOutputFormat.PCM


def test_by_response_format_unknown_raises_value_error():
    with pytest.raises(ValueError):
        SpeechOutputFormat.by_response_format("unknown-format")


def test_resolve_accepts_enum_member():
    assert resolve_speech_output_format(SpeechOutputFormat.MP3) == SpeechOutputFormat.MP3
    assert resolve_speech_output_format(SpeechOutputFormat.OPUS) == SpeechOutputFormat.OPUS


def test_resolve_accepts_case_insensitive_string():
    assert resolve_speech_output_format("MP3") == SpeechOutputFormat.MP3
    assert resolve_speech_output_format("mp3") == SpeechOutputFormat.MP3
    assert resolve_speech_output_format("MP3") == SpeechOutputFormat.MP3
    assert resolve_speech_output_format("OPUS") == SpeechOutputFormat.OPUS
    assert resolve_speech_output_format("opus") == SpeechOutputFormat.OPUS
    assert resolve_speech_output_format("AAC") == SpeechOutputFormat.AAC
    assert resolve_speech_output_format("aac") == SpeechOutputFormat.AAC
    assert resolve_speech_output_format("FLAC") == SpeechOutputFormat.FLAC
    assert resolve_speech_output_format("flac") == SpeechOutputFormat.FLAC
    assert resolve_speech_output_format("WAV") == SpeechOutputFormat.WAV
    assert resolve_speech_output_format("wav") == SpeechOutputFormat.WAV
    assert resolve_speech_output_format("PCM") == SpeechOutputFormat.PCM
    assert resolve_speech_output_format("pcm") == SpeechOutputFormat.PCM


def test_resolve_rejects_unknown_string():
    with pytest.raises(ValueError):
        resolve_speech_output_format("unknown-format")


def test_resolve_rejects_none():
    with pytest.raises(TypeError):
        resolve_speech_output_format(None)


def test_resolve_rejects_int():
    with pytest.raises(TypeError):
        resolve_speech_output_format(123)


def test_all_iteration_1_and_2_exports():
    from poemai_utils.audio import (
        DEFAULT_OPENAI_SPEECH_MODEL,
        DEFAULT_SPEECH_OUTPUT_FORMAT,
        OPENAI_SPEECH_MODEL,
        resolve_openai_speech_model_key,
        resolve_speech_output_format,
        SpeechOutputFormat,
    )

    assert DEFAULT_OPENAI_SPEECH_MODEL == OPENAI_SPEECH_MODEL.TTS_1_HD
