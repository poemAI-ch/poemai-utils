import pytest
from poemai_utils.audio import (
    DEFAULT_OPENAI_SPEECH_MODEL,
    OPENAI_SPEECH_MODEL,
    resolve_openai_speech_model_key,
)


def test_default_is_gpt_4o_mini_tts():
    assert DEFAULT_OPENAI_SPEECH_MODEL == OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS


def test_gpt_4o_mini_tts_metadata():
    assert OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS.model_key == "gpt-4o-mini-tts"
    assert OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS.supports_instructions is True
    assert OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS.supports_speed is True
    assert OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS.deprecated is False


def test_tts_1_metadata():
    assert OPENAI_SPEECH_MODEL.TTS_1.model_key == "tts-1"
    assert OPENAI_SPEECH_MODEL.TTS_1.supports_instructions is False
    assert OPENAI_SPEECH_MODEL.TTS_1.supports_speed is True
    assert OPENAI_SPEECH_MODEL.TTS_1.deprecated is False


def test_tts_1_hd_metadata():
    assert OPENAI_SPEECH_MODEL.TTS_1_HD.model_key == "tts-1-hd"
    assert OPENAI_SPEECH_MODEL.TTS_1_HD.supports_instructions is False
    assert OPENAI_SPEECH_MODEL.TTS_1_HD.supports_speed is True
    assert OPENAI_SPEECH_MODEL.TTS_1_HD.deprecated is False


def test_by_model_key_with_prefix():
    result = OPENAI_SPEECH_MODEL.by_model_key("openai.tts-1")
    assert result == OPENAI_SPEECH_MODEL.TTS_1


def test_by_model_key_without_prefix():
    result = OPENAI_SPEECH_MODEL.by_model_key("tts-1-hd")
    assert result == OPENAI_SPEECH_MODEL.TTS_1_HD


def test_by_model_key_unknown_raises_value_error():
    with pytest.raises(ValueError):
        OPENAI_SPEECH_MODEL.by_model_key("unknown-model")


def test_resolve_openai_speech_model_enum():
    assert resolve_openai_speech_model_key(OPENAI_SPEECH_MODEL.TTS_1) == "tts-1"
    assert resolve_openai_speech_model_key(OPENAI_SPEECH_MODEL.TTS_1_HD) == "tts-1-hd"


def test_resolve_openai_speech_model_string():
    unknown_key = "unknown-model-key"
    assert resolve_openai_speech_model_key(unknown_key) == unknown_key


def test_resolve_openai_speech_model_rejects_none():
    with pytest.raises(TypeError):
        resolve_openai_speech_model_key(None)


def test_resolve_openai_speech_model_rejects_int():
    with pytest.raises(TypeError):
        resolve_openai_speech_model_key(123)


def test_imports_from_poemai_utils_audio():
    from poemai_utils.audio import (
        DEFAULT_OPENAI_SPEECH_MODEL,
        OPENAI_SPEECH_MODEL,
        resolve_openai_speech_model_key,
    )

    assert DEFAULT_OPENAI_SPEECH_MODEL == OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS
    assert OPENAI_SPEECH_MODEL == OPENAI_SPEECH_MODEL
    assert resolve_openai_speech_model_key == resolve_openai_speech_model_key
