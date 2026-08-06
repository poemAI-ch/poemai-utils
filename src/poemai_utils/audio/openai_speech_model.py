from enum import Enum

from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr_attr


class OPENAI_SPEECH_MODEL(str, Enum):
    GPT_4O_MINI_TTS = "gpt-4o-mini-tts"
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"

    @classmethod
    def by_model_key(cls, model_key):
        if model_key.startswith("openai."):
            model_key = model_key[7:]
        for model in cls:
            if model.model_key == model_key:
                return model
        raise ValueError(f"Unknown model_key: {model_key}")


add_enum_repr_attr(OPENAI_SPEECH_MODEL)


add_enum_attrs(
    {
        OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS: {
            "model_key": "gpt-4o-mini-tts",
            "supports_instructions": True,
            "supports_speed": True,
            "deprecated": False,
        },
        OPENAI_SPEECH_MODEL.TTS_1: {
            "model_key": "tts-1",
            "supports_instructions": False,
            "supports_speed": True,
            "deprecated": False,
        },
        OPENAI_SPEECH_MODEL.TTS_1_HD: {
            "model_key": "tts-1-hd",
            "supports_instructions": False,
            "supports_speed": True,
            "deprecated": False,
        },
    }
)


DEFAULT_OPENAI_SPEECH_MODEL = OPENAI_SPEECH_MODEL.GPT_4O_MINI_TTS


def resolve_openai_speech_model_key(model):
    if isinstance(model, OPENAI_SPEECH_MODEL):
        return model.model_key
    if isinstance(model, str):
        return model
    raise TypeError(f"Expected OPENAI_SPEECH_MODEL or str, got {type(model).__name__}")
