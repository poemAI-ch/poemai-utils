from enum import Enum

from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr, add_enum_repr_attr


class OPENAI_MODEL(str, Enum):
    GPT_4 = "gpt_4"
    GPT_4_TURBO = "gpt_4_turbo"
    GPT_4_TURBO_1106_PREVIEW = "gpt_4_turbo_1106_preview"
    GPT_4_VISION_PREVIEW = "gpt-4-vision-preview"
    GPT_3_5_TURBO = "gpt_3_5_turbo"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_0613 = "gpt_3_5_turbo_0613"
    GPT_3_5_TURBO_16k = "gpt_3_5_turbo_16k"
    ADA_002_EMBEDDING = "ada_002_embedding"


add_enum_repr_attr(OPENAI_MODEL)


class API_TYPE(Enum):
    COMPLETIONS = "completions"
    CHAT_COMPLETIONS = "chat_completions"
    EMBEDDINGS = "embeddings"
    MODERATIONS = "moderations"


add_enum_repr(API_TYPE)

add_enum_attrs(
    {
        OPENAI_MODEL.GPT_4: {
            "model_key": "gpt-4-0314",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
        },
        OPENAI_MODEL.GPT_4_TURBO: {
            "model_key": "gpt-4-1106-preview",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
        },
        OPENAI_MODEL.GPT_4_TURBO_1106_PREVIEW: {
            "model_key": "gpt-4-1106-preview",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
        },
        OPENAI_MODEL.GPT_4_VISION_PREVIEW: {
            "model_key": "gpt-4-vision-preview",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
        },
        OPENAI_MODEL.GPT_3_5_TURBO: {
            "model_key": "gpt-3.5-turbo",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": False,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_1106: {
            "model_key": "gpt-3.5-turbo-1106",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": False,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_16k: {
            "model_key": "gpt-3.5-turbo-16k",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_0613: {
            "model_key": "gpt-3.5-turbo-0613",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
        },
        OPENAI_MODEL.ADA_002_EMBEDDING: {
            "model_key": "text-embedding-ada-002",
            "api_types": [API_TYPE.EMBEDDINGS],
            "expensive": False,
            "supports_vision": False,
        },
    }
)
