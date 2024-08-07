from enum import Enum

from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr, add_enum_repr_attr


class OPENAI_MODEL(str, Enum):
    GPT_4_o = "gpt-4o"
    GPT_4_o_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4_o_MINI = "gpt-4o-mini"
    GPT_4_o_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"

    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4_TURBO = "gpt_4_turbo"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_4_TURBO_1106_PREVIEW = "gpt_4_turbo_1106_preview"
    GPT_4_VISION_PREVIEW = "gpt-4-vision-preview"
    GPT_3_5_TURBO = "gpt_3_5_turbo"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_0613 = "gpt_3_5_turbo_0613"
    GPT_3_5_TURBO_16k = "gpt_3_5_turbo_16k"
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    ADA_002_EMBEDDING = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

    @classmethod
    def by_model_key(cls, model_key):
        if model_key.startswith("openai."):
            model_key = model_key[7:]
        for model in cls:
            if model.model_key == model_key:
                return model
        raise ValueError(f"Unknown model_key: {model_key}")


add_enum_repr_attr(OPENAI_MODEL)


class API_TYPE(Enum):
    COMPLETIONS = "completions"
    CHAT_COMPLETIONS = "chat_completions"
    EMBEDDINGS = "embeddings"
    MODERATIONS = "moderations"


add_enum_repr(API_TYPE)

add_enum_attrs(
    {
        OPENAI_MODEL.GPT_4_o: {
            "model_key": "gpt-4o",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_4_o_2024_05_13: {
            "model_key": "gpt-4o-2024-05-13",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_4_TURBO: {
            "model_key": "gpt-4-0125-preview",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_4_TURBO_2024_04_09: {
            "model_key": "gpt-4-turbo-2024-04-09",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_4_TURBO_PREVIEW: {
            "model_key": "gpt-4-turbo-preview",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_4_0125_PREVIEW: {
            "model_key": "gpt-4-0125-preview",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_4_TURBO_1106_PREVIEW: {
            "model_key": "gpt-4-1106-preview",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_4_VISION_PREVIEW: {
            "model_key": "gpt-4-vision-preview",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_3_5_TURBO: {
            "model_key": "gpt-3.5-turbo",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_1106: {
            "model_key": "gpt-3.5-turbo-1106",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_16k: {
            "model_key": "gpt-3.5-turbo-16k",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_0125: {
            "model_key": "gpt-3.5-turbo-0125",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_0613: {
            "model_key": "gpt-3.5-turbo-0613",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.ADA_002_EMBEDDING: {
            "model_key": "text-embedding-ada-002",
            "api_types": [API_TYPE.EMBEDDINGS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": 1536,
        },
        OPENAI_MODEL.TEXT_EMBEDDING_3_LARGE: {
            "model_key": "text-embedding-3-large",
            "api_types": [API_TYPE.EMBEDDINGS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": 3072,
        },
        OPENAI_MODEL.TEXT_EMBEDDING_3_SMALL: {
            "model_key": "text-embedding-3-small",
            "api_types": [API_TYPE.EMBEDDINGS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": 1536,
        },
        OPENAI_MODEL.GPT_4_o_MINI: {
            "model_key": "gpt-4o-mini",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
        },
        OPENAI_MODEL.GPT_4_o_MINI_2024_07_18: {
            "model_key": "gpt-4o-mini-2024-07-18",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
        },
    }
)
