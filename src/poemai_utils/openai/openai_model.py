from enum import Enum

from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr, add_enum_repr_attr


class OPENAI_MODEL(Enum):
    GPT_4 = "gpt_4"
    GPT_3_5_TURBO = "gpt_3_5_turbo"
    GPT_3 = "gpt_3"
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
        },
        OPENAI_MODEL.GPT_3_5_TURBO: {
            "model_key": "gpt-3.5-turbo",
            "api_types": [API_TYPE.CHAT_COMPLETIONS],
            "expensive": False,
        },
        OPENAI_MODEL.ADA_002_EMBEDDING: {
            "model_key": "text-embedding-ada-002",
            "api_types": [API_TYPE.EMBEDDINGS],
            "expensive": False,
        },
    }
)
