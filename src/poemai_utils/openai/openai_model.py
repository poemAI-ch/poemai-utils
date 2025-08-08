from enum import Enum

from poemai_utils.ai_model import AIApiType, AIModel
from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr_attr


class OPENAI_MODEL(str, Enum):
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_2025_04_14 = "gpt-4.1-2025-04-14"
    GPT_4_5_PREVIEW = "gpt-4.5-preview"
    GPT_4_5_PREVIEW_2025_02_27 = "gpt-4.5-preview-2025-02-27"
    GPT_4_o = "gpt-4o"
    GPT_4_o_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4_o_2024_08_06 = "gpt-4o-2024-08-06"
    GPT_4_o_CHATGPT_LATEST = "chatgpt-4o-latest"
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
    O1_PREVIEW = "o1-preview"
    O1_PREVIEW_2024_09_12 = "o1-preview-2024-09-12"
    O1_MINI = "o1-mini"
    O1_MINI_2024_09_12 = "o1-mini-2024-09-12"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5 = "gpt-5"
    GPT_5_MINI_2025_08_07 = "gpt-5-mini-2025-08-07"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO_2025_08_07 = "gpt-5-nano-2025-08-07"
    GPT_5_CHAT_LATEST = "gpt-5-chat-latest"
    GPT_5_2025_08_07 = "gpt-5-2025-08-07"

    @classmethod
    def by_model_key(cls, model_key):
        if model_key.startswith("openai."):
            model_key = model_key[7:]
        for model in cls:
            if model.model_key == model_key:
                return model
        raise ValueError(f"Unknown model_key: {model_key}")

    def calc_model_key(self):
        return "openai." + self.model_key

    @classmethod
    def register_ai_models(cls):
        AIModel.add_enum_members(cls, "openai")


add_enum_repr_attr(OPENAI_MODEL)


add_enum_attrs(
    {
        OPENAI_MODEL.GPT_4_1: {
            "model_key": "gpt-4.1",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_1_2025_04_14: {
            "model_key": "gpt-4.1-2025-04-14",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_5_PREVIEW: {
            "model_key": "gpt-4.5-preview",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_5_PREVIEW_2025_02_27: {
            "model_key": "gpt-4.5-preview-2025-02-27",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_o: {
            "model_key": "gpt-4o",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_o_2024_05_13: {
            "model_key": "gpt-4o-2024-05-13",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_o_2024_08_06: {
            "model_key": "gpt-4o-2024-08-06",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_o_CHATGPT_LATEST: {
            "model_key": "chatgpt-4o-latest",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_TURBO: {
            "model_key": "gpt-4-0125-preview",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_TURBO_2024_04_09: {
            "model_key": "gpt-4-turbo-2024-04-09",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_TURBO_PREVIEW: {
            "model_key": "gpt-4-turbo-preview",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_0125_PREVIEW: {
            "model_key": "gpt-4-0125-preview",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_TURBO_1106_PREVIEW: {
            "model_key": "gpt-4-1106-preview",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_VISION_PREVIEW: {
            "model_key": "gpt-4-vision-preview",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_3_5_TURBO: {
            "model_key": "gpt-3.5-turbo",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_1106: {
            "model_key": "gpt-3.5-turbo-1106",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_16k: {
            "model_key": "gpt-3.5-turbo-16k",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_0125: {
            "model_key": "gpt-3.5-turbo-0125",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_3_5_TURBO_0613: {
            "model_key": "gpt-3.5-turbo-0613",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": False,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.ADA_002_EMBEDDING: {
            "model_key": "text-embedding-ada-002",
            "api_types": [AIApiType.EMBEDDINGS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": 1536,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.TEXT_EMBEDDING_3_LARGE: {
            "model_key": "text-embedding-3-large",
            "api_types": [AIApiType.EMBEDDINGS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": 3072,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.TEXT_EMBEDDING_3_SMALL: {
            "model_key": "text-embedding-3-small",
            "api_types": [AIApiType.EMBEDDINGS],
            "expensive": False,
            "supports_vision": False,
            "embeddings_dimensions": 1536,
            "requires_max_completion_tokens": False,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_o_MINI: {
            "model_key": "gpt-4o-mini",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_4_o_MINI_2024_07_18: {
            "model_key": "gpt-4o-mini-2024-07-18",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.O1_PREVIEW: {
            "model_key": "o1-preview",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.O1_PREVIEW_2024_09_12: {
            "model_key": "o1-preview-2024-09-12",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.O1_MINI: {
            "model_key": "o1-mini",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.O1_MINI_2024_09_12: {
            "model_key": "o1-mini-2024-09-12",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": True,
        },
        OPENAI_MODEL.GPT_5_NANO: {
            "model_key": "gpt-5-nano",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": False,
        },
        OPENAI_MODEL.GPT_5: {
            "model_key": "gpt-5",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": False,
        },
        OPENAI_MODEL.GPT_5_MINI_2025_08_07: {
            "model_key": "gpt-5-mini-2025-08-07",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": False,
        },
        OPENAI_MODEL.GPT_5_MINI: {
            "model_key": "gpt-5-mini",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": False,
        },
        OPENAI_MODEL.GPT_5_NANO_2025_08_07: {
            "model_key": "gpt-5-nano-2025-08-07",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": False,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": False,
        },
        OPENAI_MODEL.GPT_5_CHAT_LATEST: {
            "model_key": "gpt-5-chat-latest",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": False,
        },
        OPENAI_MODEL.GPT_5_2025_08_07: {
            "model_key": "gpt-5-2025-08-07",
            "api_types": [AIApiType.CHAT_COMPLETIONS],
            "expensive": True,
            "supports_vision": True,
            "embeddings_dimensions": None,
            "requires_max_completion_tokens": True,
            "supports_temperature": False,
        },
    }
)
