from poemai_utils.embeddings.sentence_transformer_embedding_model import (
    SentenceTransformerEmbeddingModel,
)
from poemai_utils.enum_utils import add_enum_repr, merge_enums
from poemai_utils.openai.openai_model import API_TYPE, OPENAI_MODEL


def _member_filter(x):
    if not hasattr(x, "api_types"):
        return True
    return API_TYPE.EMBEDDINGS in x.api_types


EmbeddingModelIdentifier = merge_enums(
    OPENAI_MODEL,
    SentenceTransformerEmbeddingModel,
    name="EmbeddingModelIdentifier",
    base=str,
    module=__name__,
    fields=["model_key"],
    member_filter=_member_filter,
    original_enum_member_field_name="original_enum",
)

add_enum_repr(EmbeddingModelIdentifier)


def find_embedding_model_identifier(model_key: str):
    for model in EmbeddingModelIdentifier:
        if "." in model_key:
            model_key = model_key.split(".")[-1]

        if model.model_key == model_key:
            return model
        if model.value == model_key:
            return model

        if model.name == model_key:
            return model

        if model.name.lower() == model_key.lower():
            return model

    raise ValueError(f"Model key {model_key} not found in EmbeddingModelIdentifier")
