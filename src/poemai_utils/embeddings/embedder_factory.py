from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.embeddings.openai_embedder import OpenAIEmbedder
from poemai_utils.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbeddingModel,
)
from poemai_utils.openai.openai_model import API_TYPE, OPENAI_MODEL


def make_embedder(model_id: str, **kwargs) -> EmbedderBase:
    try:
        model_id_enum = SentenceTransformerEmbeddingModel(model_id)
        return SentenceTransformerEmbedder(model_id_enum, **kwargs)
    except ValueError:
        pass

    openai_model_id_enum = None
    try:
        openai_model_id_enum = OPENAI_MODEL.by_model_key(model_id)
    except ValueError:
        pass
    if openai_model_id_enum is not None:
        if API_TYPE.EMBEDDINGS in openai_model_id_enum.api_types:
            return OpenAIEmbedder(openai_model_id_enum, **kwargs)
        else:
            raise ValueError(f"Model {model_id} does not support embeddings")
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
