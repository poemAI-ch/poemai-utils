from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.embeddings.openai_embedder import OpenAIEmbedder
from poemai_utils.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbeddingModel,
)


def make_embedder(model_id: str, **kwargs) -> EmbedderBase:
    try:
        model_id_enum = SentenceTransformerEmbeddingModel(model_id)
        return SentenceTransformerEmbedder(model_id_enum, **kwargs)
    except ValueError:
        pass
    if model_id == "text-embedding-ada-002":
        return OpenAIEmbedder(model_id, **kwargs)
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
