from numpy import deprecate
from poemai_utils.ai_model import AIApiType
from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.embeddings.openai_embedder import OpenAIEmbedder
from poemai_utils.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbeddingModel,
)
from poemai_utils.openai.openai_model import OPENAI_MODEL


@deprecate
def make_embedder(model_id: str, **kwargs) -> EmbedderBase:
    try:
        transformer_args = {
            k: v for k, v in kwargs.items() if k in ["use_cosine_similarity"]
        }
        model_id_enum = SentenceTransformerEmbeddingModel(model_id)
        return SentenceTransformerEmbedder(model_id_enum, **transformer_args)
    except ValueError:
        pass

    openai_model_id_enum = None
    try:
        openai_model_id_enum = OPENAI_MODEL.by_model_key(model_id)
    except ValueError:
        pass
    if openai_model_id_enum is not None:
        openai_args = {k: v for k, v in kwargs.items() if k in ["openai_api_key"]}
        if AIApiType.EMBEDDINGS in openai_model_id_enum.api_types:
            return OpenAIEmbedder(openai_model_id_enum, **openai_args)
        else:
            raise ValueError(f"Model {model_id} does not support embeddings")
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
