from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.embeddings.sentence_transformer_embedding_model import (
    SentenceTransformerEmbeddingModel,
)


class SentenceTransformerEmbedder(EmbedderBase):
    # msmarco-distilbert-base-tas-b
    # distiluse-base-multilingual-cased-v1
    # sentence-transformers/LaBSE'
    # svalabs/bi-electra-ms-marco-german-uncased : use_cosine_similarity=True
    def __init__(self, model_id, use_cosine_similarity=False):
        if isinstance(model_id, str):
            model_id = SentenceTransformerEmbeddingModel(model_id)
        super().__init__(use_cosine_similarity=model_id.use_cosine_similarity)

        self.model_name = model_id.value

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "You must install sentence-transformers to use this function. Try: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_id.value)

    def calc_embedding(self, text, is_query: bool = False):
        return self.model.encode(text, show_progress_bar=False)

    def embedding_dim(self):
        return self.model.embeddings_dimensions
