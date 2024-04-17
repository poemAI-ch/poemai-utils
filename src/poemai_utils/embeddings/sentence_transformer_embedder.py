from enum import Enum

from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingModel(Enum):
    LABSE = "sentence-transformers/LaBSE"
    DISTILUSE = "distiluse-base-multilingual-cased-v1"
    BI_ELECTRA_GERMAN = "svalabs/bi-electra-ms-marco-german-uncased"
    DISTILBERT = "msmarco-distilbert-base-tas-b"


add_enum_attrs(
    {
        SentenceTransformerEmbeddingModel.LABSE: {
            "use_cosine_similarity": False,
            "embeddings_dimensions": 768,
        },
        SentenceTransformerEmbeddingModel.DISTILUSE: {
            "use_cosine_similarity": False,
            "embeddings_dimensions": 768,
        },
        SentenceTransformerEmbeddingModel.BI_ELECTRA_GERMAN: {  # best german embeddings found so far
            "use_cosine_similarity": True,
            "embeddings_dimensions": 768,
        },
        SentenceTransformerEmbeddingModel.DISTILBERT: {
            "use_cosine_similarity": False,
            "embeddings_dimensions": 768,
        },
    }
)
add_enum_repr(SentenceTransformerEmbeddingModel)


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
        self.model = SentenceTransformer(model_id.value)

    def calc_embedding(self, text, is_query: bool = False):
        return self.model.encode(text, show_progress_bar=False)

    def embedding_dim(self):
        return self.model.embeddings_dimensions
