from enum import Enum

from poemai_utils.ai_model import AIModel
from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr


class SentenceTransformerEmbeddingModel(Enum):
    LABSE = "sentence-transformers/LaBSE"
    DISTILUSE = "distiluse-base-multilingual-cased-v1"
    BI_ELECTRA_GERMAN = "svalabs/bi-electra-ms-marco-german-uncased"
    DISTILBERT = "msmarco-distilbert-base-tas-b"

    @classmethod
    def register_ai_models(cls):
        AIModel.add_enum_members(cls, "sentence_transformer")


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
