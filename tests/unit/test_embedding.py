import numpy as np
from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.embeddings.embedder_factory import make_embedder
from poemai_utils.embeddings.embedding_store import EmbeddingStore
from poemai_utils.embeddings.openai_embedder import OpenAIEmbedder
from poemai_utils.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from poemai_utils.embeddings.sgpt_embedder import SGPTEmbedder


def test_embedding():
    class MockEmbedder(EmbedderBase):
        def calc_embedding(self, text, is_query: bool = False):
            embedding = [ord(c) for c in text[:4]]
            if len(embedding) < 4:
                embedding += [0] * (4 - len(embedding))
            return np.array(embedding, dtype=np.float32)

    embedding_store = EmbeddingStore(embedder=MockEmbedder())
    embedding_store.add_text("abc")


def test_embedding_factory():

    openai_embedder = make_embedder("text-embedding-ada-002")

    assert isinstance(openai_embedder, OpenAIEmbedder)

    labse_embedder = make_embedder("sentence-transformers/LaBSE")
    assert isinstance(labse_embedder, SentenceTransformerEmbedder)

    distiluse_embedder = make_embedder("distiluse-base-multilingual-cased-v1")
    assert isinstance(distiluse_embedder, SentenceTransformerEmbedder)

    bi_electra_german_embedder = make_embedder(
        "svalabs/bi-electra-ms-marco-german-uncased"
    )
    assert isinstance(bi_electra_german_embedder, SentenceTransformerEmbedder)


def test_sgpt_embedder():
    sgpt_embedder = SGPTEmbedder()
    assert sgpt_embedder.embedding_dim() == 768
