import numpy as np
import pytest
from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.embeddings.embedder_factory import make_embedder
from poemai_utils.embeddings.embedding_store import EmbeddingStore
from poemai_utils.embeddings.openai_embedder import OpenAIEmbedder
from poemai_utils.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)

try:
    from poemai_utils.embeddings.sgpt_embedder import SGPTEmbedder

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_embedding():
    class MockEmbedder(EmbedderBase):
        def calc_embedding(self, text, is_query: bool = False):
            embedding = [ord(c) for c in text[:4]]
            if len(embedding) < 4:
                embedding += [0] * (4 - len(embedding))
            return np.array(embedding, dtype=np.float32)

    embedding_store = EmbeddingStore(embedder=MockEmbedder())
    embedding_store.add_text("abc")


try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def test_embedding_factory():

    openai_embedder = make_embedder("text-embedding-ada-002", openai_api_key="test-key")

    assert isinstance(openai_embedder, OpenAIEmbedder)

    if HAS_SENTENCE_TRANSFORMERS:
        labse_embedder = make_embedder("sentence-transformers/LaBSE")
        assert isinstance(labse_embedder, SentenceTransformerEmbedder)

        distiluse_embedder = make_embedder("distiluse-base-multilingual-cased-v1")
        assert isinstance(distiluse_embedder, SentenceTransformerEmbedder)

        bi_electra_german_embedder = make_embedder(
            "svalabs/bi-electra-ms-marco-german-uncased"
        )
        assert isinstance(bi_electra_german_embedder, SentenceTransformerEmbedder)
    else:
        pytest.skip("sentence-transformers not available, skipping transformer tests")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_sgpt_embedder():
    sgpt_embedder = SGPTEmbedder()
    assert sgpt_embedder.embedding_dim() == 768
