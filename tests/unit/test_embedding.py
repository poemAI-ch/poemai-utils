import numpy as np
from build.lib.poemai_utils.embeddings.embedding_store import EmbeddingStore
from poemai_utils.embeddings.embedder_base import EbedderBase


def test_embedding():
    class MockEmbedder(EbedderBase):
        def calc_embedding(self, text, is_query: bool = False):
            embedding = [ord(c) for c in text[:4]]
            if len(embedding) < 4:
                embedding += [0] * (4 - len(embedding))
            return np.array(embedding, dtype=np.float32)

    embedding_store = EmbeddingStore(embedder=MockEmbedder())
    embedding_store.add_text("abc")
