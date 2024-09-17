import numpy as np
from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.embeddings.embedding_cache import EmbeddingCache


class EmbeddingStore:
    def __init__(self, embedding_cache=None, embedder=None):
        self.embedder = embedder
        if self.embedder is None:
            raise ValueError("embedder must be provided")
        if not isinstance(self.embedder, EmbedderBase):
            raise ValueError("embedder must be an EmbedderBase object")

        self.texts = []
        self.embedding_matrix = None
        self.embedding_cache = embedding_cache
        if self.embedding_cache is not None:
            if not isinstance(self.embedding_cache, EmbeddingCache):
                raise ValueError("embedding_cache must be an EmbeddingCache object")

    def _store_embedding(self, embedding):
        if self.embedding_matrix is None:
            self.embedding_matrix = np.expand_dims(embedding, 0)
        else:
            self.embedding_matrix = np.concatenate(
                [self.embedding_matrix, [embedding]], 0
            )

    def add_text(self, text, is_query: bool = False):
        self.texts.append(text)
        self._store_embedding(self._create_embedding(text, is_query=is_query))
        return len(self.texts) - 1

    def add_embedding(self, text, embedding):
        self.texts.append(text)
        self._store_embedding(embedding)
        return len(self.texts) - 1

    def query_by_text(self, query, k=5):
        if self.embedding_matrix is None:
            return []
        query_embedding = self._create_embedding(query, is_query=True)
        embedding_matrix = self._scaled_embedding_matrix()

        scores = embedding_matrix.dot(query_embedding)
        idxs = np.argsort(scores)[-k:][::-1]

        return [(self.texts[i], scores[i], i) for i in idxs]

    def _scaled_embedding_matrix(self):
        if self.embedding_matrix is None:
            return None
        if self.embedder.use_cosine_similarity:
            return self.embedding_matrix / np.linalg.norm(
                self.embedding_matrix, ord=2, axis=1, keepdims=True
            )
        else:
            return self.embedding_matrix

    def query_by_embedding(self, query_embedding, k=5):
        if self.embedding_matrix is None:
            return []

        embedding_matrix = self._scaled_embedding_matrix()
        scores = embedding_matrix.dot(query_embedding)

        scores = self.embedding_matrix.dot(query_embedding)
        idxs = np.argsort(scores)[-k:][::-1]

        return [(self.texts[i], scores[i], i) for i in idxs]

    def _create_embedding(self, text, is_query):
        if self.embedding_cache is not None:
            embedding = self.embedding_cache.get(text, self.embedder.model_name)
            if embedding is not None:
                return embedding

        embedding = self.embedder.calc_embedding(text, is_query=is_query)

        if self.embedding_cache is not None:
            self.embedding_cache.put(text, embedding, self.embedder.model_name)

        return embedding

    def similarity(self, index_1, index_2):
        if self.embedding_matrix is None:
            return None

        embedding_matrix = self._scaled_embedding_matrix()

        score = embedding_matrix[index_1].dot(embedding_matrix[index_2])
        return score
