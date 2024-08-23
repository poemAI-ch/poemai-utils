import numpy as np
from poemai_utils.embeddings.embedding_store import EmbeddingStore


class SelfSimilarityAnalyzer:
    def __init__(self, embedder, embedding_cache=None):
        if embedder is None:
            raise ValueError("embedder must be provided")
        self.embedding_store = EmbeddingStore(embedding_cache, embedder=embedder)
        self.use_cosine_similarity = embedder.use_cosine_similarity
        self.metadata = {}

    def add_text(self, text, metadata, is_query=False):
        if text is None:
            return
        text_id = self.embedding_store.add_text(text, is_query=is_query)
        self.metadata[text_id] = metadata

    def add_embedding(self, text, embedding, metadata):
        text_id = self.embedding_store.add_embedding(text, embedding)
        self.metadata[text_id] = metadata

    def query(self, query, k=5):
        top_k = self.embedding_store.query_by_text(query, k)
        return [
            (int(text_id), float(score), self.metadata.get(text_id))
            for _, score, text_id in top_k
        ]

    @staticmethod
    def normalize_rows(x):
        return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    def similarity_matrix(self):
        m1 = self.embedding_store.embedding_matrix
        if self.use_cosine_similarity:
            m1 = self.normalize_rows(m1)

        return np.matmul(
            m1,
            m1.T,
        )

    def find_top_k_similar_pairs(self, k=5):
        similarity_matrix = self.similarity_matrix()

        similarity_matrix = np.tril(similarity_matrix)
        np.fill_diagonal(similarity_matrix, 0)

        # find the top k most similar sections
        top_k_idxs = np.argsort(similarity_matrix, axis=None)[-k:]
        rows, cols = np.unravel_index(top_k_idxs, similarity_matrix.shape)
        values = similarity_matrix[rows, cols]

        retval = []
        for row, col, value in zip(rows, cols, values):
            row_metadata = self.metadata.get(row)
            col_metadata = self.metadata.get(col)

            retval.append(
                (
                    row_metadata,
                    col_metadata,
                    value,
                )
            )

        return retval
