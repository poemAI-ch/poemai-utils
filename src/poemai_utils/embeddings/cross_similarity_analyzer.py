import numpy as np
from poemai_utils.embeddings.self_similarity_analyzer import SelfSimilarityAnalyzer


class CrossSimilarityAnalyzer:
    """A class to analyze the similarity between two lists of embeddings."""

    def __init__(
        self,
        embedding_cache=None,
        similarity_analyzer_1=None,
        similarity_analyzer_2=None,
        embedder=None,
    ):
        if similarity_analyzer_1 is None:
            self.analyzer_1 = SelfSimilarityAnalyzer(
                embedder,
                embedding_cache=embedding_cache,
            )
        else:
            self.analyzer_1 = similarity_analyzer_1
        if similarity_analyzer_2 is None:
            self.analyzer_2 = SelfSimilarityAnalyzer(
                embedder, embedding_cache=embedding_cache
            )
        else:
            self.analyzer_2 = similarity_analyzer_2

        if (
            self.analyzer_1.use_cosine_similarity
            != self.analyzer_2.use_cosine_similarity
        ):
            raise ValueError("Both analyzers must have the same use_cosine_similarity")

        self.use_cosine_similarity = self.analyzer_1.use_cosine_similarity

    def add_text_for_1(self, text, metadata):
        self.analyzer_1.add_text(text, metadata)

    def add_embedding_for_1(self, text, embedding, metadata):
        self.analyzer_1.add_embedding(text, embedding, metadata)

    def add_text_for_2(self, text, metadata):
        self.analyzer_2.add_text(text, metadata)

    def add_embedding_for_2(self, text, embedding, metadata):
        self.analyzer_2.add_embedding(text, embedding, metadata)

    @staticmethod
    def normalize_rows(x):
        return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    def cross_similarity_matrix(self):
        m1 = self.analyzer_1.embedding_store.embedding_matrix
        m2 = self.analyzer_2.embedding_store.embedding_matrix

        if self.use_cosine_similarity:
            m1 = self.normalize_rows(m1)
            m2 = self.normalize_rows(m2)

        return np.matmul(
            m1,
            m2.T,
        )

    def find_top_k_similar_pairs(self, k=5):
        similarity_matrix = self.cross_similarity_matrix()

        top_k_idxs = np.argsort(similarity_matrix, axis=None)[-k:]
        rows, cols = np.unravel_index(top_k_idxs, similarity_matrix.shape)
        values = similarity_matrix[rows, cols]

        retval = []
        for row, col, value in zip(rows, cols, values):
            row_metadata = self.analyzer_1.metadata.get(row)
            col_metadata = self.analyzer_2.metadata.get(col)

            retval.append(
                (
                    row_metadata,
                    col_metadata,
                    value,
                )
            )

        return retval

    def map_1_to_2(self, k=1):
        similarity_matrix = self.cross_similarity_matrix()
        best = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]

        retval = []
        for i, row in enumerate(best):
            row_metadata = self.analyzer_1.metadata.get(i)
            col_metadata = [self.analyzer_2.metadata.get(b) for b in row]

            retval.append(
                (
                    row_metadata,
                    col_metadata,
                )
            )
        return retval

    def map_1_to_2_with_scores(self, k=1):
        similarity_matrix = self.cross_similarity_matrix()
        best = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]

        retval = []
        for i, row in enumerate(best):
            row_metadata = self.analyzer_1.metadata.get(i)
            col_metadata = [self.analyzer_2.metadata.get(b) for b in row]

            col_result = list(zip(col_metadata, similarity_matrix[i, row]))
            retval.append(
                (
                    row_metadata,
                    col_result,
                )
            )
        return retval
