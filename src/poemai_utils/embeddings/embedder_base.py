class EmbedderBase:
    def __init__(self, use_cosine_similarity=False):
        self.use_cosine_similarity = use_cosine_similarity

    def embedding_dim(self):
        raise NotImplementedError

    def calc_embedding_batch(self, texts, is_query=False):
        """
        Generate embeddings for a batch of texts.
        Default implementation calls calc_embedding for each text individually.
        Subclasses should override this for more efficient batch processing.

        Args:
            texts (list): List of text strings to embed
            is_query (bool): Whether this is a query embedding

        Returns:
            list: List of embeddings, one per input text
        """
        return [self.calc_embedding(text, is_query=is_query) for text in texts]
