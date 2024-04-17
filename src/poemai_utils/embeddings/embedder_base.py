class EmbedderBase:
    def __init__(self, use_cosine_similarity=False):
        self.use_cosine_similarity = use_cosine_similarity

    def embedding_dim(self):
        raise NotImplementedError
