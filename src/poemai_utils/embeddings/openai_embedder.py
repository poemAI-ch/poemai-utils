import numpy as np
import openai
from poemai_utils.embeddings.embedder_base import EbedderBase


class OpenAIEmbedder(EbedderBase):
    def __init__(self, model_name="text-embedding-ada-002"):
        super().__init__()

        self.model_name = model_name

    def calc_embedding(self, text, is_query: bool = False):
        response = openai.Embedding.create(input=text, model=self.model_name)
        embedding = response["data"][0]["embedding"]
        embedding = np.array(embedding, dtype=np.float32)
        return embedding
