import numpy as np
from poemai_utils.embeddings.embedder_base import EbedderBase


class OpenAIEmbedder(EbedderBase):
    def __init__(self, model_name="text-embedding-ada-002", openai_api_key=None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "You must install openai to use this function. Try: pip install openai"
            )

        super().__init__()

        self.model_name = model_name
        if openai_api_key is not None:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = OpenAI()

    def calc_embedding(self, text, is_query: bool = False):
        response = self.client.embeddings.create(input=text, model=self.model_name)
        embedding = response.data[0].embedding
        embedding = np.array(embedding, dtype=np.float32)
        return embedding
