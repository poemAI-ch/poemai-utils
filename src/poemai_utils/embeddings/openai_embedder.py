import logging

import numpy as np
from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.openai.openai_model import API_TYPE, OPENAI_MODEL

_logger = logging.getLogger(__name__)


class OpenAIEmbedder(EmbedderBase):
    def __init__(self, model_name="text-embedding-ada-002", openai_api_key=None):
        _logger.info(f"Start initializing OpenAIEmbedder with model {model_name}")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "You must install openai to use this function. Try: pip install openai"
            )

        super().__init__()

        self.model_name = model_name

        try:
            openai_model_id_enum = OPENAI_MODEL(model_name)
        except ValueError:
            raise ValueError(f"Unknown model_id: {model_name}")

        if API_TYPE.EMBEDDINGS not in openai_model_id_enum.api_types:

            raise ValueError(f"Model {model_name} does not support embeddings")

        self.openai_model = openai_model_id_enum
        if openai_api_key is not None:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = OpenAI()

        _logger.info(f"Initialized OpenAIEmbedder with model {model_name}")

    def calc_embedding(self, text, is_query: bool = False):
        response = self.client.embeddings.create(input=text, model=self.model_name)
        embedding = response.data[0].embedding
        embedding = np.array(embedding, dtype=np.float32)
        return embedding

    def embedding_dim(self):
        return self.openai_model.embeddings_dimensions
