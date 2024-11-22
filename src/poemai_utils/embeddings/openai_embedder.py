import logging
from types import SimpleNamespace

import numpy as np
from poemai_utils.ai_model import AIApiType
from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.openai.openai_model import OPENAI_MODEL

_logger = logging.getLogger(__name__)


class OpenAIEmbedder(EmbedderBase):
    def __init__(
        self, model_name="text-embedding-ada-002", openai_api_key=None, base_url=None
    ):
        _logger.info(f"Start initializing OpenAIEmbedder with model {model_name}")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "You must install openai to use this function. Try: pip install openai"
            )

        super().__init__()

        if isinstance(model_name, OPENAI_MODEL):
            self.model_name = model_name.model_key
        else:
            self.model_name = model_name

        if base_url is None:
            openai_model_id_enum = None
            try:
                openai_model_id_enum = OPENAI_MODEL(model_name)
            except ValueError:
                pass

            if openai_model_id_enum is None:
                try:
                    openai_model_id_enum = OPENAI_MODEL[model_name]
                    self.model_name = openai_model_id_enum.model_key
                except KeyError:
                    raise ValueError(f"Unknown model name {model_name}")

            _logger.info(f"OpenAI model: {openai_model_id_enum}")
            if AIApiType.EMBEDDINGS not in openai_model_id_enum.api_types:

                raise ValueError(f"Model {model_name} does not support embeddings")

            self.openai_model = openai_model_id_enum
            self.embeddings_dimensions = self.openai_model.embeddings_dimensions

        else:
            self.openai_model = SimpleNamespace(embeddings_dimensions=None)
            self.embeddings_dimensions = None

        openai_args = {}
        if openai_api_key is not None:
            openai_args["api_key"] = str(openai_api_key).strip()

        if base_url is not None:
            _logger.info(f"Using base_url {base_url}")
            openai_args["base_url"] = base_url

        self.client = OpenAI(**openai_args)

        _logger.info(f"Initialized OpenAIEmbedder with model {model_name}")

    def calc_embedding(self, text, is_query: bool = False):
        response = self.client.embeddings.create(input=text, model=self.model_name)
        embedding = response.data[0].embedding
        embedding = np.array(embedding, dtype=np.float32)
        return embedding

    def embedding_dim(self):
        return self.embeddings_dimensions
