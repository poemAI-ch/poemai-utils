import logging
from types import SimpleNamespace

import numpy as np
import requests
from poemai_utils.ai_model import AIApiType
from poemai_utils.embeddings.embedder_base import EmbedderBase
from poemai_utils.openai.openai_model import OPENAI_MODEL

_logger = logging.getLogger(__name__)


class OpenAIEmbedderLean(EmbedderBase):
    """
    A lean version of an embedding generator using OpenAI's API, optimized for environments
    with limited resources, such as AWS Lambda. The `OpenAIEmbedderLean` class directly
    interacts with the OpenAI API via `requests`, avoiding the need for the `openai`
    library and its associated dependencies like `httpx`.

    Attributes:
        model_name (str): The OpenAI model name for generating embeddings.
        openai_api_key (str): The API key used to authenticate requests to the OpenAI API.
        base_url (str): The base URL for the OpenAI API (defaults to "https://api.openai.com/v1").

    Methods:
        calc_embedding(text, is_query=False): Generates an embedding for the given text.
        embedding_dim(): Returns the dimensionality of the model's embeddings.

    Advantages:
        - **Reduced Dependency Overhead**: Unlike other embedding classes that rely on the `openai` package,
          this lean version uses `requests` to minimize dependencies. This reduction leads to smaller package
          sizes, faster cold start times, and improved deployment efficiency, especially suitable for
          serverless environments like AWS Lambda.
        - **Customizable Base URL**: The `base_url` parameter allows for flexibility when interacting with
          alternative OpenAI endpoints, such as those hosted in a specific cloud region or within private networks.
        - **Error Logging**: Provides clear logging for any API request errors, making debugging and monitoring
          easier in production environments.

    Usage:
        This lean version is ideal for applications that:
        - Run in restricted environments, where reducing dependencies and package size is critical (e.g., AWS Lambda).
        - Require only the core functionality of generating embeddings from text, without the need for additional
          features of the `openai` package.
        - Benefit from faster response times and reduced deployment sizes by removing dependency-heavy packages.

    Example:
        ```python
        embedder = OpenAIEmbedderLean(
            model_name="text-embedding-ada-002",
            openai_api_key="your_openai_api_key"
        )
        embedding = embedder.calc_embedding("Example text")
        print(embedding)
        ```

    """

    def __init__(
        self, model_name="text-embedding-ada-002", openai_api_key=None, base_url=None
    ):
        _logger.info(f"Start initializing OpenAIEmbedder with model {model_name}")

        super().__init__()

        self.model_name = model_name
        self.openai_api_key = openai_api_key

        # Construct base URL for the API if not provided
        if base_url is None:
            base_url = "https://api.openai.com/v1"

        self.base_url = base_url

        openai_model_id_enum = None
        # Validate model and retrieve dimension info from poemai_utils
        try:
            openai_model_id_enum = OPENAI_MODEL(model_name)
        except ValueError:
            pass

        if openai_model_id_enum is None:
            for model in OPENAI_MODEL:
                if model_name == model.name:
                    openai_model_id_enum = model
                    break

        if openai_model_id_enum is None:
            raise ValueError(f"Model {model_name} not found in OpenAI models")

        self.model_key = openai_model_id_enum.model_key

        if AIApiType.EMBEDDINGS not in openai_model_id_enum.api_types:
            raise ValueError(f"Model {model_name} does not support embeddings")

        self.embeddings_dimensions = openai_model_id_enum.embeddings_dimensions

        _logger.info(f"Initialized OpenAIEmbedder with model {model_name}")

    def calc_embedding(self, text, is_query: bool = False):
        # API endpoint for creating embeddings
        url = f"{self.base_url}/embeddings"

        # Headers for OpenAI API request
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        # Payload for the embeddings request
        data = {"input": text, "model": self.model_key}

        # Perform the request
        response = requests.post(url, headers=headers, json=data)

        # Check for request errors
        if response.status_code != 200:
            _logger.error(f"Failed to retrieve embedding: {response.text}")
            raise RuntimeError(f"Failed to retrieve embedding: {response.status_code}")

        # Parse the response JSON
        response_data = response.json()
        embedding = response_data["data"][0]["embedding"]
        embedding = np.array(embedding, dtype=np.float32)

        return embedding

    def embedding_dim(self):
        return self.embeddings_dimensions
