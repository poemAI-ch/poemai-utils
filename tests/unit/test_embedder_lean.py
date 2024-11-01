from unittest.mock import MagicMock, patch

import numpy as np
from poemai_utils.embeddings.openai_embedder_lean import OpenAIEmbedderLean


def test_openai_embedder_lean():
    with patch(
        "poemai_utils.embeddings.openai_embedder_lean.requests"
    ) as requests_mock:

        embedder = OpenAIEmbedderLean(
            model_name="text-embedding-ada-002", openai_api_key="your_openai_api_key"
        )

        response_mock = MagicMock()
        response_mock.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        response_mock.status_code = 200

        requests_mock.post.return_value = response_mock

        embedding = embedder.calc_embedding("Example text")

        np.array_equal(embedding, np.array([0.1, 0.2, 0.3], dtype=np.float32))
