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


def test_openai_embedder_lean_custom_model():
    """Test OpenAIEmbedderLean with custom model and custom base URL"""
    with patch(
        "poemai_utils.embeddings.openai_embedder_lean.requests"
    ) as requests_mock:

        # Test custom model with custom base URL (should not validate against OpenAI models)
        embedder = OpenAIEmbedderLean(
            model_name="JINA_EMBEDDINGS_V2_BASE_DE",
            openai_api_key="dummy",
            base_url="http://host.docker.internal:29899/v1",
        )

        # Verify custom model properties
        assert embedder.model_key == "JINA_EMBEDDINGS_V2_BASE_DE"
        assert (
            embedder.embeddings_dimensions is None
        )  # Custom models have None dimensions initially
        assert embedder.base_url == "http://host.docker.internal:29899/v1"

        # Mock API response for embedding calculation
        response_mock = MagicMock()
        response_mock.json.return_value = {
            "data": [{"embedding": [0.5, 0.6, 0.7, 0.8]}]
        }
        response_mock.status_code = 200
        requests_mock.post.return_value = response_mock

        # Test embedding calculation works
        embedding = embedder.calc_embedding("Test text for custom model")

        # Verify the request was made to the custom base URL
        requests_mock.post.assert_called_once()
        call_args = requests_mock.post.call_args
        assert call_args[0][0] == "http://host.docker.internal:29899/v1/embeddings"

        # Verify embedding result
        expected_embedding = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        np.testing.assert_array_equal(embedding, expected_embedding)


def test_openai_embedder_lean_invalid_standard_model():
    """Test that invalid standard models still raise ValueError when using default base URL"""
    try:
        embedder = OpenAIEmbedderLean(
            model_name="INVALID_MODEL_NAME",
            openai_api_key="test_key",
            # No custom base_url, so should validate against OpenAI models
        )
        assert False, "Should have raised ValueError for invalid model"
    except ValueError as e:
        assert "not found in OpenAI models" in str(e)
