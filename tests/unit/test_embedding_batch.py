import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from poemai_utils.embeddings.openai_embedder import OpenAIEmbedder
from poemai_utils.embeddings.openai_embedder_lean import OpenAIEmbedderLean

_logger = logging.getLogger(__name__)


class TestEmbeddingBatch:
    """Test cases for embedding batch functionality in poemai-utils"""

    def test_openai_embedder_lean_batch_mock(self):
        """Test batch embedding with mocked API responses for OpenAIEmbedderLean"""
        embedder = OpenAIEmbedderLean(
            "text-embedding-3-small", openai_api_key="test_key"
        )

        # Mock the requests.post method
        with patch("requests.post") as mock_post:
            # Mock response for batch request
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                    {"embedding": [0.7, 0.8, 0.9]},
                ]
            }
            mock_post.return_value = mock_response

            # Test batch processing
            texts = ["First text", "Second text", "Third text"]
            embeddings = embedder.calc_embedding_batch(texts)

            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            request_data = call_args[1]["json"]
            assert request_data["input"] == texts
            assert request_data["model"] == embedder.model_key

            # Verify results
            assert len(embeddings) == 3
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            assert embeddings[0].shape == (3,)
            np.testing.assert_array_almost_equal(
                embeddings[0], [0.1, 0.2, 0.3], decimal=6
            )
            np.testing.assert_array_almost_equal(
                embeddings[1], [0.4, 0.5, 0.6], decimal=6
            )
            np.testing.assert_array_almost_equal(
                embeddings[2], [0.7, 0.8, 0.9], decimal=6
            )

    def test_openai_embedder_batch_mock(self):
        """Test batch embedding with mocked API responses for OpenAIEmbedder"""
        # Skip if OpenAI package not available
        pytest.importorskip("openai")

        embedder = OpenAIEmbedder("text-embedding-3-small", openai_api_key="test_key")

        # Mock the OpenAI client
        with patch.object(embedder.client.embeddings, "create") as mock_create:
            # Mock response for batch request
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.4, 0.5, 0.6]),
                MagicMock(embedding=[0.7, 0.8, 0.9]),
            ]
            mock_create.return_value = mock_response

            # Test batch processing
            texts = ["First text", "Second text", "Third text"]
            embeddings = embedder.calc_embedding_batch(texts)

            # Verify API call
            mock_create.assert_called_once_with(input=texts, model=embedder.model_name)

            # Verify results
            assert len(embeddings) == 3
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            assert embeddings[0].shape == (3,)
            np.testing.assert_array_almost_equal(
                embeddings[0], [0.1, 0.2, 0.3], decimal=6
            )
            np.testing.assert_array_almost_equal(
                embeddings[1], [0.4, 0.5, 0.6], decimal=6
            )
            np.testing.assert_array_almost_equal(
                embeddings[2], [0.7, 0.8, 0.9], decimal=6
            )

    def test_openai_embedder_lean_single_vs_batch_compatibility(self):
        """Test that single embedding still works alongside batch for OpenAIEmbedderLean"""
        embedder = OpenAIEmbedderLean(
            "text-embedding-3-small", openai_api_key="test_key"
        )

        with patch("requests.post") as mock_post:
            # Test single embedding (existing functionality)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            mock_post.return_value = mock_response

            embedding = embedder.calc_embedding("Single text")

            # Verify single call format
            call_args = mock_post.call_args
            request_data = call_args[1]["json"]
            assert request_data["input"] == "Single text"
            assert isinstance(embedding, np.ndarray)
            np.testing.assert_array_almost_equal(embedding, [0.1, 0.2, 0.3], decimal=6)

    def test_openai_embedder_single_vs_batch_compatibility(self):
        """Test that single embedding still works alongside batch for OpenAIEmbedder"""
        pytest.importorskip("openai")

        embedder = OpenAIEmbedder("text-embedding-3-small", openai_api_key="test_key")

        with patch.object(embedder.client.embeddings, "create") as mock_create:
            # Test single embedding (existing functionality)
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_create.return_value = mock_response

            embedding = embedder.calc_embedding("Single text")

            # Verify single call format
            mock_create.assert_called_once_with(
                input="Single text", model=embedder.model_name
            )
            assert isinstance(embedding, np.ndarray)
            np.testing.assert_array_almost_equal(embedding, [0.1, 0.2, 0.3], decimal=6)

    def test_batch_error_handling_lean(self):
        """Test error handling in batch embedding for OpenAIEmbedderLean"""
        embedder = OpenAIEmbedderLean(
            "text-embedding-3-small", openai_api_key="test_key"
        )

        with patch("requests.post") as mock_post:
            # Test API error
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"
            mock_post.return_value = mock_response

            texts = ["Test text"]

            with pytest.raises(RuntimeError, match="Failed to retrieve embedding: 400"):
                embedder.calc_embedding_batch(texts)

    def test_batch_error_handling_regular(self):
        """Test error handling in batch embedding for OpenAIEmbedder"""
        pytest.importorskip("openai")

        embedder = OpenAIEmbedder("text-embedding-3-small", openai_api_key="test_key")

        with patch.object(embedder.client.embeddings, "create") as mock_create:
            # Test API error - use generic Exception instead of APIError
            mock_create.side_effect = Exception("API Error")

            texts = ["Test text"]

            with pytest.raises(Exception):
                embedder.calc_embedding_batch(texts)

    def test_empty_batch_handling(self):
        """Test handling of empty batch for both embedder types"""
        # Test OpenAIEmbedderLean
        embedder_lean = OpenAIEmbedderLean(
            "text-embedding-3-small", openai_api_key="test_key"
        )
        embeddings = embedder_lean.calc_embedding_batch([])
        assert embeddings == []

        # Test OpenAIEmbedder if available
        try:
            pytest.importorskip("openai")
            embedder = OpenAIEmbedder(
                "text-embedding-3-small", openai_api_key="test_key"
            )
            embeddings = embedder.calc_embedding_batch([])
            assert embeddings == []
        except:
            pass  # Skip if OpenAI not available
