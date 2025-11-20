import logging
import os

import numpy as np
import pytest
from poemai_utils.embeddings.openai_embedder import OpenAIEmbedder
from poemai_utils.embeddings.openai_embedder_lean import OpenAIEmbedderLean

_logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_openai_embedder_lean_real_api_batch():
    """Test batch embedding with real OpenAI API using OpenAIEmbedderLean"""
    # Skip if no API key available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")

    embedder = OpenAIEmbedderLean(
        "text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
    ]

    # Test batch embedding
    batch_embeddings = embedder.calc_embedding_batch(texts)

    # Verify results
    assert len(batch_embeddings) == len(texts)
    assert all(isinstance(emb, np.ndarray) for emb in batch_embeddings)
    assert all(emb.shape == (embedder.embedding_dim(),) for emb in batch_embeddings)

    # Test that individual embeddings are close to batch results
    # Note: Due to OpenAI API variations, we use looser tolerance
    for i, text in enumerate(texts):
        individual_embedding = embedder.calc_embedding(text)
        # Use cosine similarity for comparison instead of exact equality
        similarity = np.dot(individual_embedding, batch_embeddings[i]) / (
            np.linalg.norm(individual_embedding) * np.linalg.norm(batch_embeddings[i])
        )
        assert (
            similarity > 0.99
        ), f"Batch embedding {i} cosine similarity {similarity:.6f} too low"

    _logger.info(f"Successfully tested batch embedding with {len(texts)} texts")
    _logger.info(f"Embedding dimensions: {batch_embeddings[0].shape}")


@pytest.mark.integration
def test_openai_embedder_real_api_batch():
    """Test batch embedding with real OpenAI API using OpenAIEmbedder"""
    # Skip if no API key available or OpenAI package not available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")

    pytest.importorskip("openai")

    embedder = OpenAIEmbedder(
        "text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Test texts
    texts = [
        "Document processing with batch embeddings.",
        "Performance optimization for text vectorization.",
    ]

    # Test batch embedding
    batch_embeddings = embedder.calc_embedding_batch(texts)

    # Verify results
    assert len(batch_embeddings) == len(texts)
    assert all(isinstance(emb, np.ndarray) for emb in batch_embeddings)
    assert all(emb.shape == (embedder.embedding_dim(),) for emb in batch_embeddings)

    # Test that individual embeddings are close to batch results
    # Note: Due to OpenAI API variations, we use looser tolerance
    for i, text in enumerate(texts):
        individual_embedding = embedder.calc_embedding(text)
        # Use cosine similarity for comparison instead of exact equality
        similarity = np.dot(individual_embedding, batch_embeddings[i]) / (
            np.linalg.norm(individual_embedding) * np.linalg.norm(batch_embeddings[i])
        )
        assert (
            similarity > 0.99
        ), f"Batch embedding {i} cosine similarity {similarity:.6f} too low"

    _logger.info(f"Successfully tested batch embedding with {len(texts)} texts")
    _logger.info(f"Embedding dimensions: {batch_embeddings[0].shape}")


@pytest.mark.integration
def test_batch_vs_individual_performance_comparison():
    """Compare performance of batch vs individual embedding calls"""
    # Skip if no API key available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not available for integration test")

    embedder = OpenAIEmbedderLean(
        "text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    texts = [
        "Performance test text number one.",
        "Performance test text number two.",
        "Performance test text number three.",
        "Performance test text number four.",
        "Performance test text number five.",
    ]

    import time

    # Test batch timing
    start_time = time.time()
    batch_embeddings = embedder.calc_embedding_batch(texts)
    batch_time = time.time() - start_time

    # Test individual timing
    start_time = time.time()
    individual_embeddings = []
    for text in texts:
        individual_embeddings.append(embedder.calc_embedding(text))
    individual_time = time.time() - start_time

    # Verify results are similar
    assert len(batch_embeddings) == len(individual_embeddings)
    for i in range(len(texts)):
        similarity = np.dot(individual_embeddings[i], batch_embeddings[i]) / (
            np.linalg.norm(individual_embeddings[i])
            * np.linalg.norm(batch_embeddings[i])
        )
        assert similarity > 0.99

    # Log performance comparison
    _logger.info(f"Batch processing time: {batch_time:.3f}s")
    _logger.info(f"Individual processing time: {individual_time:.3f}s")
    _logger.info(f"Speed improvement: {individual_time / batch_time:.2f}x")

    # Batch should be faster than individual calls
    assert (
        batch_time < individual_time
    ), "Batch processing should be faster than individual calls"
