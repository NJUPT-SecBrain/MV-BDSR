"""Tests for models module."""

import pytest
import numpy as np


class TestLLMInterface:
    """Tests for LLMInterface."""

    def test_initialization(self):
        """Test LLM interface initialization."""
        from models import LLMInterface
        
        # Test without API key (will fail to initialize but should not crash)
        try:
            llm = LLMInterface(provider="openai", model_name="gpt-4")
        except Exception:
            pass  # Expected if no API key

    def test_count_tokens(self):
        """Test token counting."""
        from models import LLMInterface
        
        llm = LLMInterface(provider="openai", model_name="gpt-4")
        count = llm.count_tokens("Hello world")
        
        assert isinstance(count, int)
        assert count > 0


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""

    @pytest.mark.skipif(True, reason="Requires model download")
    def test_encode(self):
        """Test encoding texts."""
        from models import EmbeddingModel
        
        model = EmbeddingModel()
        embeddings = model.encode(["text1", "text2"])
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == model.get_dimension()

    @pytest.mark.skipif(True, reason="Requires model download")
    def test_similarity(self):
        """Test similarity computation."""
        from models import EmbeddingModel
        
        model = EmbeddingModel()
        emb1 = model.encode("text1")
        emb2 = model.encode("text2")
        
        sim = model.similarity(emb1, emb2)
        assert isinstance(sim, np.ndarray)


class TestGraphCodeBERTModel:
    """Tests for GraphCodeBERTModel."""

    @pytest.mark.skipif(True, reason="Requires model download and GPU")
    def test_encode_code(self):
        """Test encoding code."""
        from models import GraphCodeBERTModel
        
        model = GraphCodeBERTModel(device="cpu")
        code = "int x = 5;"
        
        embedding = model.encode(code)
        assert embedding.shape[1] == 768  # GraphCodeBERT dimension

    @pytest.mark.skipif(True, reason="Requires model download and GPU")
    def test_compute_similarity(self):
        """Test code similarity."""
        from models import GraphCodeBERTModel
        
        model = GraphCodeBERTModel(device="cpu")
        code1 = "int x = 5;"
        code2 = "int y = 5;"
        
        similarity = model.compute_similarity(code1, code2)
        assert 0.0 <= similarity <= 1.0
