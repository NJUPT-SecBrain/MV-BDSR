"""Tests for offline_indexing module."""

import pytest
import numpy as np
from offline_indexing import MultiViewGenerator, ViewDistillation, VectorStore


class MockLLM:
    """Mock LLM for testing."""
    
    def generate(self, prompt, **kwargs):
        return "Mock analysis result"


class MockEmbedding:
    """Mock embedding model for testing."""
    
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), 768)


class TestMultiViewGenerator:
    """Tests for MultiViewGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        llm = MockLLM()
        generator = MultiViewGenerator(llm)
        
        assert len(generator.VIEW_TYPES) == 3

    def test_generate_single_view(self):
        """Test single view generation."""
        llm = MockLLM()
        generator = MultiViewGenerator(llm)
        
        view = generator.generate_single_view("int x = 5;", "data_flow")
        assert isinstance(view, str)

    def test_generate_blind_views(self):
        """Test multi-view generation."""
        llm = MockLLM()
        generator = MultiViewGenerator(llm)
        
        views = generator.generate_blind_views("int x = 5;")
        assert len(views) == 3
        assert "data_flow" in views


class TestViewDistillation:
    """Tests for ViewDistillation."""

    def test_distill_view(self):
        """Test view distillation."""
        llm = MockLLM()
        distillation = ViewDistillation(llm)
        
        distilled = distillation.distill_view(
            "Long analysis text",
            "int x = 5;",
            "data_flow",
        )
        assert isinstance(distilled, str)

    def test_extract_key_facts(self):
        """Test key fact extraction."""
        llm = MockLLM()
        distillation = ViewDistillation(llm)
        
        facts = distillation.extract_key_facts(
            "Fact 1: something\nFact 2: another thing"
        )
        assert isinstance(facts, list)


class TestVectorStore:
    """Tests for VectorStore."""

    def test_create_index(self):
        """Test index creation."""
        store = VectorStore(dimension=768)
        store.create_index("data_flow")
        
        assert "data_flow" in store.indices

    def test_add_vectors(self):
        """Test adding vectors."""
        store = VectorStore(dimension=768)
        
        embeddings = np.random.rand(10, 768).astype(np.float32)
        metadata = [{"id": i} for i in range(10)]
        
        store.add_vectors("data_flow", embeddings, metadata)
        
        stats = store.get_statistics("data_flow")
        assert stats["total_vectors"] == 10

    def test_search(self):
        """Test vector search."""
        store = VectorStore(dimension=768)
        
        # Add some vectors
        embeddings = np.random.rand(10, 768).astype(np.float32)
        metadata = [{"id": i} for i in range(10)]
        store.add_vectors("data_flow", embeddings, metadata)
        
        # Search
        query = np.random.rand(768).astype(np.float32)
        distances, results = store.search("data_flow", query, k=5)
        
        assert len(results) == 5
