"""Embedding model wrapper using sentence-transformers."""

from typing import List, Union
import hashlib
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
    SentenceTransformer = None


class EmbeddingModel:
    """Wrapper for sentence embedding models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda",
        fallback_dimension: int = 768,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: Model name or path
            device: Device to use (cuda or cpu)
        """
        self.model_name = model_name
        self.device = device
        self.fallback_dimension = fallback_dimension

        if SentenceTransformer is None:
            logger.warning(
                "sentence-transformers 不可用，EmbeddingModel 将使用哈希降级向量。"
                "这会降低检索质量，但可保证流程继续运行。"
            )
            self.model = None
            return

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

    def _fallback_encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        """Deterministic embedding without heavy dependencies."""
        raw = text if isinstance(text, str) else str(text)
        digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.fallback_dimension).astype(np.float32)
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size
            show_progress_bar: Whether to show progress
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.model is None:
            embeddings = [self._fallback_encode_one(t, normalize=normalize) for t in texts]
            return np.array(embeddings)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
        )

        return np.array(embeddings)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        Encode queries (alias for encode).

        Args:
            queries: List of query texts
            **kwargs: Additional arguments for encode

        Returns:
            Query embeddings
        """
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: List[str], **kwargs) -> np.ndarray:
        """
        Encode corpus documents (alias for encode).

        Args:
            corpus: List of document texts
            **kwargs: Additional arguments for encode

        Returns:
            Document embeddings
        """
        return self.encode(corpus, **kwargs)

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute pairwise similarities.

        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)
            metric: Similarity metric (cosine or dot)

        Returns:
            Similarity matrix (N x M)
        """
        if metric == "cosine":
            # Normalize if not already
            emb1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            emb2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            return np.dot(emb1_norm, emb2_norm.T)
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def save(self, save_path: str):
        """
        Save model.

        Args:
            save_path: Path to save
        """
        logger.info(f"Saving embedding model to {save_path}")
        self.model.save(save_path)

    @classmethod
    def load(cls, load_path: str, device: str = "cuda") -> "EmbeddingModel":
        """
        Load model from path.

        Args:
            load_path: Path to load from
            device: Device to use

        Returns:
            Loaded model instance
        """
        logger.info(f"Loading embedding model from {load_path}")
        return cls(model_name=load_path, device=device)
