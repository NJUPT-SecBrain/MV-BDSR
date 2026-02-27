"""GraphCodeBERT model wrapper for code embeddings."""

from typing import List, Optional, Union
import numpy as np
from loguru import logger

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    logger.warning("transformers not installed. Install with: pip install transformers torch")
    torch = None
    AutoModel = None
    AutoTokenizer = None


class GraphCodeBERTModel:
    """Wrapper for GraphCodeBERT model."""

    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        device: str = "cuda",
        max_length: int = 512,
    ):
        """
        Initialize GraphCodeBERT model.

        Args:
            model_name: Model name or path
            device: Device to use (cuda or cpu)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length

        if torch is None or AutoModel is None:
            raise ImportError("torch and transformers required")

        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        logger.info(f"Loading GraphCodeBERT from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings)

        # Concatenate
        embeddings = np.vstack(all_embeddings)

        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts.

        Args:
            texts: List of texts

        Returns:
            Batch embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def compute_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
        metric: str = "cosine",
    ) -> float:
        """
        Compute similarity between two texts or embeddings.

        Args:
            text1: First text or embedding
            text2: Second text or embedding
            metric: Similarity metric (cosine or euclidean)

        Returns:
            Similarity score
        """
        # Get embeddings if texts provided
        if isinstance(text1, str):
            emb1 = self.encode(text1)
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self.encode(text2)
        else:
            emb2 = text2

        # Compute similarity
        if metric == "cosine":
            similarity = np.dot(emb1.flatten(), emb2.flatten()) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
        elif metric == "euclidean":
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1 / (1 + distance)  # Convert distance to similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return float(similarity)

    def batch_compute_similarity(
        self,
        queries: List[str],
        documents: List[str],
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute pairwise similarities between queries and documents.

        Args:
            queries: List of query texts
            documents: List of document texts
            metric: Similarity metric

        Returns:
            Similarity matrix (queries x documents)
        """
        # Encode all texts
        query_embs = self.encode(queries)
        doc_embs = self.encode(documents)

        # Compute similarity matrix
        if metric == "cosine":
            # Cosine similarity
            similarities = np.dot(query_embs, doc_embs.T)
        elif metric == "euclidean":
            # Euclidean distance -> similarity
            from scipy.spatial.distance import cdist
            distances = cdist(query_embs, doc_embs, metric="euclidean")
            similarities = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return similarities

    def save(self, save_path: str):
        """
        Save model and tokenizer.

        Args:
            save_path: Path to save
        """
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def load(cls, load_path: str, device: str = "cuda") -> "GraphCodeBERTModel":
        """
        Load model from path.

        Args:
            load_path: Path to load from
            device: Device to use

        Returns:
            Loaded model instance
        """
        logger.info(f"Loading model from {load_path}")
        return cls(model_name=load_path, device=device)
