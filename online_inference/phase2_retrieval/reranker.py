"""Structure-aware re-ranking using text and GraphCodeBERT similarity."""

from typing import Dict, List
import numpy as np
from loguru import logger


class Reranker:
    """Structure-aware re-ranker combining text and code structure similarity."""

    def __init__(
        self,
        graphcodebert_model,
        embedding_model,
        text_weight: float = 0.4,
        code_weight: float = 0.6,
    ):
        """
        Initialize re-ranker.

        Args:
            graphcodebert_model: GraphCodeBERT model for code similarity
            embedding_model: Embedding model for text similarity
            text_weight: Weight for text similarity
            code_weight: Weight for GraphCodeBERT similarity
        """
        self.graphcodebert = graphcodebert_model
        self.embedding_model = embedding_model
        self.text_weight = text_weight
        self.code_weight = code_weight

    def rerank(
        self,
        query_context: str,
        query_code: str,
        candidates: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Re-rank candidates using structure-aware soft voting.

        Args:
            query_context: Query context (enhanced context)
            query_code: Query buggy code
            candidates: List of candidate dictionaries
            top_k: Number of top candidates to return

        Returns:
            Top-k re-ranked candidates
        """
        logger.info(f"Re-ranking {len(candidates)} candidates")

        if not candidates:
            return []

        # Compute text similarity scores
        text_scores = self._compute_text_similarity(query_context, candidates)

        # Compute code structure similarity scores
        code_scores = self._compute_code_similarity(query_code, candidates)

        # Combine scores
        final_scores = []
        for i in range(len(candidates)):
            combined_score = (
                self.text_weight * text_scores[i] + self.code_weight * code_scores[i]
            )
            final_scores.append(combined_score)
            candidates[i]["rerank_score"] = combined_score
            candidates[i]["text_similarity"] = text_scores[i]
            candidates[i]["code_similarity"] = code_scores[i]

        # Sort by final score (descending)
        sorted_indices = np.argsort(final_scores)[::-1]
        reranked_candidates = [candidates[i] for i in sorted_indices]

        # Return top-k
        top_candidates = reranked_candidates[:top_k]
        
        logger.info(f"Top-{top_k} candidates selected")
        for i, candidate in enumerate(top_candidates):
            logger.debug(
                f"Rank {i+1}: sample_id={candidate.get('sample_id')}, "
                f"score={candidate.get('rerank_score', 0):.4f}"
            )

        return top_candidates

    def _compute_text_similarity(
        self, query_context: str, candidates: List[Dict]
    ) -> List[float]:
        """
        Compute text similarity between query and candidates.

        Args:
            query_context: Query context text
            candidates: Candidate list

        Returns:
            List of similarity scores
        """
        # Extract candidate texts (use distilled views)
        candidate_texts = []
        for candidate in candidates:
            # Combine views if available
            views = candidate.get("views", {})
            if views:
                text = " ".join(views.values())
            else:
                text = candidate.get("buggy_code", "")
            candidate_texts.append(text)

        # Encode query and candidates
        try:
            query_embedding = self.embedding_model.encode(query_context)
            candidate_embeddings = self.embedding_model.encode(candidate_texts)

            # Compute cosine similarity
            similarities = self.embedding_model.similarity(
                query_embedding.reshape(1, -1),
                candidate_embeddings,
                metric="cosine",
            )[0]

            return similarities.tolist()

        except Exception as e:
            logger.error(f"Text similarity computation failed: {e}")
            return [0.0] * len(candidates)

    def _compute_code_similarity(
        self, query_code: str, candidates: List[Dict]
    ) -> List[float]:
        """
        Compute code structure similarity using GraphCodeBERT.

        Args:
            query_code: Query code
            candidates: Candidate list

        Returns:
            List of similarity scores
        """
        # Extract candidate codes
        candidate_codes = [c.get("buggy_code", "") for c in candidates]

        try:
            # Encode with GraphCodeBERT
            query_embedding = self.graphcodebert.encode(query_code)
            candidate_embeddings = self.graphcodebert.encode(candidate_codes)

            # Compute cosine similarity
            similarities = []
            for cand_emb in candidate_embeddings:
                sim = self.graphcodebert.compute_similarity(
                    query_embedding, cand_emb, metric="cosine"
                )
                similarities.append(sim)

            return similarities

        except Exception as e:
            logger.error(f"Code similarity computation failed: {e}")
            return [0.0] * len(candidates)

    def rerank_with_rca(
        self,
        query_rca: Dict[str, str],
        candidates: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Re-rank using RCA (Root Cause Analysis) representations.

        Args:
            query_rca: Dictionary of query RCA for each view type
            candidates: Candidates with view-specific RCA
            top_k: Number of top candidates

        Returns:
            Top-k re-ranked candidates
        """
        logger.info("Re-ranking with RCA similarity")

        # Compute view-specific similarities
        view_similarities = {}
        
        for view_type, query_rca_text in query_rca.items():
            candidate_rca_texts = [
                c.get("views", {}).get(view_type, "") for c in candidates
            ]

            # Encode and compute similarity
            try:
                query_emb = self.embedding_model.encode(query_rca_text)
                cand_embs = self.embedding_model.encode(candidate_rca_texts)
                
                sims = self.embedding_model.similarity(
                    query_emb.reshape(1, -1), cand_embs, metric="cosine"
                )[0]
                
                view_similarities[view_type] = sims.tolist()

            except Exception as e:
                logger.error(f"RCA similarity for {view_type} failed: {e}")
                view_similarities[view_type] = [0.0] * len(candidates)

        # Soft voting: average across views
        final_scores = []
        for i in range(len(candidates)):
            view_scores = [view_similarities[vt][i] for vt in view_similarities]
            avg_score = sum(view_scores) / len(view_scores) if view_scores else 0.0
            final_scores.append(avg_score)
            candidates[i]["rca_similarity"] = avg_score

        # Sort and return top-k
        sorted_indices = np.argsort(final_scores)[::-1]
        reranked = [candidates[i] for i in sorted_indices]
        
        return reranked[:top_k]
