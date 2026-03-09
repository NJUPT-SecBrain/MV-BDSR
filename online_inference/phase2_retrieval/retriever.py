"""Routed retrieval from multi-view indices."""

from typing import Dict, List, Tuple
import numpy as np
from loguru import logger


class Retriever:
    """Multi-view retriever with routing and pooling."""

    def __init__(self, vector_store, embedding_model, top_k_per_view: int = 10):
        """
        Initialize retriever.

        Args:
            vector_store: VectorStore instance with indices
            embedding_model: Model for encoding queries
            top_k_per_view: Number of results to retrieve per view
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k_per_view = top_k_per_view

    def retrieve(self, queries: Dict[str, str]) -> List[Dict]:
        """
        Retrieve candidates from all views and pool results.

        Args:
            queries: Dictionary mapping view types to query strings

        Returns:
            Pooled list of candidate dictionaries
        """
        logger.info("Retrieving from multi-view indices")
        
        # 获取可用的视角列表
        available_views = self.vector_store.get_available_views()
        logger.info(f"可用视角: {available_views}")
        
        # 过滤查询，只保留可用视角的查询
        filtered_queries = {
            view_type: query 
            for view_type, query in queries.items() 
            if view_type in available_views
        }
        
        if not filtered_queries:
            logger.warning("没有可用的视角进行检索！请重新构建完整索引。")
            return []
        
        if len(filtered_queries) < len(queries):
            missing_views = set(queries.keys()) - set(filtered_queries.keys())
            logger.warning(
                f"部分视角不可用，已跳过: {missing_views}。"
                f"建议重新构建完整索引以获得更好的检索效果。"
            )
        
        all_candidates = {}  # Use dict to avoid duplicates by sample_id

        for view_type, query in filtered_queries.items():
            logger.debug(f"Retrieving from {view_type} index")
            
            # Encode query
            query_embedding = self.embedding_model.encode(query)

            # Search in vector store
            try:
                distances, metadata_list = self.vector_store.search(
                    view_type, query_embedding, k=self.top_k_per_view
                )

                # Add candidates
                for i, metadata in enumerate(metadata_list):
                    sample_id = metadata.get("sample_id")
                    
                    if sample_id not in all_candidates:
                        all_candidates[sample_id] = {
                            "sample_id": sample_id,
                            "buggy_code": metadata.get("buggy_code", ""),
                            "patch": metadata.get("patch", ""),
                            "views": {},
                            "scores": {},
                        }

                    # Store view-specific info
                    all_candidates[sample_id]["views"][view_type] = metadata.get("distilled_view", "")
                    all_candidates[sample_id]["scores"][view_type] = float(distances[i])

            except Exception as e:
                logger.error(f"Retrieval from {view_type} failed: {e}")
                continue

        # Convert to list
        pooled_candidates = list(all_candidates.values())
        
        logger.info(f"Pooled {len(pooled_candidates)} unique candidates")
        return pooled_candidates

    def retrieve_single_view(self, query: str, view_type: str, k: int = 10) -> List[Dict]:
        """
        Retrieve from a single view.

        Args:
            query: Query string
            view_type: View type to search
            k: Number of results

        Returns:
            List of candidates
        """
        logger.debug(f"Retrieving {k} candidates from {view_type}")

        # Encode query
        query_embedding = self.embedding_model.encode(query)

        # Search
        try:
            distances, metadata_list = self.vector_store.search(view_type, query_embedding, k=k)
            
            candidates = []
            for i, metadata in enumerate(metadata_list):
                candidate = {
                    "sample_id": metadata.get("sample_id"),
                    "buggy_code": metadata.get("buggy_code", ""),
                    "patch": metadata.get("patch", ""),
                    "distilled_view": metadata.get("distilled_view", ""),
                    "view_type": view_type,
                    "score": float(distances[i]),
                }
                candidates.append(candidate)

            return candidates

        except Exception as e:
            logger.error(f"Single-view retrieval failed: {e}")
            return []

    def compute_fusion_scores(self, candidates: List[Dict]) -> List[Dict]:
        """
        Compute fusion scores for candidates based on multi-view scores.

        Args:
            candidates: List of candidates with view-specific scores

        Returns:
            Candidates with fusion scores
        """
        for candidate in candidates:
            scores = candidate.get("scores", {})
            
            if not scores:
                candidate["fusion_score"] = 0.0
                continue

            # Simple average fusion
            # Could use weighted average, reciprocal rank fusion, etc.
            avg_score = sum(scores.values()) / len(scores)
            candidate["fusion_score"] = avg_score

        return candidates

    def filter_by_threshold(self, candidates: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """
        Filter candidates by score threshold.

        Args:
            candidates: List of candidates
            threshold: Minimum fusion score

        Returns:
            Filtered candidates
        """
        filtered = [c for c in candidates if c.get("fusion_score", 0.0) >= threshold]
        logger.info(f"Filtered {len(filtered)}/{len(candidates)} candidates above threshold {threshold}")
        return filtered
