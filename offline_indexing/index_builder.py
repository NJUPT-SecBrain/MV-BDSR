"""Main orchestrator for building offline indices.

This module supports checkpoint-based resume.

Design:
- After each batch, we persist the computed embeddings + metadata to disk under:
  <output_dir>/cache/<view_type>/batch_<idx>.{npz,pkl}
- On restart, we reuse cached batches to avoid re-calling the LLM.
- After finishing a view, we assemble all cached batches, build the FAISS index,
  and (optionally) save indices to disk.
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import json
import pickle
from loguru import logger
from tqdm import tqdm


class IndexBuilder:
    """Orchestrator for the complete offline indexing pipeline."""

    def __init__(
        self,
        multiview_generator,
        distillation,
        vector_store,
        embedding_model,
    ):
        """
        Initialize index builder.

        Args:
            multiview_generator: MultiViewGenerator instance
            distillation: ViewDistillation instance
            vector_store: VectorStore instance
            embedding_model: Model for generating embeddings
        """
        self.multiview_gen = multiview_generator
        self.distillation = distillation
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def build_from_dataset(
        self,
        dataset: List[Dict[str, str]],
        save_dir: Path,
        batch_size: int = 50,
        resume: bool = True,
    ):
        """
        Build complete multi-view indices from dataset.

        Args:
            dataset: List of samples with 'buggy_code' and 'patch'
            save_dir: Directory to save indices
            batch_size: Batch size for processing
            resume: Whether to resume from checkpoint if available
        """
        logger.info(f"Building indices from {len(dataset)} samples")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / "checkpoint.json"
        
        # Load checkpoint if resuming
        checkpoint = None
        if resume and checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                logger.info(f"Found checkpoint: resuming from view '{checkpoint['current_view']}', batch {checkpoint['batch_idx']}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
                checkpoint = None

        # Process each view type
        view_types = self.multiview_gen.VIEW_TYPES
        start_view_idx = 0
        
        if checkpoint:
            # Find where to resume
            if checkpoint["current_view"] in view_types:
                start_view_idx = view_types.index(checkpoint["current_view"])
                logger.info(f"Resuming from view {start_view_idx + 1}/{len(view_types)}: {checkpoint['current_view']}")
        
        for view_idx, view_type in enumerate(view_types[start_view_idx:], start=start_view_idx):
            logger.info(f"Processing {view_type} view ({view_idx + 1}/{len(view_types)})")
            
            start_batch_idx = 0
            if checkpoint and checkpoint["current_view"] == view_type:
                start_batch_idx = checkpoint["batch_idx"]
                logger.info(f"Resuming from batch {start_batch_idx}")
            
            self._build_single_view_index(
                dataset, 
                view_type, 
                batch_size, 
                save_dir=save_dir,
                start_batch_idx=start_batch_idx,
            )

            # Save partial indices after each view, so users can observe artifacts
            # like index_data_flow.faiss without waiting for all views.
            logger.info(f"Saving indices after finishing view '{view_type}'")
            self.vector_store.save(save_dir)

        # Save all indices
        logger.info(f"Saving indices to {save_dir}")
        self.vector_store.save(save_dir)
        
        # Remove checkpoint after successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint removed after successful completion")

        logger.info("Index building complete")

    def _cache_dir(self, save_dir: Path, view_type: str) -> Path:
        return Path(save_dir) / "cache" / view_type

    def _batch_cache_paths(self, cache_dir: Path, batch_idx: int) -> tuple[Path, Path]:
        emb_path = cache_dir / f"batch_{batch_idx:05d}.npz"
        meta_path = cache_dir / f"batch_{batch_idx:05d}.pkl"
        return emb_path, meta_path

    def _list_cached_batches(self, cache_dir: Path) -> set[int]:
        if not cache_dir.exists():
            return set()
        cached: set[int] = set()
        for p in cache_dir.glob("batch_*.npz"):
            name = p.stem  # batch_00001
            try:
                idx = int(name.split("_", 1)[1])
                cached.add(idx)
            except Exception:
                continue
        return cached

    def _save_batch_cache(
        self,
        cache_dir: Path,
        batch_idx: int,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path, meta_path = self._batch_cache_paths(cache_dir, batch_idx)
        # embeddings: store as float32 to reduce disk footprint
        np.savez_compressed(emb_path, embeddings=np.asarray(embeddings, dtype=np.float32))
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_batch_cache(
        self,
        cache_dir: Path,
        batch_idx: int,
    ) -> tuple[np.ndarray, List[Dict]]:
        emb_path, meta_path = self._batch_cache_paths(cache_dir, batch_idx)
        with np.load(emb_path) as z:
            embeddings = z["embeddings"]
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        return embeddings, metadata

    def _build_single_view_index(
        self,
        dataset: List[Dict[str, str]],
        view_type: str,
        batch_size: int,
        save_dir: Optional[Path] = None,
        start_batch_idx: int = 0,
    ):
        """
        Build index for a single view type.

        Args:
            dataset: Dataset samples
            view_type: View type to build
            batch_size: Batch size
            save_dir: Directory to save checkpoint (optional)
            start_batch_idx: Batch index to start from (for resuming)
        """
        # Process in batches
        batch_indices = list(range(0, len(dataset), batch_size))

        cache_dir: Optional[Path] = None
        cached_batches: set[int] = set()
        if save_dir is not None:
            cache_dir = self._cache_dir(save_dir, view_type)
            cached_batches = self._list_cached_batches(cache_dir)
            if cached_batches:
                logger.info(
                    f"Found {len(cached_batches)} cached batches for view '{view_type}' "
                    f"under {cache_dir}"
                )

        # IMPORTANT:
        # Even if we "resume" from a later checkpoint batch (e.g. batch 31),
        # earlier batches might not be cached if they were computed by older code.
        # In that case, we must recompute missing batches to be able to assemble
        # a complete index for this view.
        if start_batch_idx > 0 and cache_dir is not None:
            missing_before = [
                b for b in range(start_batch_idx) if b not in cached_batches
            ]
            if missing_before:
                logger.warning(
                    f"Checkpoint asks to resume from batch {start_batch_idx}, but "
                    f"{len(missing_before)} earlier batches are not cached. "
                    f"Will recompute missing batches to build a complete index."
                )

        pbar = tqdm(
            range(0, len(batch_indices)),
            desc=f"Building {view_type}",
            total=len(batch_indices),
        )

        for batch_idx in pbar:
            i = batch_indices[batch_idx]
            # If this batch is cached, skip LLM calls and just update checkpoint.
            if cache_dir is not None and batch_idx in cached_batches:
                if save_dir:
                    checkpoint = {
                        "current_view": view_type,
                        "batch_idx": batch_idx + 1,  # Next batch to process
                        "total_batches": len(batch_indices),
                        "samples_processed": (batch_idx + 1) * batch_size,
                        "total_samples": len(dataset),
                    }
                    checkpoint_path = Path(save_dir) / "checkpoint.json"
                    try:
                        with open(checkpoint_path, "w") as f:
                            json.dump(checkpoint, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint: {e}")
                continue

            batch = dataset[i : i + batch_size]

            # Step 1: Generate blind views
            blind_views = []
            for sample in batch:
                blind_view = self.multiview_gen.generate_single_view(
                    # Blind view: do NOT condition on ground-truth patch.
                    sample["buggy_code"],
                    view_type,
                    None,
                )
                blind_views.append(blind_view)

            # Step 2: Blind distillation (without patch)
            blind_distilled_views = []
            for j, sample in enumerate(batch):
                blind_distilled = self.distillation._distill_without_patch(
                    blind_views[j],
                    view_type,
                )
                blind_distilled_views.append(blind_distilled)

            # Step 3: Quality assessment & conditional patch-guided refinement
            distilled_views = []
            quality_stats = {"accurate": 0, "refined": 0}
            
            for j, sample in enumerate(batch):
                # Assess quality of blind distillation
                assessment = self.distillation.assess_quality(
                    blind_distilled_views[j],
                    sample["buggy_code"],
                    view_type,
                )
                
                if assessment["accurate"]:
                    # Blind distillation is good enough, use it directly
                    final_view = blind_distilled_views[j]
                    quality_stats["accurate"] += 1
                    logger.debug(
                        f"Sample {j}: Blind distillation accepted "
                        f"(confidence={assessment['confidence']:.2f})"
                    )
                else:
                    # Need patch-guided refinement
                    if sample.get("patch"):
                        final_view = self.distillation._refine_with_patch(
                            blind_views[j],
                            sample["buggy_code"],
                            sample["patch"],
                            view_type,
                        )
                        quality_stats["refined"] += 1
                        logger.debug(
                            f"Sample {j}: Applied patch-guided refinement "
                            f"(reason: {assessment['reasoning']})"
                        )
                    else:
                        # No patch available, use blind distillation anyway
                        final_view = blind_distilled_views[j]
                        quality_stats["accurate"] += 1
                        logger.warning(
                            f"Sample {j}: No patch for refinement, using blind distillation"
                        )
                
                distilled_views.append(final_view)
            
            # Log quality statistics for this batch
            logger.info(
                f"Batch quality: {quality_stats['accurate']} accepted, "
                f"{quality_stats['refined']} refined (total {len(batch)})"
            )

            # Step 4: Generate embeddings
            embeddings = self._generate_embeddings(distilled_views)

            # Step 5: Prepare metadata
            batch_metadata: List[Dict] = []
            for j, sample in enumerate(batch):
                metadata = {
                    "sample_id": sample.get("id", i + j),
                    "buggy_code": sample["buggy_code"],
                    "patch": sample.get("patch"),
                    "distilled_view": distilled_views[j],
                    "view_type": view_type,
                }
                batch_metadata.append(metadata)

            # Persist batch cache (for real resume without recomputation)
            if cache_dir is not None:
                try:
                    self._save_batch_cache(cache_dir, batch_idx, embeddings, batch_metadata)
                    cached_batches.add(batch_idx)
                except Exception as e:
                    logger.warning(f"Failed to save batch cache for {view_type} batch {batch_idx}: {e}")

            # Save checkpoint after each batch
            if save_dir:
                checkpoint = {
                    "current_view": view_type,
                    "batch_idx": batch_idx + 1,  # Next batch to process
                    "total_batches": len(batch_indices),
                    "samples_processed": (batch_idx + 1) * batch_size,
                    "total_samples": len(dataset),
                }
                checkpoint_path = Path(save_dir) / "checkpoint.json"
                try:
                    with open(checkpoint_path, "w") as f:
                        json.dump(checkpoint, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")

        # Assemble cached batches to build the index for this view
        if cache_dir is None:
            raise ValueError("save_dir is required for resume-capable offline indexing")

        missing = [b for b in range(len(batch_indices)) if b not in cached_batches]
        if missing:
            # At this point, missing batches means we could not cache them (disk error)
            # or the process terminated before finishing. Fail fast with a clear message.
            raise RuntimeError(
                f"Missing cached batches for view '{view_type}': {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
                + f". Cache dir: {cache_dir}"
            )

        all_embeddings_parts: List[np.ndarray] = []
        all_metadata: List[Dict] = []
        for b in range(len(batch_indices)):
            emb, meta = self._load_batch_cache(cache_dir, b)
            all_embeddings_parts.append(emb)
            all_metadata.extend(meta)

        all_embeddings = np.vstack(all_embeddings_parts) if all_embeddings_parts else np.zeros((0, self.vector_store.dimension))
        if len(all_metadata) == 0:
            raise RuntimeError(f"No metadata assembled for view '{view_type}'")

        # Add to vector store (train+add)
        self.vector_store.add_vectors(view_type, all_embeddings, all_metadata)

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts

        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.vector_store.dimension))

    def build_incremental(
        self,
        new_samples: List[Dict[str, str]],
        load_dir: Path,
        save_dir: Optional[Path] = None,
    ):
        """
        Incrementally add new samples to existing indices.

        Args:
            new_samples: New samples to add
            load_dir: Directory with existing indices
            save_dir: Directory to save updated indices (defaults to load_dir)
        """
        if save_dir is None:
            save_dir = load_dir

        logger.info(f"Loading existing indices from {load_dir}")
        self.vector_store.load(load_dir)

        logger.info(f"Adding {len(new_samples)} new samples")
        self.build_from_dataset(new_samples, save_dir)

    def get_index_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all indices.

        Returns:
            Dictionary mapping view types to their statistics
        """
        stats = {}
        for view_type in self.multiview_gen.VIEW_TYPES:
            stats[view_type] = self.vector_store.get_statistics(view_type)
        return stats
