"""向量存储（使用 FAISS）

说明：
=====
这个模块不涉及 LLM API 调用，但会在索引构建完成后
将向量和元数据保存到磁盘，用于在线检索阶段。
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

try:
    import faiss
except ImportError:
    logger.warning("FAISS 未安装。请运行: pip install faiss-cpu")
    faiss = None


class VectorStore:
    """向量存储（多视角索引）"""

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "Flat",
        metric: str = "L2",
        nlist: int = 100,
    ):
        """
        初始化向量存储
        
        Args:
            dimension: 向量维度
            index_type: FAISS 索引类型 (Flat, IVFFlat, HNSW 等)
            metric: 距离度量 (L2 或 IP 内积)
            nlist: IVF 聚类数（仅 IVFFlat 使用）
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        
        self.indices: Dict[str, object] = {}  # 每个视角一个索引
        self.metadata: Dict[str, List[Dict]] = {}  # 每个向量的元数据

        if faiss is None:
            raise ImportError("需要 FAISS。请运行: pip install faiss-cpu")

    def create_index(self, view_type: str):
        """
        为某个视角创建新索引
        
        Args:
            view_type: 视角类型 (data_flow, control_flow, api_semantic)
        """
        logger.info(f"为 {view_type} 创建 {self.index_type} 索引")

        if self.index_type == "Flat":
            if self.metric == "L2":
                index = faiss.IndexFlatL2(self.dimension)
            else:  # IP (Inner Product)
                index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVFFlat":
            # IVF 倒排索引（更快的搜索）
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        self.indices[view_type] = index
        self.metadata[view_type] = []

    def add_vectors(
        self,
        view_type: str,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ):
        """
        添加向量到索引
        
        Args:
            view_type: 视角类型
            embeddings: 向量 embeddings (N x dimension)
            metadata: 每个向量的元数据列表
        """
        if view_type not in self.indices:
            self.create_index(view_type)

        index = self.indices[view_type]

        # 如果使用 IP 度量，归一化 embeddings
        if self.metric == "IP":
            faiss.normalize_L2(embeddings)

        # 训练索引（IVFFlat 需要）
        if self.index_type == "IVFFlat" and not index.is_trained:
            n_samples = len(embeddings)
            # FAISS IVFFlat 经验要求：样本数应至少为 nlist * 39 (通常为 30-40)
            # 否则训练可能失败（或触发 segmentation fault）
            min_required = self.nlist * 39
            
            if n_samples < min_required:
                logger.warning(
                    f"IVFFlat 训练样本不足：{n_samples} < {min_required} (nlist={self.nlist} × 39)。"
                    f"自动降级为 Flat 索引（{view_type}）"
                )
                # 主动降级到 Flat 索引（无需训练）
                if self.metric == "L2":
                    index = faiss.IndexFlatL2(self.dimension)
                else:
                    index = faiss.IndexFlatIP(self.dimension)
                self.indices[view_type] = index
            else:
                # 样本数足够，正常训练
                logger.info(f"用 {n_samples} 个样本训练 {view_type} 索引")
                try:
                    index.train(embeddings)
                except RuntimeError as e:
                    if "Number of training points" in str(e):
                        logger.error(
                            f"IVFFlat 训练失败：{e}。"
                            f"切换到 Flat 索引（{view_type}）"
                        )
                        # 降级到 Flat 索引
                        if self.metric == "L2":
                            index = faiss.IndexFlatL2(self.dimension)
                        else:
                            index = faiss.IndexFlatIP(self.dimension)
                        self.indices[view_type] = index
                    else:
                        raise

        # 添加向量
        index.add(embeddings)
        self.metadata[view_type].extend(metadata)

        logger.info(f"向 {view_type} 索引添加了 {len(embeddings)} 个向量")

    def _effective_index_type(self, index: object) -> str:
        """获取实际的索引类型（用于统计）"""
        try:
            name = index.__class__.__name__
        except Exception:
            name = str(type(index))
        if "IndexIVF" in name:
            return "IVF"
        if "IndexFlat" in name:
            return "Flat"
        return name

    def search(
        self,
        view_type: str,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        搜索最近邻
        
        Args:
            view_type: 要搜索的视角类型
            query_embedding: 查询向量 (1 x dimension)
            k: 返回的近邻数
            
        Returns:
            (distances, metadata_list) 元组
            
        Raises:
            KeyError: 如果视角的索引或元数据不存在
        """
        # 检查索引是否存在
        if view_type not in self.indices:
            available = list(self.indices.keys())
            raise KeyError(
                f"未找到 {view_type} 的索引文件。"
                f"可用视角: {available}。"
                f"提示: 使用 --views 参数重新构建所需视角的索引。"
            )
        
        # 检查元数据是否存在
        if view_type not in self.metadata:
            available = list(self.metadata.keys())
            raise KeyError(
                f"未找到 {view_type} 的元数据。"
                f"可用视角: {available}。"
                f"提示: 该视角的索引文件存在但元数据缺失，需要重新构建索引。"
            )

        index = self.indices[view_type]

        # 调整 query 形状
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # 如果使用 IP 度量，归一化
        if self.metric == "IP":
            faiss.normalize_L2(query_embedding)

        # 搜索
        distances, indices = index.search(query_embedding, k)

        # 获取元数据
        metadata_list = [
            self.metadata[view_type][idx]
            for idx in indices[0]
            if idx < len(self.metadata[view_type])
        ]

        return distances[0], metadata_list

    def save(self, save_dir: Path):
        """
        保存所有索引到磁盘
        
        Args:
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for view_type, index in self.indices.items():
            index_path = save_dir / f"index_{view_type}.faiss"
            faiss.write_index(index, str(index_path))
            logger.info(f"保存 {view_type} 索引到 {index_path}")

        # 保存元数据（合并写入，避免只跑单个视角时覆盖丢失其他视角）
        import pickle
        metadata_path = save_dir / "metadata.pkl"
        merged: Dict[str, List[Dict]] = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "rb") as f:
                    existing = pickle.load(f)
                if isinstance(existing, dict):
                    merged.update(existing)
            except Exception as e:
                logger.warning(f"读取已有元数据失败，将覆盖写入: {e}")

        # 用当前内存中的元数据覆盖/更新对应视角
        for view_type, meta_list in (self.metadata or {}).items():
            merged[view_type] = meta_list

        with open(metadata_path, "wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"保存元数据到 {metadata_path}（视角: {list(merged.keys())}）")

    def load(self, load_dir: Path):
        """
        从磁盘加载索引
        
        Args:
            load_dir: 包含保存的索引的目录
        """
        load_dir = Path(load_dir)

        # 先加载元数据（确定哪些视角可用）
        import pickle
        metadata_path = load_dir / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"从 {metadata_path} 加载元数据")
            logger.info(f"可用视角: {list(self.metadata.keys())}")
        
        # 只加载元数据中存在的视角的索引
        available_views = list(self.metadata.keys()) if self.metadata else []
        
        for view_type in ["data_flow", "control_flow", "api_semantic"]:
            index_path = load_dir / f"index_{view_type}.faiss"
            
            # 检查元数据和索引文件是否都存在
            if index_path.exists():
                if view_type in available_views:
                    self.indices[view_type] = faiss.read_index(str(index_path))
                    logger.info(f"从 {index_path} 加载 {view_type} 索引")
                else:
                    logger.warning(
                        f"跳过 {view_type} 索引：索引文件存在但元数据缺失 "
                        f"(可能需要重新构建该视角的索引)"
                    )

    def get_available_views(self) -> List[str]:
        """
        获取可用的视角列表（同时有索引和元数据的视角）
        
        Returns:
            可用视角列表
        """
        available = []
        for view_type in ["data_flow", "control_flow", "api_semantic"]:
            if view_type in self.indices and view_type in self.metadata:
                available.append(view_type)
        return available
    
    def get_statistics(self, view_type: str) -> Dict:
        """
        获取索引统计信息
        
        Args:
            view_type: 视角类型
            
        Returns:
            统计信息字典
        """
        if view_type not in self.indices:
            return {}

        index = self.indices[view_type]
        return {
            "view_type": view_type,
            "total_vectors": index.ntotal,
            "dimension": self.dimension,
            # 显示实际运行时的索引类型（可能因样本数不足而降级）
            "index_type": self._effective_index_type(index),
            "is_trained": getattr(index, "is_trained", True),
        }
