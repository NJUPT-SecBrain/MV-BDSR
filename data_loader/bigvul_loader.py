"""Big-Vul dataset loader."""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger


class BigVulLoader:
    """Loader for Big-Vul vulnerability dataset."""

    def __init__(self, data_path: Path):
        """
        Initialize Big-Vul loader.

        Args:
            data_path: Path to Big-Vul dataset file
        """
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """
        Load Big-Vul dataset from file.

        Returns:
            DataFrame containing vulnerability data
        """
        logger.info(f"Loading Big-Vul dataset from {self.data_path}")
        
        # TODO: Implement actual loading logic based on dataset format
        # This is a placeholder
        if self.data_path.suffix == ".csv":
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.suffix == ".json":
            self.data = pd.read_json(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        original_count = len(self.data)
        logger.info(f"Loaded {original_count} vulnerability samples")
        
        # 过滤掉 buggy_code 超长的样本，避免超过 LLM context 限制
        if "buggy_code" in self.data.columns:
            # 估算：50000 字符约等于 12500 tokens，为 128k 上下文留足空间
            max_code_length = 50000
            before_filter = len(self.data)
            self.data = self.data[
                self.data["buggy_code"].fillna("").str.len() <= max_code_length
            ].copy()
            filtered_count = before_filter - len(self.data)
            if filtered_count > 0:
                logger.warning(
                    f"Filtered out {filtered_count} samples with buggy_code > {max_code_length} chars"
                )
        
        logger.info(f"Final dataset size: {len(self.data)} samples")
        return self.data

    def filter_by_language(self, language: str = "C") -> pd.DataFrame:
        """
        Filter dataset by programming language.

        Args:
            language: Target programming language

        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        filtered = self.data[self.data["language"] == language]
        logger.info(f"Filtered to {len(filtered)} {language} samples")
        return filtered

    def get_sample(self, index: int) -> Dict:
        """
        Get a single vulnerability sample.

        Args:
            index: Sample index

        Returns:
            Dictionary containing sample data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        row = self.data.iloc[index]
        return {
            "buggy_code": row.get("buggy_code", ""),
            "fixed_code": row.get("fixed_code", ""),
            "patch": row.get("patch", ""),
            "cve_id": row.get("cve_id", ""),
            "cwe_id": row.get("cwe_id", ""),
        }

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets.

        Args:
            test_size: Proportion of test set
            random_state: Random seed

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            self.data, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Split: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df
