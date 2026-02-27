"""General helper functions."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict
from loguru import logger


def load_yaml(file_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with configuration
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Loaded YAML from {file_path}")
    return config


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to JSON file
        indent: Indentation level
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.debug(f"Saved JSON to {file_path}")


def load_json(file_path: str) -> Any:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from {file_path}")
    return data


def ensure_dir(dir_path: str):
    """
    Ensure directory exists.

    Args:
        dir_path: Directory path
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def read_file(file_path: str) -> str:
    """
    Read text file.

    Args:
        file_path: File path

    Returns:
        File contents
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str):
    """
    Write text file.

    Args:
        file_path: File path
        content: Content to write
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_code_snippet(code: str, max_lines: int = 10) -> str:
    """
    Format code snippet for display.

    Args:
        code: Source code
        max_lines: Maximum lines to show

    Returns:
        Formatted code
    """
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return code

    shown_lines = lines[:max_lines]
    remaining = len(lines) - max_lines
    return "\n".join(shown_lines) + f"\n... ({remaining} more lines)"


def compute_file_hash(file_path: str) -> str:
    """
    Compute hash of file.

    Args:
        file_path: File path

    Returns:
        Hash string
    """
    import hashlib

    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    return file_hash
