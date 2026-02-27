"""Wrapper for Tree-sitter parsing."""

from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

try:
    from tree_sitter import Language, Parser
except ImportError:
    logger.warning("tree-sitter not installed. Install with: pip install tree-sitter")
    Language = None
    Parser = None


class TreeSitterAnalyzer:
    """Wrapper for Tree-sitter code parsing."""

    def __init__(self, language: str = "c"):
        """
        Initialize Tree-sitter analyzer.

        Args:
            language: Programming language (c, cpp, java, etc.)
        """
        self.language_name = language
        self.parser = None
        self.language = None

        if Parser is not None:
            self._initialize_parser()

    def _initialize_parser(self):
        """Initialize Tree-sitter parser for the specified language."""
        try:
            # TODO: Build language libraries first
            # This requires running: tree-sitter build
            # For now, this is a placeholder
            logger.warning("Tree-sitter language library not built. Run setup first.")
            # self.language = Language('build/languages.so', self.language_name)
            # self.parser = Parser()
            # self.parser.set_language(self.language)
        except Exception as e:
            logger.error(f"Failed to initialize Tree-sitter: {e}")

    def parse(self, code: str) -> Optional[object]:
        """
        Parse code into AST.

        Args:
            code: Source code

        Returns:
            Tree-sitter tree object
        """
        if self.parser is None:
            logger.error("Parser not initialized")
            return None

        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            return None

    def get_ast_dict(self, code: str) -> Optional[Dict]:
        """
        Get AST as dictionary.

        Args:
            code: Source code

        Returns:
            AST in dictionary format
        """
        tree = self.parse(code)
        if tree is None:
            return None

        return self._tree_to_dict(tree.root_node)

    def _tree_to_dict(self, node) -> Dict:
        """
        Convert Tree-sitter node to dictionary.

        Args:
            node: Tree-sitter node

        Returns:
            Dictionary representation
        """
        result = {
            "type": node.type,
            "start_point": node.start_point,
            "end_point": node.end_point,
        }

        if node.children:
            result["children"] = [self._tree_to_dict(child) for child in node.children]

        return result

    def extract_functions(self, code: str) -> List[Dict]:
        """
        Extract all function definitions.

        Args:
            code: Source code

        Returns:
            List of function definitions
        """
        tree = self.parse(code)
        if tree is None:
            return []

        functions = []
        self._find_functions(tree.root_node, code.encode("utf8"), functions)
        return functions

    def _find_functions(self, node, source_code: bytes, functions: List[Dict]):
        """
        Recursively find function definitions.

        Args:
            node: Current AST node
            source_code: Source code as bytes
            functions: List to accumulate results
        """
        if node.type == "function_definition":
            func_dict = {
                "name": self._get_function_name(node, source_code),
                "start_line": node.start_point[0],
                "end_line": node.end_point[0],
                "code": source_code[node.start_byte : node.end_byte].decode("utf8"),
            }
            functions.append(func_dict)

        for child in node.children:
            self._find_functions(child, source_code, functions)

    def _get_function_name(self, node, source_code: bytes) -> str:
        """
        Extract function name from function definition node.

        Args:
            node: Function definition node
            source_code: Source code as bytes

        Returns:
            Function name
        """
        for child in node.children:
            if child.type == "function_declarator":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        return source_code[subchild.start_byte : subchild.end_byte].decode(
                            "utf8"
                        )
        return "unknown"

    def extract_variables(self, code: str) -> List[str]:
        """
        Extract all variable declarations.

        Args:
            code: Source code

        Returns:
            List of variable names
        """
        tree = self.parse(code)
        if tree is None:
            return []

        variables = []
        self._find_variables(tree.root_node, code.encode("utf8"), variables)
        return list(set(variables))  # Remove duplicates

    def _find_variables(self, node, source_code: bytes, variables: List[str]):
        """
        Recursively find variable declarations.

        Args:
            node: Current AST node
            source_code: Source code as bytes
            variables: List to accumulate results
        """
        if node.type in ["declaration", "variable_declaration"]:
            for child in node.children:
                if child.type == "identifier":
                    var_name = source_code[child.start_byte : child.end_byte].decode("utf8")
                    variables.append(var_name)

        for child in node.children:
            self._find_variables(child, source_code, variables)
