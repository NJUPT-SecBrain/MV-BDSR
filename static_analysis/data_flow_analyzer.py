"""Data flow analysis tool."""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from loguru import logger


class DataFlowAnalyzer:
    """Analyzer for data flow in source code."""

    def __init__(self):
        """Initialize data flow analyzer."""
        self.definitions: Dict[str, List[int]] = defaultdict(list)
        self.uses: Dict[str, List[int]] = defaultdict(list)
        self.def_use_chains: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def analyze(self, code: str, ast: Optional[Dict] = None) -> Dict:
        """
        Perform data flow analysis on code.

        Args:
            code: Source code
            ast: Optional AST representation

        Returns:
            Data flow analysis results
        """
        # TODO: Implement full data flow analysis
        # This is a simplified version
        lines = code.split("\n")
        
        for line_num, line in enumerate(lines, 1):
            self._analyze_line(line, line_num)

        return {
            "definitions": dict(self.definitions),
            "uses": dict(self.uses),
            "def_use_chains": dict(self.def_use_chains),
            "reaching_definitions": self._compute_reaching_definitions(),
        }

    def _analyze_line(self, line: str, line_num: int):
        """
        Analyze a single line for data flow.

        Args:
            line: Line of code
            line_num: Line number
        """
        # Simple heuristic: look for assignments (definitions)
        if "=" in line and not "==" in line:
            # Likely a definition
            parts = line.split("=")
            if len(parts) >= 2:
                var = parts[0].strip().split()[-1]  # Get last token before =
                var = var.strip("*&")  # Remove pointer/reference symbols
                if var.isidentifier():
                    self.definitions[var].append(line_num)

        # Look for variable uses (simple heuristic)
        # In production, use proper AST analysis
        import re
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line)
        for ident in identifiers:
            if ident not in ["if", "else", "for", "while", "return", "int", "char", "void"]:
                self.uses[ident].append(line_num)

    def _compute_reaching_definitions(self) -> Dict[str, Set[int]]:
        """
        Compute reaching definitions for each variable.

        Returns:
            Dictionary mapping variables to reaching definition line numbers
        """
        reaching = {}
        for var, def_lines in self.definitions.items():
            # Simplified: all definitions reach
            # In full analysis, need to consider control flow
            reaching[var] = set(def_lines)
        return reaching

    def get_def_use_chain(self, variable: str) -> List[Tuple[int, int]]:
        """
        Get definition-use chain for a variable.

        Args:
            variable: Variable name

        Returns:
            List of (definition_line, use_line) tuples
        """
        chains = []
        defs = self.definitions.get(variable, [])
        uses = self.uses.get(variable, [])

        for def_line in defs:
            for use_line in uses:
                if use_line > def_line:
                    chains.append((def_line, use_line))

        return chains

    def find_undefined_uses(self) -> Dict[str, List[int]]:
        """
        Find variables used before definition.

        Returns:
            Dictionary of variables and their undefined use line numbers
        """
        undefined = {}
        
        for var, use_lines in self.uses.items():
            def_lines = self.definitions.get(var, [])
            
            if not def_lines:
                # Variable used but never defined
                undefined[var] = use_lines
            else:
                first_def = min(def_lines)
                early_uses = [line for line in use_lines if line < first_def]
                if early_uses:
                    undefined[var] = early_uses

        return undefined

    def get_dependencies(self, variable: str) -> Set[str]:
        """
        Get variables that the given variable depends on.

        Args:
            variable: Variable name

        Returns:
            Set of variable names it depends on
        """
        # TODO: Implement dependency analysis
        # Requires analyzing RHS of assignments
        return set()

    def visualize_flow(self) -> str:
        """
        Generate a text visualization of data flow.

        Returns:
            String representation of data flow
        """
        output = ["Data Flow Analysis:", "=" * 50]
        
        for var in sorted(self.definitions.keys()):
            output.append(f"\nVariable: {var}")
            output.append(f"  Definitions: {self.definitions[var]}")
            output.append(f"  Uses: {self.uses[var]}")
            
            chains = self.get_def_use_chain(var)
            if chains:
                output.append(f"  Def-Use Chains: {chains}")

        return "\n".join(output)
