"""Control flow analysis tool."""

from typing import Dict, List, Set, Optional
import networkx as nx
from loguru import logger


class ControlFlowAnalyzer:
    """Analyzer for control flow graphs."""

    def __init__(self):
        """Initialize control flow analyzer."""
        self.cfg: nx.DiGraph = nx.DiGraph()
        self.entry_node = "ENTRY"
        self.exit_node = "EXIT"

    def analyze(self, code: str, ast: Optional[Dict] = None) -> Dict:
        """
        Build control flow graph from code.

        Args:
            code: Source code
            ast: Optional AST representation

        Returns:
            Control flow analysis results
        """
        self.cfg.clear()
        self.cfg.add_node(self.entry_node)
        self.cfg.add_node(self.exit_node)

        # Simplified CFG construction
        lines = [line for line in code.split("\n") if line.strip()]
        
        if not lines:
            self.cfg.add_edge(self.entry_node, self.exit_node)
            return self.get_analysis_results()

        # Build linear CFG (simplified)
        self._build_cfg(lines)

        return self.get_analysis_results()

    def _build_cfg(self, lines: List[str]):
        """
        Build control flow graph from lines.

        Args:
            lines: Lines of code
        """
        # Simplified CFG: just linear flow with basic branching detection
        prev_node = self.entry_node

        for i, line in enumerate(lines):
            node_id = f"L{i+1}"
            self.cfg.add_node(node_id, line=line.strip(), line_num=i+1)
            self.cfg.add_edge(prev_node, node_id)

            # Detect control structures (simplified)
            if "if" in line or "while" in line or "for" in line:
                # Branch point - in full version, add conditional edges
                pass
            elif "return" in line:
                # Connect to exit
                self.cfg.add_edge(node_id, self.exit_node)
                prev_node = None  # Dead code after return
                continue

            prev_node = node_id

        # Connect last node to exit if not already connected
        if prev_node:
            self.cfg.add_edge(prev_node, self.exit_node)

    def get_analysis_results(self) -> Dict:
        """
        Get analysis results.

        Returns:
            Dictionary containing CFG analysis
        """
        return {
            "nodes": list(self.cfg.nodes()),
            "edges": list(self.cfg.edges()),
            "num_nodes": self.cfg.number_of_nodes(),
            "num_edges": self.cfg.number_of_edges(),
            "dominators": self._compute_dominators(),
            "post_dominators": self._compute_post_dominators(),
            "control_dependencies": self._compute_control_dependencies(),
        }

    def _compute_dominators(self) -> Dict[str, Set[str]]:
        """
        Compute dominator tree.

        Returns:
            Dictionary mapping nodes to their dominators
        """
        try:
            return {
                node: set(nx.dominance.immediate_dominators(self.cfg, self.entry_node).keys())
                for node in self.cfg.nodes()
                if node != self.entry_node
            }
        except Exception as e:
            logger.warning(f"Dominator computation failed: {e}")
            return {}

    def _compute_post_dominators(self) -> Dict[str, Set[str]]:
        """
        Compute post-dominator tree.

        Returns:
            Dictionary mapping nodes to their post-dominators
        """
        try:
            reversed_cfg = self.cfg.reverse()
            return {
                node: set(nx.dominance.immediate_dominators(reversed_cfg, self.exit_node).keys())
                for node in reversed_cfg.nodes()
                if node != self.exit_node
            }
        except Exception as e:
            logger.warning(f"Post-dominator computation failed: {e}")
            return {}

    def _compute_control_dependencies(self) -> Dict[str, Set[str]]:
        """
        Compute control dependencies.

        Returns:
            Dictionary mapping nodes to nodes they control-depend on
        """
        # Simplified control dependency computation
        # Full version requires post-dominance frontier
        control_deps = {node: set() for node in self.cfg.nodes()}
        return control_deps

    def get_reachable_nodes(self, start_node: str) -> Set[str]:
        """
        Get all nodes reachable from start node.

        Args:
            start_node: Starting node

        Returns:
            Set of reachable node IDs
        """
        if start_node not in self.cfg:
            return set()

        return set(nx.descendants(self.cfg, start_node)) | {start_node}

    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in control flow graph.

        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        try:
            return list(nx.simple_cycles(self.cfg))
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")
            return []

    def visualize(self) -> str:
        """
        Generate text visualization of CFG.

        Returns:
            String representation
        """
        output = ["Control Flow Graph:", "=" * 50]
        
        for node in self.cfg.nodes():
            successors = list(self.cfg.successors(node))
            output.append(f"{node} -> {successors}")

        return "\n".join(output)

    def export_dot(self, output_path: str):
        """
        Export CFG to DOT format for visualization.

        Args:
            output_path: Path to save DOT file
        """
        try:
            from networkx.drawing.nx_pydot import write_dot
            write_dot(self.cfg, output_path)
            logger.info(f"CFG exported to {output_path}")
        except ImportError:
            logger.error("pydot not installed. Install with: pip install pydot")
