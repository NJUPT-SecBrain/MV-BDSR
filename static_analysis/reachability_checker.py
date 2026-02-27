"""Reachability analysis tool."""

from typing import Dict, List, Set, Optional
from loguru import logger


class ReachabilityChecker:
    """Tool for checking code reachability and finding dead code."""

    def __init__(self):
        """Initialize reachability checker."""
        self.reachable_lines: Set[int] = set()
        self.unreachable_lines: Set[int] = set()

    def check_reachability(self, code: str, cfg: Optional[Dict] = None) -> Dict:
        """
        Check code reachability.

        Args:
            code: Source code
            cfg: Optional control flow graph

        Returns:
            Reachability analysis results
        """
        lines = code.split("\n")
        total_lines = len(lines)

        if cfg is not None:
            # Use provided CFG
            return self._analyze_with_cfg(cfg, total_lines)
        else:
            # Heuristic-based analysis
            return self._analyze_heuristic(lines)

    def _analyze_with_cfg(self, cfg: Dict, total_lines: int) -> Dict:
        """
        Analyze reachability using CFG.

        Args:
            cfg: Control flow graph
            total_lines: Total number of lines

        Returns:
            Analysis results
        """
        # Extract reachable nodes from CFG
        reachable_nodes = cfg.get("nodes", [])
        
        # Extract line numbers from nodes
        for node in reachable_nodes:
            if isinstance(node, str) and node.startswith("L"):
                try:
                    line_num = int(node[1:])
                    self.reachable_lines.add(line_num)
                except ValueError:
                    pass

        # Find unreachable lines
        all_lines = set(range(1, total_lines + 1))
        self.unreachable_lines = all_lines - self.reachable_lines

        return self._build_results()

    def _analyze_heuristic(self, lines: List[str]) -> Dict:
        """
        Heuristic-based reachability analysis.

        Args:
            lines: Lines of code

        Returns:
            Analysis results
        """
        # Simple heuristic: detect unreachable code after return/break/continue
        in_unreachable_region = False
        scope_depth = 0

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue

            # Track scope depth
            scope_depth += stripped.count("{") - stripped.count("}")

            if in_unreachable_region:
                # Check if we exit the unreachable region (new scope)
                if scope_depth <= 0 or stripped.endswith("}"):
                    in_unreachable_region = False
                    self.reachable_lines.add(i)
                else:
                    self.unreachable_lines.add(i)
            else:
                self.reachable_lines.add(i)

                # Check for unconditional control flow statements
                if any(keyword in stripped for keyword in ["return", "break", "continue", "exit"]):
                    # Next lines might be unreachable until scope closes
                    if not stripped.endswith("}"):
                        in_unreachable_region = True

        return self._build_results()

    def _build_results(self) -> Dict:
        """
        Build analysis results dictionary.

        Returns:
            Results dictionary
        """
        return {
            "reachable_lines": sorted(list(self.reachable_lines)),
            "unreachable_lines": sorted(list(self.unreachable_lines)),
            "num_reachable": len(self.reachable_lines),
            "num_unreachable": len(self.unreachable_lines),
            "reachability_ratio": (
                len(self.reachable_lines) / (len(self.reachable_lines) + len(self.unreachable_lines))
                if (len(self.reachable_lines) + len(self.unreachable_lines)) > 0
                else 0.0
            ),
        }

    def is_line_reachable(self, line_num: int) -> bool:
        """
        Check if a specific line is reachable.

        Args:
            line_num: Line number to check

        Returns:
            True if reachable, False otherwise
        """
        return line_num in self.reachable_lines

    def find_dead_code_blocks(self, code: str) -> List[Dict]:
        """
        Find blocks of dead code.

        Args:
            code: Source code

        Returns:
            List of dead code blocks with start/end lines
        """
        if not self.unreachable_lines:
            return []

        # Group consecutive unreachable lines into blocks
        blocks = []
        sorted_unreachable = sorted(self.unreachable_lines)
        
        if not sorted_unreachable:
            return blocks

        start = sorted_unreachable[0]
        end = start

        for line_num in sorted_unreachable[1:]:
            if line_num == end + 1:
                end = line_num
            else:
                blocks.append({"start_line": start, "end_line": end})
                start = line_num
                end = start

        # Add last block
        blocks.append({"start_line": start, "end_line": end})

        return blocks

    def get_report(self) -> str:
        """
        Generate reachability report.

        Returns:
            Report string
        """
        output = ["Reachability Analysis Report:", "=" * 50]
        
        output.append(f"\nTotal reachable lines: {len(self.reachable_lines)}")
        output.append(f"Total unreachable lines: {len(self.unreachable_lines)}")
        
        if self.unreachable_lines:
            output.append(f"\nUnreachable lines: {sorted(self.unreachable_lines)}")
            
            blocks = self.find_dead_code_blocks("")
            if blocks:
                output.append("\nDead code blocks:")
                for block in blocks:
                    output.append(f"  Lines {block['start_line']}-{block['end_line']}")

        return "\n".join(output)
