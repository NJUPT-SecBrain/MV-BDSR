"""Tool registry for diagnostic agent."""

from typing import Dict, Callable, Any
from loguru import logger


class ToolRegistry:
    """Registry of analysis tools for diagnostic agent."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Callable] = {}
        self.descriptions: Dict[str, str] = {}

    def register(self, name: str, func: Callable, description: str):
        """
        Register a tool.

        Args:
            name: Tool name
            func: Tool function (takes code and parameters)
            description: Tool description
        """
        self.tools[name] = func
        self.descriptions[name] = description
        logger.debug(f"Registered tool: {name}")

    def execute(self, name: str, code: str, parameters: str = "") -> Any:
        """
        Execute a tool.

        Args:
            name: Tool name
            code: Code to analyze
            parameters: Tool parameters

        Returns:
            Tool result
        """
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")

        logger.debug(f"Executing tool: {name}")
        return self.tools[name](code, parameters)

    def get_tool_descriptions(self) -> str:
        """
        Get formatted tool descriptions.

        Returns:
            String describing all tools
        """
        descriptions = []
        for name, desc in self.descriptions.items():
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)

    def list_tools(self) -> list:
        """
        List available tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())


def create_default_tool_registry(
    data_flow_analyzer,
    control_flow_analyzer,
    reachability_checker,
) -> ToolRegistry:
    """
    Create tool registry with default analysis tools.

    Args:
        data_flow_analyzer: DataFlowAnalyzer instance
        control_flow_analyzer: ControlFlowAnalyzer instance
        reachability_checker: ReachabilityChecker instance

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()

    # Data flow analysis tool
    def data_flow_tool(code: str, parameters: str) -> str:
        result = data_flow_analyzer.analyze(code)
        return data_flow_analyzer.visualize_flow()

    registry.register(
        "data_flow_analyzer",
        data_flow_tool,
        "Analyze data flow, variable definitions and uses",
    )

    # Control flow analysis tool
    def control_flow_tool(code: str, parameters: str) -> str:
        result = control_flow_analyzer.analyze(code)
        return control_flow_analyzer.visualize()

    registry.register(
        "control_flow_analyzer",
        control_flow_tool,
        "Analyze control flow graph and dependencies",
    )

    # Reachability checker tool
    def reachability_tool(code: str, parameters: str) -> str:
        result = reachability_checker.check_reachability(code)
        return reachability_checker.get_report()

    registry.register(
        "reachability_checker",
        reachability_tool,
        "Check code reachability and find dead code",
    )

    # Variable tracker tool
    def variable_tracker_tool(code: str, parameters: str) -> str:
        var_name = parameters.strip() if parameters else None
        result = data_flow_analyzer.analyze(code)
        
        if var_name:
            chains = data_flow_analyzer.get_def_use_chain(var_name)
            return f"Def-use chains for '{var_name}': {chains}"
        else:
            return f"All variables: {list(data_flow_analyzer.definitions.keys())}"

    registry.register(
        "variable_tracker",
        variable_tracker_tool,
        "Track specific variable usage (parameter: variable_name)",
    )

    # Dependency analyzer tool
    def dependency_tool(code: str, parameters: str) -> str:
        cfg_result = control_flow_analyzer.analyze(code)
        df_result = data_flow_analyzer.analyze(code)
        
        return f"Control dependencies: {cfg_result.get('control_dependencies', {})}\n" \
               f"Data dependencies: {df_result.get('def_use_chains', {})}"

    registry.register(
        "dependency_analyzer",
        dependency_tool,
        "Analyze control and data dependencies",
    )

    return registry
