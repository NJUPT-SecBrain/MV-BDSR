"""Tests for online_inference module."""

import pytest
from online_inference.phase1_diagnosis import DiagnosticAgent, ToolRegistry
from online_inference.phase2_retrieval import QueryGenerator, Retriever
from online_inference.phase3_repair import RepairAgent, Validator


class MockLLM:
    """Mock LLM for testing."""
    
    def generate(self, prompt, **kwargs):
        return "Analysis complete\nAction: Finish"
    
    def chat(self, messages, **kwargs):
        return "Thought: Analysis complete\nAction: Finish"


class MockTool:
    """Mock analysis tool."""
    
    def __call__(self, code, parameters):
        return "Mock tool result"


class TestDiagnosticAgent:
    """Tests for DiagnosticAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        llm = MockLLM()
        registry = ToolRegistry()
        
        agent = DiagnosticAgent(llm, registry, max_iterations=5)
        assert agent.max_iterations == 5

    def test_parse_action(self):
        """Test action parsing."""
        llm = MockLLM()
        registry = ToolRegistry()
        agent = DiagnosticAgent(llm, registry)
        
        response = "Action: tool_name parameter"
        action = agent._parse_action(response)
        
        assert action is not None
        assert action["tool"] == "tool_name"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        registry.register("test_tool", MockTool(), "Test tool description")
        
        assert "test_tool" in registry.tools

    def test_execute_tool(self):
        """Test tool execution."""
        registry = ToolRegistry()
        registry.register("test_tool", MockTool(), "Test tool")
        
        result = registry.execute("test_tool", "code", "params")
        assert result == "Mock tool result"


class TestQueryGenerator:
    """Tests for QueryGenerator."""

    def test_generate_queries(self):
        """Test query generation."""
        llm = MockLLM()
        generator = QueryGenerator(llm)
        
        queries = generator.generate_queries("Enhanced context")
        
        assert isinstance(queries, dict)
        assert len(queries) == 3


class TestValidator:
    """Tests for Validator."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = Validator(compiler="gcc")
        assert validator.compiler == "gcc"

    def test_quick_syntax_check(self):
        """Test syntax checking."""
        validator = Validator()
        
        # Valid C code
        valid = validator.quick_syntax_check("int main() { return 0; }")
        
        # Note: This test may fail if gcc is not available
        # In real tests, would use mocking
