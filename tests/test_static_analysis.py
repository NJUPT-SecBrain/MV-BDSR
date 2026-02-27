"""Tests for static_analysis module."""

import pytest
from static_analysis import (
    DataFlowAnalyzer,
    ControlFlowAnalyzer,
    ReachabilityChecker,
)


class TestDataFlowAnalyzer:
    """Tests for DataFlowAnalyzer."""

    def test_analyze_simple_code(self):
        """Test data flow analysis on simple code."""
        analyzer = DataFlowAnalyzer()
        code = """
int x = 5;
int y = x + 1;
return y;
"""
        result = analyzer.analyze(code)
        
        assert "definitions" in result
        assert "uses" in result

    def test_def_use_chain(self):
        """Test definition-use chain."""
        analyzer = DataFlowAnalyzer()
        code = "int x = 5;\nint y = x;"
        
        analyzer.analyze(code)
        chain = analyzer.get_def_use_chain("x")
        
        assert isinstance(chain, list)


class TestControlFlowAnalyzer:
    """Tests for ControlFlowAnalyzer."""

    def test_analyze_code(self):
        """Test control flow analysis."""
        analyzer = ControlFlowAnalyzer()
        code = """
if (x > 0) {
    return 1;
} else {
    return 0;
}
"""
        result = analyzer.analyze(code)
        
        assert "nodes" in result
        assert "edges" in result

    def test_find_cycles(self):
        """Test cycle detection."""
        analyzer = ControlFlowAnalyzer()
        code = "while (true) { break; }"
        
        analyzer.analyze(code)
        cycles = analyzer.find_cycles()
        
        assert isinstance(cycles, list)


class TestReachabilityChecker:
    """Tests for ReachabilityChecker."""

    def test_reachability_check(self):
        """Test reachability analysis."""
        checker = ReachabilityChecker()
        code = """
return 1;
int x = 5;
"""
        result = checker.check_reachability(code)
        
        assert "reachable_lines" in result
        assert "unreachable_lines" in result

    def test_dead_code_detection(self):
        """Test dead code detection."""
        checker = ReachabilityChecker()
        code = "return 1;\nint x = 5;"
        
        checker.check_reachability(code)
        blocks = checker.find_dead_code_blocks(code)
        
        assert isinstance(blocks, list)
