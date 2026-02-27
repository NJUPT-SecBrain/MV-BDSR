"""Wrapper for Joern static analysis tool."""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import requests


class JoernAnalyzer:
    """Wrapper for Joern code analysis platform."""

    def __init__(self, server_url: str = "http://localhost:8080", use_server: bool = True):
        """
        Initialize Joern analyzer.

        Args:
            server_url: URL of Joern server
            use_server: Whether to use Joern server or CLI
        """
        self.server_url = server_url
        self.use_server = use_server

    def analyze_code(self, code: str, language: str = "c") -> Dict:
        """
        Analyze code using Joern.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            Analysis results
        """
        if self.use_server:
            return self._analyze_via_server(code, language)
        else:
            return self._analyze_via_cli(code, language)

    def _analyze_via_server(self, code: str, language: str) -> Dict:
        """
        Analyze code via Joern server API.

        Args:
            code: Source code
            language: Programming language

        Returns:
            Analysis results
        """
        try:
            response = requests.post(
                f"{self.server_url}/analyze",
                json={"code": code, "language": language},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Joern server analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_via_cli(self, code: str, language: str) -> Dict:
        """
        Analyze code via Joern CLI.

        Args:
            code: Source code
            language: Programming language

        Returns:
            Analysis results
        """
        # TODO: Implement Joern CLI analysis
        # This requires writing code to temp file and calling joern-parse
        logger.warning("Joern CLI analysis not yet implemented")
        return {"error": "CLI analysis not implemented"}

    def get_cpg(self, code: str) -> Optional[Dict]:
        """
        Get Code Property Graph (CPG) for code.

        Args:
            code: Source code

        Returns:
            CPG representation
        """
        result = self.analyze_code(code)
        return result.get("cpg")

    def get_data_flow(self, code: str, variable: Optional[str] = None) -> List[Dict]:
        """
        Extract data flow information.

        Args:
            code: Source code
            variable: Specific variable to track (optional)

        Returns:
            List of data flow edges
        """
        result = self.analyze_code(code)
        data_flow = result.get("data_flow", [])

        if variable:
            # Filter for specific variable
            data_flow = [
                edge for edge in data_flow if edge.get("variable") == variable
            ]

        return data_flow

    def get_control_flow(self, code: str) -> List[Dict]:
        """
        Extract control flow graph.

        Args:
            code: Source code

        Returns:
            Control flow graph edges
        """
        result = self.analyze_code(code)
        return result.get("control_flow", [])

    def find_vulnerabilities(self, code: str) -> List[Dict]:
        """
        Find potential vulnerabilities using Joern queries.

        Args:
            code: Source code

        Returns:
            List of vulnerability findings
        """
        result = self.analyze_code(code)
        return result.get("vulnerabilities", [])
