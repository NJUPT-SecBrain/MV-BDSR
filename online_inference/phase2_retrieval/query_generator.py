"""Multi-view query generation from enhanced context."""

from pathlib import Path
from typing import Dict, List
from loguru import logger


class QueryGenerator:
    """Generator for multi-view queries from enhanced diagnostic context."""

    VIEW_TYPES = ["data_flow", "control_flow", "api_semantic"]

    def __init__(self, llm_interface):
        """
        Initialize query generator.

        Args:
            llm_interface: LLM interface for query generation
        """
        self.llm = llm_interface
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        prompts_dir = project_root / "prompts" / "online" / "retrieval"
        
        prompts = {}
        prompt_files = {
            "data_flow": "query_data_flow.txt",
            "control_flow": "query_control_flow.txt",
            "api_semantic": "query_api_semantic.txt",
            "refine": "refine_query.txt",
        }
        
        for key, filename in prompt_files.items():
            filepath = prompts_dir / filename
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    prompts[key] = f.read()
                logger.debug(f"加载检索 prompt: {filename}")
            except FileNotFoundError:
                logger.warning(f"Prompt 文件未找到: {filepath}，将使用默认 prompt")
                # Fallback to default prompts
                if key == "data_flow":
                    prompts[key] = "Based on the context:\n{context}\n\nGenerate a query for data flow issues:"
                elif key == "control_flow":
                    prompts[key] = "Based on the context:\n{context}\n\nGenerate a query for control flow issues:"
                elif key == "api_semantic":
                    prompts[key] = "Based on the context:\n{context}\n\nGenerate a query for API/semantic issues:"
                elif key == "refine":
                    prompts[key] = "Refine query:\n{original_query}\n\nFeedback:\n{feedback}\n\nRefined query:"
        
        return prompts

    def generate_queries(self, enhanced_context: str) -> Dict[str, str]:
        """
        Generate queries for all view types.

        Args:
            enhanced_context: Enhanced context from diagnostic phase

        Returns:
            Dictionary mapping view types to queries
        """
        logger.info("Generating multi-view queries")
        
        queries = {}
        for view_type in self.VIEW_TYPES:
            query = self.generate_single_query(enhanced_context, view_type)
            queries[view_type] = query
            logger.debug(f"{view_type} query: {query[:100]}...")

        return queries

    def generate_single_query(self, enhanced_context: str, view_type: str) -> str:
        """
        Generate query for a single view type.
        Parses JSON output and extracts the search_query field.

        Args:
            enhanced_context: Enhanced context
            view_type: View type

        Returns:
            Generated query string (extracted from JSON if applicable)
        """
        import json
        import re
        
        if view_type not in self.prompts:
            raise ValueError(f"Unknown view type: {view_type}")

        # IMPORTANT: Query generation must NOT include full buggy code.
        # enhanced_context (from Phase1) starts with "=== BUGGY CODE ===" + full source,
        # which can easily exceed model context limits (128k) and also increases proxy timeouts.
        condensed_context = self._condense_enhanced_context_for_query(enhanced_context)
        prompt = self.prompts[view_type].format(context=condensed_context)

        try:
            response = self.llm.generate(prompt, max_tokens=1024, temperature=0.3)
            response = response.strip()
            
            # 尝试解析 JSON 格式的输出
            try:
                # 移除可能的 markdown 代码块标记
                json_str = re.sub(r'^```json\s*', '', response)
                json_str = re.sub(r'```\s*$', '', json_str)
                
                query_data = json.loads(json_str)
                
                # 提取 search_query 字段
                if "search_query" in query_data:
                    search_query = query_data["search_query"].strip()
                    
                    # 记录 abstract_logic 用于调试
                    if "abstract_logic" in query_data:
                        logger.debug(f"{view_type} abstract_logic: {query_data['abstract_logic'][:100]}...")
                    
                    logger.debug(f"✓ 从 JSON 提取 search_query: {search_query[:100]}")
                    return search_query
                else:
                    logger.warning(f"JSON 输出缺少 'search_query' 字段: {query_data}")
                    return response[:500]  # 降级处理
            
            except json.JSONDecodeError as e:
                logger.warning(f"无法解析 JSON 输出（{view_type}）: {e}")
                logger.debug(f"原始响应: {response[:200]}...")
                
                # 降级：尝试提取类似 "search_query": "..." 的内容
                match = re.search(r'"search_query"\s*:\s*"([^"]+)"', response)
                if match:
                    logger.debug(f"✓ 使用正则提取 search_query")
                    return match.group(1).strip()
                
                # 完全降级：返回原始响应（截断）
                logger.warning(f"⚠️  使用原始响应作为查询（{view_type}）")
                return self._sanitize_query_text(response)
            
        except Exception as e:
            logger.error(f"Query generation failed for {view_type}: {e}")
            # Fallback: never return raw enhanced_context/buggy code as a query.
            return self._fallback_query(enhanced_context, view_type)

    def _condense_enhanced_context_for_query(self, enhanced_context: str, max_chars: int = 8000) -> str:
        """
        Reduce Phase1 enhanced_context to a compact context for query generation.
        We keep DIAGNOSTIC REPORT and STATIC ANALYSIS FACTS, drop full BUGGY CODE.
        """
        import re

        text = enhanced_context or ""

        def extract_between(start: str, end: str) -> str:
            if start not in text:
                return ""
            s = text.split(start, 1)[1]
            # If end delimiter is empty, treat it as "to the end of string".
            if not end:
                return s.strip()
            if end in s:
                s = s.split(end, 1)[0]
            return s.strip()

        diag = extract_between("=== DIAGNOSTIC REPORT ===", "=== STATIC ANALYSIS FACTS ===")
        facts = extract_between("=== STATIC ANALYSIS FACTS ===", "") if "=== STATIC ANALYSIS FACTS ===" in text else ""

        # Try to further extract just [FACTS] and [SEARCH_KEYWORDS] for stability.
        diag_compact = diag
        if diag:
            facts_m = re.search(r"\[FACTS\](.*?)(?=\[SEARCH_KEYWORDS\]|\Z)", diag, re.DOTALL | re.IGNORECASE)
            keywords_m = re.search(r"\[SEARCH_KEYWORDS\](.*)$", diag, re.DOTALL | re.IGNORECASE)
            parts = []
            if facts_m and facts_m.group(1).strip():
                parts.append("[FACTS]\n" + facts_m.group(1).strip())
            if keywords_m and keywords_m.group(1).strip():
                parts.append("[SEARCH_KEYWORDS]\n" + keywords_m.group(1).strip())
            if parts:
                diag_compact = "\n\n".join(parts)

        # Truncate each section to avoid blowing the prompt.
        def trunc(s: str, n: int) -> str:
            s = (s or "").strip()
            if len(s) <= n:
                return s
            return s[:n] + "\n...[TRUNCATED]..."

        diag_compact = trunc(diag_compact, 4000)
        facts = trunc(facts, 3500)

        out_parts = []
        if diag_compact:
            out_parts.append("=== DIAGNOSTIC REPORT (COMPACT) ===\n" + diag_compact)
        if facts:
            out_parts.append("=== STATIC ANALYSIS FACTS (TRUNCATED) ===\n" + facts)

        condensed = "\n\n".join(out_parts).strip()
        if not condensed:
            # As a last resort, keep only a tiny prefix (never include full code).
            condensed = trunc(re.sub(r"\s+", " ", text), 1200)

        if len(condensed) > max_chars:
            condensed = condensed[:max_chars] + "\n...[TRUNCATED]..."

        return condensed

    def _sanitize_query_text(self, s: str, max_len: int = 280) -> str:
        """Keep the query short and remove obvious HTML/code blocks."""
        import re
        s = (s or "").strip()
        # If HTML error page, do not pass through.
        if s.lower().startswith("<!doctype html") or "<html" in s.lower():
            return ""
        s = re.sub(r"\s+", " ", s)
        if len(s) > max_len:
            s = s[:max_len]
        return s.strip()

    def _fallback_query(self, enhanced_context: str, view_type: str) -> str:
        """
        Build a small heuristic query when LLM query generation fails.
        MUST NOT include raw buggy code.
        """
        import re
        text = enhanced_context or ""

        # Prefer SEARCH_KEYWORDS from diagnostic report
        m = re.search(r"\[SEARCH_KEYWORDS\](.*?)(?:\n===|\Z)", text, re.DOTALL | re.IGNORECASE)
        keywords = (m.group(1).strip() if m else "")
        keywords = self._sanitize_query_text(keywords, max_len=220)

        # If no keywords, fallback to FACTS
        if not keywords:
            m2 = re.search(r"\[FACTS\](.*?)(?:\n===|\Z)", text, re.DOTALL | re.IGNORECASE)
            facts = (m2.group(1).strip() if m2 else "")
            keywords = self._sanitize_query_text(facts, max_len=220)

        # View hint
        view_hint = {
            "data_flow": "taint data flow validation sanitization",
            "control_flow": "control flow guard check bypass validation",
            "api_semantic": "api misuse unsafe call semantic vulnerability",
        }.get(view_type, "security vulnerability fix")

        if keywords:
            q = f"{view_hint} {keywords}"
        else:
            q = view_hint

        q = self._sanitize_query_text(q, max_len=280) or view_hint
        logger.warning(f"⚠️  Fallback query for {view_type}: {q}")
        return q

    def refine_query(self, original_query: str, view_type: str, feedback: str) -> str:
        """
        Refine query based on feedback.

        Args:
            original_query: Original query
            view_type: View type
            feedback: Feedback for refinement

        Returns:
            Refined query
        """
        # Use refine prompt template
        prompt_template = self.prompts.get("refine", "")
        if not prompt_template:
            logger.warning("使用默认 refine prompt（模板未加载）")
            prompt_template = "Refine query:\n{original_query}\n\nFeedback:\n{feedback}\n\nRefined query:"
        
        prompt = prompt_template.format(
            view_type=view_type,
            original_query=original_query,
            feedback=feedback
        )

        try:
            refined = self.llm.generate(prompt, max_tokens=256, temperature=0.2)
            return refined.strip()
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return original_query

    def combine_queries(self, queries: Dict[str, str]) -> str:
        """
        Combine multiple view queries into unified query.

        Args:
            queries: Dictionary of view-specific queries

        Returns:
            Combined query
        """
        combined_parts = []
        for view_type, query in queries.items():
            combined_parts.append(f"[{view_type.upper()}] {query}")

        return " ".join(combined_parts)
