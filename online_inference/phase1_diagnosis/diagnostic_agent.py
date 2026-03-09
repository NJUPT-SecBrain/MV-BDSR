"""Diagnostic agent using ReAct pattern for bug analysis."""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import re


class DiagnosticAgent:
    """ReAct-based diagnostic agent for analyzing buggy code."""

    def __init__(self, llm_interface, tool_registry, max_iterations: int = 10):
        """
        Initialize diagnostic agent.

        Args:
            llm_interface: LLM interface for reasoning
            tool_registry: Registry of available analysis tools
            max_iterations: Maximum ReAct iterations
        """
        self.llm = llm_interface
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.conversation_history: List[Dict] = []
        
        # Load prompt templates
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        prompts_dir = project_root / "prompts" / "online" / "diagnosis"
        
        prompts = {}
        prompt_files = {
            "initial_analysis": "initial_analysis.txt",
        }
        
        for key, filename in prompt_files.items():
            filepath = prompts_dir / filename
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    prompts[key] = f.read()
                logger.debug(f"加载诊断 prompt: {filename}")
            except FileNotFoundError:
                logger.warning(f"Prompt 文件未找到: {filepath}，将使用默认 prompt")
                prompts[key] = ""
        
        return prompts

    def diagnose(self, buggy_code: str) -> Dict:
        """
        Perform diagnostic analysis on buggy code.

        Args:
            buggy_code: Source code with bug

        Returns:
            Enhanced context with static facts
        """
        logger.info("Starting diagnostic analysis")
        
        self.conversation_history = []
        static_facts = []

        # Initial prompt
        initial_prompt = self._build_initial_prompt(buggy_code)
        self.conversation_history.append({"role": "user", "content": initial_prompt})

        # ReAct loop
        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Get LLM response
            response = self.llm.chat(self.conversation_history)
            self.conversation_history.append({"role": "assistant", "content": response})

            # Parse response for actions
            action = self._parse_action(response)

            if action is None or action["type"] == "finish":
                # Agent decided to finish — check if report is present; if not, ask once more
                has_report = bool(
                    re.search(r'<DIAGNOSTIC_REPORT>', response, re.IGNORECASE)
                    or re.search(r'\[FACTS\]', response, re.IGNORECASE)
                    or re.search(r'\*{1,2}Facts\*{1,2}', response, re.IGNORECASE)
                )
                if not has_report:
                    logger.debug("Action: Finish detected but no structured report found — requesting report output")
                    follow_up = (
                        "Please now output the structured diagnostic report in the required format:\n\n"
                        "<DIAGNOSTIC_REPORT>\n"
                        "[FACTS]\n"
                        "- (list your key findings here)\n\n"
                        "[SEARCH_KEYWORDS]\n"
                        "(3-5 search keywords)\n"
                        "</DIAGNOSTIC_REPORT>"
                    )
                    self.conversation_history.append({"role": "user", "content": follow_up})
                    follow_up_response = self.llm.chat(self.conversation_history)
                    self.conversation_history.append({"role": "assistant", "content": follow_up_response})
                    logger.debug("Received follow-up report response")
                logger.info("Diagnostic complete")
                break

            # Execute action/tool
            observation = self._execute_action(action, buggy_code)
            static_facts.append({
                "tool": action["tool"],
                "result": observation
            })

            # Add observation to conversation
            # 🔧 优化：截断过长的观察结果（工具输出可能很长）
            max_observation_len = 2000  # 限制每个观察结果的长度
            if len(observation) > max_observation_len:
                observation_truncated = observation[:max_observation_len] + "\n... [截断，保留前2000字符] ..."
                logger.debug(f"✂️  Truncated observation: {len(observation)} -> {len(observation_truncated)} chars")
            else:
                observation_truncated = observation
            
            obs_message = f"Observation: {observation_truncated}"
            self.conversation_history.append({"role": "user", "content": obs_message})
            
            # 🔧 优化：滑动窗口记忆 - 限制 conversation_history 长度
            # 只保留初始 prompt + 最近 N 轮对话，防止 context length exceeded
            max_history_turns = 3  # 保留最近 3 轮（每轮 2 条消息：assistant + user observation）
            max_messages = 1 + (max_history_turns * 2)  # 初始 prompt + N*2 条消息
            
            if len(self.conversation_history) > max_messages:
                # 保留初始 prompt + 最近的消息
                initial_prompt = self.conversation_history[0]
                recent_messages = self.conversation_history[-max_history_turns * 2:]
                
                old_len = len(self.conversation_history)
                self.conversation_history = [initial_prompt] + recent_messages
                
                logger.debug(f"🔄 Sliding window: trimmed history from {old_len} to {len(self.conversation_history)} messages")

        # 提取最后一次 assistant 响应
        last_response = ""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                last_response = msg["content"]
                break
        
        # 解析结构化报告
        diagnostic_report = self._extract_diagnostic_report(last_response)
        
        # 保持向后兼容：analysis 字段
        final_analysis = diagnostic_report.get("facts", "") or diagnostic_report.get("raw_text", "")

        return {
            "buggy_code": buggy_code,
            "diagnostic_report": diagnostic_report,  # 新增：结构化报告
            "analysis": final_analysis,  # 保持向后兼容
            "keywords": diagnostic_report.get("keywords", ""),  # 新增：搜索关键词
            "static_facts": static_facts,
            "enhanced_context": self._build_enhanced_context_v2(buggy_code, diagnostic_report, static_facts),
        }

    def _build_initial_prompt(self, buggy_code: str) -> str:
        """Build initial ReAct prompt."""
        tool_descriptions = self.tools.get_tool_descriptions()

        # Use prompt template
        prompt_template = self.prompts.get("initial_analysis", "")
        if not prompt_template:
            # Fallback to default if template not loaded
            logger.warning("使用默认诊断 prompt（模板未加载）")
            prompt_template = """You are a code analysis expert. Analyze the following buggy code to identify potential issues.

Available Tools:
{tool_descriptions}

Use the ReAct format:
Thought: [your reasoning about what to investigate]
Action: [tool_name] [parameters]
[Wait for Observation]
... (repeat Thought/Action/Observation as needed)
Thought: [final analysis]
Action: Finish

Buggy Code:
```
{buggy_code}
```

Begin your analysis:"""
        
        prompt = prompt_template.format(
            tool_descriptions=tool_descriptions,
            buggy_code=buggy_code
        )

        return prompt

    def _parse_action(self, response: str) -> Optional[Dict]:
        """
        Parse action from LLM response.

        Args:
            response: LLM response text

        Returns:
            Action dictionary or None
        """
        # Look for "Action: tool_name parameters" pattern
        action_match = re.search(r'Action:\s*(\w+)(?:\s+(.*))?', response, re.IGNORECASE)

        if not action_match:
            return None

        tool_name = action_match.group(1).lower()
        parameters = action_match.group(2) if action_match.group(2) else ""

        if tool_name == "finish":
            return {"type": "finish"}

        return {
            "type": "tool_call",
            "tool": tool_name,
            "parameters": parameters.strip(),
        }

    def _execute_action(self, action: Dict, buggy_code: str) -> str:
        """
        Execute tool action.

        Args:
            action: Action dictionary
            buggy_code: Code to analyze

        Returns:
            Observation string
        """
        tool_name = action["tool"]
        parameters = action.get("parameters", "")

        try:
            result = self.tools.execute(tool_name, buggy_code, parameters)
            return str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error: {str(e)}"

    def _extract_diagnostic_report(self, response: str) -> Dict[str, str]:
        """
        从 LLM 响应中提取结构化的诊断报告.
        支持多种格式（兼容 GPT/Gemini/Claude 的不同输出风格）：
          1. <DIAGNOSTIC_REPORT>...</DIAGNOSTIC_REPORT> XML 标签（首选）
          2. [FACTS] / [SEARCH_KEYWORDS] 段落（无 XML 外层）
          3. **Facts** / **Search Keywords** Markdown 标题
          4. Thought: 段落（最低级降级）

        Args:
            response: LLM response text

        Returns:
            Dictionary with 'raw_text', 'facts', 'keywords'
        """
        # ── 方式 1：标准 XML 标签 ───────────────────────────────────────────
        report_match = re.search(
            r'<DIAGNOSTIC_REPORT>(.*?)</DIAGNOSTIC_REPORT>',
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if report_match:
            report_content = report_match.group(1)
            facts = self._extract_section(report_content, r'\[FACTS\]', r'\[SEARCH_KEYWORDS\]')
            keywords = self._extract_section(report_content, r'\[SEARCH_KEYWORDS\]', None)
            logger.debug(f"✓ [XML] 提取诊断报告: facts={len(facts)}字符, keywords={len(keywords)}字符")
            return {"raw_text": report_content, "facts": facts, "keywords": keywords}

        # ── 方式 2：[FACTS] / [SEARCH_KEYWORDS] 直接出现在响应中（无 XML）──
        facts_direct = self._extract_section(response, r'\[FACTS\]', r'\[SEARCH_KEYWORDS\]')
        keywords_direct = self._extract_section(response, r'\[SEARCH_KEYWORDS\]', None)
        if facts_direct or keywords_direct:
            raw_text = (facts_direct + "\n" + keywords_direct).strip()
            logger.debug(f"✓ [SECTION] 提取诊断报告: facts={len(facts_direct)}字符, keywords={len(keywords_direct)}字符")
            return {"raw_text": raw_text, "facts": facts_direct, "keywords": keywords_direct}

        # ── 方式 3：Markdown 标题 **Facts** / **Keywords** ──────────────────
        md_facts = self._extract_section(
            response,
            r'\*{1,2}Facts\*{1,2}:?',
            r'\*{1,2}(?:Search\s+)?Keywords?\*{1,2}:?',
        )
        md_keywords = self._extract_section(
            response,
            r'\*{1,2}(?:Search\s+)?Keywords?\*{1,2}:?',
            None,
        )
        if md_facts or md_keywords:
            raw_text = (md_facts + "\n" + md_keywords).strip()
            logger.debug(f"✓ [MARKDOWN] 提取诊断报告: facts={len(md_facts)}字符, keywords={len(md_keywords)}字符")
            return {"raw_text": raw_text, "facts": md_facts, "keywords": md_keywords}

        # ── 方式 4：最终降级——取 Thought: 段落或全文 ────────────────────────
        logger.warning("未找到结构化诊断报告标签（<DIAGNOSTIC_REPORT>/[FACTS]/Markdown），使用原始响应")
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL | re.IGNORECASE)
        raw_text = thought_match.group(1).strip() if thought_match else response
        return {"raw_text": raw_text, "facts": "", "keywords": ""}

    def _extract_section(self, text: str, start_pattern: str, end_pattern: Optional[str]) -> str:
        """在 text 中提取 start_pattern 到 end_pattern 之间的内容（不含标题行本身）。"""
        if end_pattern:
            m = re.search(
                rf'{start_pattern}\s*(.*?)(?={end_pattern}|$)',
                text,
                re.DOTALL | re.IGNORECASE,
            )
        else:
            m = re.search(
                rf'{start_pattern}\s*(.*?)$',
                text,
                re.DOTALL | re.IGNORECASE,
            )
        return m.group(1).strip() if m else ""
    
    def _extract_final_analysis(self) -> str:
        """Extract final analysis from conversation history (legacy method)."""
        # Get last assistant message
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                # Extract text after "Thought:" if present
                content = msg["content"]
                thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', content, re.DOTALL | re.IGNORECASE)
                if thought_match:
                    return thought_match.group(1).strip()
                return content

        return "No analysis available"

    def _build_enhanced_context_v2(
        self, 
        buggy_code: str, 
        diagnostic_report: Dict, 
        static_facts: List[Dict]
    ) -> str:
        """
        Build enhanced context using structured diagnostic report.

        Args:
            buggy_code: Original code
            diagnostic_report: Structured diagnostic report
            static_facts: List of static analysis results

        Returns:
            Enhanced context string
        """
        context_parts = [
            "=== BUGGY CODE ===",
            buggy_code,
            "",
            "=== DIAGNOSTIC REPORT ===",
        ]
        
        # 添加结构化的诊断报告
        facts = diagnostic_report.get("facts", "")
        keywords = diagnostic_report.get("keywords", "")
        
        if facts or keywords:
            if facts:
                context_parts.append("[FACTS]")
                context_parts.append(facts)
                context_parts.append("")
            if keywords:
                context_parts.append("[SEARCH_KEYWORDS]")
                context_parts.append(keywords)
                context_parts.append("")
        else:
            # 降级：使用 raw_text
            raw_text = diagnostic_report.get("raw_text", "No analysis available")
            context_parts.append(raw_text)
            context_parts.append("")
        
        context_parts.append("=== STATIC ANALYSIS FACTS ===")

        for fact in static_facts:
            context_parts.append(f"[{fact['tool']}]")
            context_parts.append(str(fact['result']))
            context_parts.append("")

        return "\n".join(context_parts)
    
    def _build_enhanced_context(
        self, buggy_code: str, analysis: str, static_facts: List[Dict]
    ) -> str:
        """
        Build enhanced context combining code, analysis, and facts (legacy method).

        Args:
            buggy_code: Original code
            analysis: Agent analysis
            static_facts: List of static analysis results

        Returns:
            Enhanced context string
        """
        context_parts = [
            "=== BUGGY CODE ===",
            buggy_code,
            "",
            "=== DIAGNOSTIC ANALYSIS ===",
            analysis,
            "",
            "=== STATIC FACTS ===",
        ]

        for fact in static_facts:
            context_parts.append(f"[{fact['tool']}]")
            context_parts.append(str(fact['result']))
            context_parts.append("")

        return "\n".join(context_parts)

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
