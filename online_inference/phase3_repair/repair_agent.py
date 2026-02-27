"""Iterative repair agent for patch generation."""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class RepairAgent:
    """LLM-based iterative repair agent for generating and refining patches."""

    def __init__(
        self,
        llm_interface,
        validator,
        max_iterations: int = 5,
    ):
        """
        Initialize repair agent.

        Args:
            llm_interface: LLM interface for code generation
            validator: Validator for compilation and testing
            max_iterations: Maximum repair iterations
        """
        self.llm = llm_interface
        self.validator = validator
        self.max_iterations = max_iterations
        self.repair_history: List[Dict] = []
        
        # Load prompt templates
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        prompts_dir = project_root / "prompts" / "online" / "repair"
        
        prompts = {}
        prompt_files = {
            "initial_repair": "initial_repair.txt",
            "refine_repair": "refine_repair.txt",
            "exemplar_template": "exemplar_template.txt",
        }
        
        for key, filename in prompt_files.items():
            filepath = prompts_dir / filename
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    prompts[key] = f.read()
                logger.debug(f"加载 prompt: {filename}")
            except FileNotFoundError:
                logger.warning(f"Prompt 文件未找到: {filepath}，将使用默认 prompt")
                prompts[key] = ""
        
        return prompts

    def repair(
        self,
        buggy_code: str,
        exemplars: List[Dict],
        diagnostic_report: Optional[Dict] = None,
        project_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Iteratively generate and validate patch.

        Args:
            buggy_code: Buggy source code
            exemplars: Top-k exemplars from retrieval phase
            diagnostic_report: Diagnostic report from Phase 1 (dict with facts/keywords)
            project_info: Optional project information (test commands, etc.)

        Returns:
            Repair result with final patch or failure info
        """
        logger.info("Starting iterative repair")
        
        self.repair_history = []
        final_result = {
            "success": False,
            "buggy_code": buggy_code,
            "final_patch": None,
            "iterations": 0,
            "history": self.repair_history,
        }

        # Generate initial patch (传递诊断报告)
        draft_patch = self._generate_initial_patch(buggy_code, exemplars, diagnostic_report)
        
        # 🔧 优化：滑动窗口记忆 - 只保留最近一轮的尝试和错误
        previous_patch = None
        previous_error = None

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"Iteration {iteration}/{self.max_iterations}")

            # Record attempt
            attempt = {
                "iteration": iteration,
                "patch": draft_patch,
                "validation_result": None,
            }

            # 🔧 前置检查：检测修复代码是否遗漏了原始函数定义
            missing_funcs = self._find_missing_functions(buggy_code, draft_patch)
            
            # 如果缺失函数，直接失败并进入 refine（避免浪费 Docker 时间）
            if missing_funcs:
                validation_result = {
                    "success": False,
                    "stage": "completeness_check",
                    "error_log": (
                        "Generated code is incomplete: it is missing functions that exist in the original file.\n"
                        f"Missing functions: {missing_funcs}\n"
                        "You MUST output the COMPLETE file, including ALL original functions/methods/imports.\n"
                        "Do not truncate. Do not delete unrelated functions."
                    ),
                }
            else:
                # Validate patch
                validation_result = self.validator.validate(buggy_code, draft_patch, project_info)
            attempt["validation_result"] = validation_result

            # 将缺失函数信息追加到 error_log（若 Docker 验证失败）
            if not validation_result["success"] and missing_funcs:
                existing_log = validation_result.get("error_log", "")
                missing_hint = (
                    f"\n\n[FRAMEWORK HINT] The following functions are DEFINED in the original "
                    f"code but MISSING from your generated code (you deleted them): "
                    f"{missing_funcs}. "
                    f"You MUST include ALL original functions in the output."
                )
                validation_result["error_log"] = existing_log + missing_hint

            # Check if successful
            if validation_result["success"]:
                logger.info(f"Repair successful at iteration {iteration}")
                final_result["success"] = True
                final_result["final_patch"] = draft_patch
                final_result["iterations"] = iteration
                self.repair_history.append(attempt)
                break

            # Log failure
            error_log = validation_result.get("error_log", "Unknown error")
            logger.debug(f"Validation failed: {error_log[:200]}")

            # 🔧 优化：更新滑动窗口（只保留当前轮）
            previous_patch = draft_patch
            previous_error = error_log

            # Generate next patch based on feedback (只传递最近一轮)
            draft_patch = self._refine_patch(
                buggy_code, previous_patch, previous_error, iteration
            )

            self.repair_history.append(attempt)

        if not final_result["success"]:
            logger.warning(f"Repair failed after {self.max_iterations} iterations")
            final_result["iterations"] = self.max_iterations

        return final_result

    def _generate_initial_patch(
        self, 
        buggy_code: str, 
        exemplars: List[Dict],
        diagnostic_report: Optional[Dict] = None
    ) -> str:
        """
        Generate initial patch using exemplars and diagnostic report.

        Args:
            buggy_code: Buggy code
            exemplars: Retrieved exemplars
            diagnostic_report: Diagnostic report from Phase 1

        Returns:
            Generated patch
        """
        logger.info(f"生成初始补丁（提供 {len(exemplars)} 个示例）")

        # Build prompt with exemplars and diagnostic report
        prompt = self._build_initial_prompt(buggy_code, exemplars, diagnostic_report)
        logger.debug(f"Prompt 长度: {len(prompt)} 字符")

        try:
            # 动态设置 max_tokens：根据代码长度调整
            # 估算：完整修复代码 ≈ buggy_code 长度 * 1.2
            estimated_output_tokens = (len(buggy_code) // 4) + 1000
            max_tokens = min(max(estimated_output_tokens, 2048), 16384)  # 2K-16K 之间
            
            logger.debug(f"设置 max_tokens={max_tokens}（基于代码长度 {len(buggy_code)} 字符）")
            
            raw_response = self.llm.generate(prompt, max_tokens=max_tokens, temperature=0.2)
            logger.info(f"✓ LLM 响应长度: {len(raw_response)} 字符")
            logger.debug(f"LLM raw response preview:\n{raw_response[:500]}\n{'...' if len(raw_response) > 500 else ''}")
            
            # 提取推理过程（可选，用于调试）
            thought = self._extract_thought(raw_response)
            if thought:
                logger.info(f"💡 LLM 推理过程: {thought[:200]}...")
            
            extracted_patch = self._extract_patch(raw_response)
            logger.debug(f"提取后的补丁长度: {len(extracted_patch)}")
            
            if not extracted_patch or len(extracted_patch.strip()) < 10:
                logger.warning(f"提取的补丁为空或过短（长度: {len(extracted_patch) if extracted_patch else 0}）")
                logger.info(f"原始响应前500字符:\n{raw_response[:500]}")
                # 如果提取失败，返回原始响应（让后续验证逻辑判断）
                if raw_response.strip():
                    logger.info("将使用原始响应作为补丁")
                    return raw_response.strip()
                else:
                    logger.error("✗ LLM 返回空响应，无法生成补丁")
                    return ""  # 返回空字符串，让 validator 报错
            
            # 检查输出是否可能被截断
            if self._is_likely_truncated(extracted_patch):
                logger.warning("⚠️  生成的代码可能被截断（末尾不完整）")
                logger.warning(f"   末尾100字符: ...{extracted_patch[-100:]}")
                logger.info(f"   建议增加 max_tokens（当前: {max_tokens}）")
            
            logger.info(f"✓ 成功提取补丁（长度: {len(extracted_patch)}）")
            return extracted_patch
        except Exception as e:
            logger.error(f"✗ 初始补丁生成失败: {e}")
            import traceback
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
            return ""

    def _build_initial_prompt(
        self, 
        buggy_code: str, 
        exemplars: List[Dict],
        diagnostic_report: Optional[Dict] = None
    ) -> str:
        """Build prompt for initial patch generation with diagnostic report."""
        # 固定使用示例（已有滑动窗口和输出剥离优化，无需动态跳过）
        max_exemplars = 2  # 最多 2 个示例
        max_snippet_len = 1000  
        max_fix_len = 800  
        
        # 🎯 动态 Top-K: 根据相似度分数动态调整示例数量
        if exemplars and len(exemplars) > 0:
            # 检查 Top-1 的相似度分数
            top1_score = self._get_similarity_score(exemplars[0])
            
            if top1_score >= 0.9:
                # Top-1 相似度极高，只用 1 个示例即可
                actual_exemplars = 1
                logger.info(f"🎯 动态 Top-K: Top-1 相似度极高 ({top1_score:.3f} ≥ 0.9)，仅使用 1 个示例")
            elif top1_score >= 0.75:
                # Top-1 相似度较高，用 1-2 个示例
                actual_exemplars = min(2, len(exemplars))
                logger.info(f"🎯 动态 Top-K: Top-1 相似度较高 ({top1_score:.3f} ≥ 0.75)，使用 {actual_exemplars} 个示例")
            else:
                # Top-1 相似度一般，使用更多示例提供灵感
                actual_exemplars = min(max_exemplars, len(exemplars))
                logger.info(f"🎯 动态 Top-K: Top-1 相似度一般 ({top1_score:.3f} < 0.75)，使用 {actual_exemplars} 个示例提供更多灵感")
        else:
            actual_exemplars = 0
        
        # Build exemplars section
        exemplars_section = ""
        if exemplars and len(exemplars) > 0 and actual_exemplars > 0:
            example_parts = []
            
            for i, exemplar in enumerate(exemplars[:actual_exemplars], 1):
                buggy_code_snippet = exemplar.get("buggy_code", "")
                if not buggy_code_snippet:
                    logger.warning(f"Exemplar {i} has no buggy_code, skipping")
                    continue
                    
                example_parts.append(f"Example {i}:")
                example_parts.append("Buggy Code:")
                example_parts.append(buggy_code_snippet[:max_snippet_len])
                
                # Show patch or RCA for context
                patch_or_rca = exemplar.get("patch", "") or exemplar.get("rca_distilled", "")
                if patch_or_rca:
                    example_parts.append("Fix Info:")
                    example_parts.append(patch_or_rca[:max_fix_len])
                example_parts.append("")
            
            if example_parts:
                # Use exemplar template if available
                exemplar_template = self.prompts.get("exemplar_template", "Here are some similar examples of bug fixes:\n\n{examples}")
                exemplars_section = exemplar_template.format(examples="\n".join(example_parts))
                actual_count = len(example_parts) // 4  # 每个示例有 4 个部分
                logger.info(f"✅ 实际使用 {actual_count} 个示例（动态调整后）")
            else:
                logger.warning("没有有效的示例可用，将仅基于诊断报告生成修复")
        else:
            logger.warning("没有可用的示例，将仅基于漏洞代码生成修复")
        
        # 准备 diagnostic_report 文本
        if diagnostic_report:
            if isinstance(diagnostic_report, dict):
                # 提取 facts 和 keywords
                facts = diagnostic_report.get("facts", "") or diagnostic_report.get("raw_text", "")
                keywords = diagnostic_report.get("keywords", "")
                if facts or keywords:
                    report_parts = []
                    if facts:
                        report_parts.append(f"[FACTS]\n{facts}")
                    if keywords:
                        report_parts.append(f"[SEARCH_KEYWORDS]\n{keywords}")
                    report_text = "\n\n".join(report_parts)
                else:
                    report_text = str(diagnostic_report)
            else:
                report_text = str(diagnostic_report)
        else:
            report_text = "No diagnostic report available."
            logger.warning("未提供诊断报告，修复质量可能下降")
        
        # Use initial repair prompt template
        prompt_template = self.prompts.get("initial_repair", "")
        if not prompt_template:
            # Fallback to default if template not loaded
            logger.warning("使用默认 prompt（模板未加载）")
            prompt_template = "You are an expert code repair assistant.\n\nDiagnostic Report:\n{diagnostic_report}\n\n{exemplars_section}\n\nBuggy Code:\n{buggy_code}\n\nOutput the complete fixed code:"
        
        final_prompt = prompt_template.format(
            diagnostic_report=report_text,
            exemplars_section=exemplars_section,
            buggy_code=buggy_code
        )
        
        # 检查 prompt 长度（粗略估算：1 token ≈ 4 字符）
        estimated_tokens = len(final_prompt) // 4
        logger.debug(f"Prompt 估算长度: ~{estimated_tokens} tokens ({len(final_prompt)} 字符)")
        
        # 根据常见模型限制发出警告
        if estimated_tokens > 100000:  # 超过 100K tokens
            logger.error(f"❌ Prompt 过长: ~{estimated_tokens} tokens，远超大多数模型限制 (128K)")
            logger.error(f"   必须减少内容！当前 buggy_code 长度: {len(buggy_code)} 字符")
            raise ValueError(f"Prompt too long: {estimated_tokens} tokens exceeds model limit")
        elif estimated_tokens > 30000:  # 超过 30K tokens
            logger.warning(f"⚠️  Prompt 较长: ~{estimated_tokens} tokens（接近 128K 限制的 1/4）")
            logger.info("   如果失败，建议: 1) 进一步减少 top_k  2) 截断 buggy_code")
        
        return final_prompt

    def _refine_patch(
        self,
        buggy_code: str,
        previous_patch: str,
        error_log: str,
        iteration: int,
    ) -> str:
        """
        Refine patch based on error feedback.
        采用滑动窗口记忆：只使用最近一轮的 patch 和 error。

        Args:
            buggy_code: Buggy code (Immutable Context)
            previous_patch: Previous patch attempt (Short-term Memory)
            error_log: Error log from validation (Short-term Memory)
            iteration: Current iteration number

        Returns:
            Refined patch
        """
        logger.debug(f"Refining patch (iteration {iteration})")
        
        # 🔧 优化：输出剥离 - 从 previous_patch 中移除 <THOUGHT> 标签
        cleaned_previous_patch = self._strip_thought_tags(previous_patch)
        if len(cleaned_previous_patch) < len(previous_patch):
            saved_chars = len(previous_patch) - len(cleaned_previous_patch)
            logger.debug(f"✂️  Stripped THOUGHT tags, saved {saved_chars} chars (~{saved_chars//4} tokens)")

        # Use refine repair prompt template
        prompt_template = self.prompts.get("refine_repair", "")
        if not prompt_template:
            # Fallback to default if template not loaded
            logger.warning("使用默认 refine prompt（模板未加载）")
            prompt_template = """The previous repair attempt failed validation.

Buggy Code:
{buggy_code}

Previous Attempt:
{previous_patch}

Validation Error:
{error_log}

Output the complete fixed code below:"""
        
        # 🔧 优化：使用剥离后的完整 patch（不截断）
        prompt = prompt_template.format(
            buggy_code=buggy_code,
            previous_patch=cleaned_previous_patch,  # 使用剥离后的 patch
            error_log=error_log[:500]  # 适当增加错误日志长度
        )

        try:
            # 动态设置 max_tokens（与初始生成保持一致）
            estimated_output_tokens = (len(buggy_code) // 4) + 1000
            max_tokens = min(max(estimated_output_tokens, 2048), 16384)
            logger.debug(f"Refine 阶段设置 max_tokens={max_tokens}")
            
            refined_patch = self.llm.generate(prompt, max_tokens=max_tokens, temperature=0.3)
            return self._extract_patch(refined_patch)
        except Exception as e:
            logger.error(f"Patch refinement failed: {e}")
            return previous_patch  # Fallback to previous

    def _extract_patch(self, llm_response: str) -> str:
        """
        Extract patch from LLM response.
        Supports XML tags, markdown code blocks, and plain text.

        Args:
            llm_response: Raw LLM response

        Returns:
            Extracted patch (always free of markdown fences)
        """
        import re
        
        # Priority 1: Try to extract <FIXED_CODE> XML tag
        xml_match = re.search(
            r'<FIXED_CODE>(.*?)</FIXED_CODE>',
            llm_response,
            re.DOTALL | re.IGNORECASE
        )
        if xml_match:
            logger.debug("✓ 使用 XML 标签 <FIXED_CODE> 提取代码")
            return xml_match.group(1).strip()
        
        # Priority 2: Try to extract markdown code blocks
        # Pattern: ```[language]\n...code...\n```
        # Use re.DOTALL so '.' matches newlines inside the capture group.
        # The (?s:...) inline flag can also be used, but DOTALL on findall is cleaner.
        code_blocks = re.findall(
            r'```(?:[a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```',
            llm_response,
            re.DOTALL
        )
        if code_blocks:
            logger.debug("✓ 使用 markdown 代码块提取代码")
            return code_blocks[0].strip()
        
        # Priority 3: Fence at the start/end of the response (no language tag or no newline)
        stripped = llm_response.strip()
        if stripped.startswith('```'):
            # Remove leading fence + optional language tag + optional whitespace
            cleaned = re.sub(r'^```[a-zA-Z0-9_+-]*[ \t]*\n?', '', stripped)
            # Remove trailing fence
            cleaned = re.sub(r'\n?```[ \t]*$', '', cleaned)
            if cleaned.strip():
                logger.debug("✓ 使用简单清理提取代码（起始 fence）")
                return cleaned.strip()
        
        # Priority 4: Return the whole response (cleaned)
        # Remove common prefixes like "Here is the fixed code:"
        response = stripped
        response = re.sub(
            r'^(Here is|Here\'s|Below is|The fixed code is)[:\s]*',
            '',
            response,
            flags=re.IGNORECASE,
        )
        
        logger.warning("⚠️  未找到标准格式标记，返回清理后的原始响应")
        return response.strip()
    
    def _extract_thought(self, llm_response: str) -> Optional[str]:
        """
        提取 LLM 的推理过程（用于调试和可解释性）.
        
        Args:
            llm_response: Raw LLM response
        
        Returns:
            Extracted thought or None
        """
        import re
        
        # 尝试提取 <THOUGHT> 或 <DEBUG_THOUGHT>
        thought_match = re.search(
            r'<(?:DEBUG_)?THOUGHT>(.*?)</(?:DEBUG_)?THOUGHT>',
            llm_response,
            re.DOTALL | re.IGNORECASE
        )
        
        if thought_match:
            return thought_match.group(1).strip()
        
        return None
    
    def _get_similarity_score(self, exemplar: Dict) -> float:
        """
        提取示例的相似度分数.
        
        优先级: rerank_score > rca_similarity > code_similarity > text_similarity > fusion_score
        
        Args:
            exemplar: 示例字典
        
        Returns:
            相似度分数 (0.0-1.0)，默认 0.0
        """
        # 尝试多个可能的分数字段
        score_fields = [
            "rerank_score",      # Phase 2 重排序分数（最准确）
            "rca_similarity",    # RCA 相似度
            "code_similarity",   # 代码结构相似度
            "text_similarity",   # 文本相似度
            "fusion_score",      # 多视角融合分数
            "score",             # 通用分数字段
        ]
        
        for field in score_fields:
            if field in exemplar:
                score = exemplar[field]
                if isinstance(score, (int, float)):
                    # 确保分数在 [0, 1] 范围内
                    return max(0.0, min(1.0, float(score)))
        
        # 如果没有找到任何分数字段，返回默认值
        logger.debug(f"Exemplar 没有相似度分数，使用默认值 0.0")
        return 0.0
    
    def _strip_thought_tags(self, text: str) -> str:
        """
        🔧 优化：输出剥离 - 移除 <THOUGHT> 和 <DEBUG_THOUGHT> 标签及其内容.
        
        用于在迭代修复时，从历史中剥离推理过程，只保留代码。
        这可以节省 30%-50% 的 Token。
        
        Args:
            text: 包含可能的 THOUGHT 标签的文本
        
        Returns:
            移除 THOUGHT 标签后的文本
        """
        import re
        
        # 移除 <THOUGHT>...</THOUGHT> 和 <DEBUG_THOUGHT>...</DEBUG_THOUGHT>
        cleaned = re.sub(
            r'<(?:DEBUG_)?THOUGHT>.*?</(?:DEBUG_)?THOUGHT>',
            '',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # 移除多余的空行
        cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
        
        return cleaned.strip()

    def _find_missing_functions(self, original_code: str, fixed_code: str) -> List[str]:
        """
        检测修复代码中缺失的函数/方法定义。
        比较原始代码和修复代码中的函数名，返回缺失函数名列表。
        
        Args:
            original_code: 原始（有漏洞）代码
            fixed_code: LLM 生成的修复代码
            
        Returns:
            缺失的函数名列表
        """
        import re
        
        # 提取函数定义名称（Python 和 JavaScript）
        def extract_func_names(code: str) -> set:
            names = set()
            js_reserved = {
                "if", "for", "while", "switch", "catch", "with", "return", "else",
                "function", "class", "const", "let", "var", "try", "do", "case",
                "break", "continue", "default", "new", "typeof", "void", "delete",
            }
            # Python: def func_name(
            for m in re.finditer(r'^\s*def\s+(\w+)\s*\(', code, re.MULTILINE):
                names.add(m.group(1))
            # JavaScript: function func_name( 或 func_name: function(
            for m in re.finditer(r'\bfunction\s+(\w+)\s*\(', code):
                names.add(m.group(1))
            # JavaScript: const/let/var func_name = function
            for m in re.finditer(r'\b(?:const|let|var)\s+(\w+)\s*=\s*function', code):
                names.add(m.group(1))
            # JavaScript object-literal method: key: function(...)
            # Avoid control-flow keywords false positives such as "if: function(...)"
            for m in re.finditer(r'^\s*([A-Za-z_$][\w$]*)\s*:\s*function\s*\(', code, re.MULTILINE):
                name = m.group(1)
                if name not in js_reserved:
                    names.add(name)
            return names
        
        original_funcs = extract_func_names(original_code)
        fixed_funcs = extract_func_names(fixed_code)
        
        missing = sorted(original_funcs - fixed_funcs)
        if missing:
            logger.warning(f"⚠️  修复代码缺失 {len(missing)} 个函数: {missing}")
        return missing

    def _is_likely_truncated(self, code: str) -> bool:
        """检测代码是否可能被截断"""
        if not code:
            return False
        
        # 检查末尾是否不完整
        endings = code.rstrip()[-100:].lower()
        
        # 常见的截断信号
        truncation_signals = [
            "def ",          # Python 函数定义开始但没有 body
            "function ",     # JavaScript 函数定义开始
            "class ",        # 类定义开始
            "if ",           # 控制流开始
            "for ",
            "while ",
            "{",             # 代码块开始但没有闭合
            "= function",    # JavaScript 赋值开始
        ]
        
        for signal in truncation_signals:
            if endings.endswith(signal) or endings.endswith(signal.rstrip()):
                return True
        
        # 检查括号/引号是否平衡
        try:
            brace_balance = code.count('{') - code.count('}')
            paren_balance = code.count('(') - code.count(')')
            bracket_balance = code.count('[') - code.count(']')
            
            if abs(brace_balance) > 3 or abs(paren_balance) > 3 or abs(bracket_balance) > 3:
                logger.debug(f"括号不平衡: {{ {brace_balance}, ( {paren_balance}, [ {bracket_balance}")
                return True
        except Exception:
            pass
        
        return False
    
    def get_repair_summary(self) -> str:
        """
        Get summary of repair attempts.

        Returns:
            Summary string
        """
        summary_parts = [
            "=== Repair Summary ===",
            f"Total iterations: {len(self.repair_history)}",
            "",
        ]

        for i, attempt in enumerate(self.repair_history, 1):
            summary_parts.append(f"Iteration {i}:")
            val_result = attempt.get("validation_result", {})
            if val_result.get("success"):
                summary_parts.append("  Status: SUCCESS")
            else:
                summary_parts.append("  Status: FAILED")
                error = val_result.get("error_log", "Unknown")
                summary_parts.append(f"  Error: {error[:100]}")
            summary_parts.append("")

        return "\n".join(summary_parts)

    def reset(self):
        """Reset repair history."""
        self.repair_history = []
