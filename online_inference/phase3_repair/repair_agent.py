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
        single_attempt: bool = False,
    ):
        """
        Initialize repair agent.

        Args:
            llm_interface: LLM interface for code generation
            validator: Validator for compilation and testing
            max_iterations: Maximum repair iterations
            single_attempt: If True, generate exactly one patch and validate it
                without any refinement iterations (ablation: no iterative repair).
        """
        self.llm = llm_interface
        self.validator = validator
        self.max_iterations = max_iterations
        self.single_attempt = single_attempt
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
        if self.single_attempt:
            logger.info("Starting single-attempt repair (ablation: no iterative refinement)")
        else:
            logger.info("Starting iterative repair")
        
        self.repair_history = []
        final_result = {
            "success": False,
            "buggy_code": buggy_code,
            "final_patch": None,
            "iterations": 0,
            "history": self.repair_history,
        }

        # Generate initial patch (传递诊断报告 + project_info)
        draft_patch = self._generate_initial_patch(buggy_code, exemplars, diagnostic_report, project_info)
        
        # 🔧 优化：滑动窗口记忆 - 只保留最近一轮的尝试和错误
        previous_patch = None
        previous_error = None

        # 🔧 最优完整补丁跟踪：记录已通过 completeness_check 的最近一个补丁
        # 防止 Gemini 在某轮生成完整代码后下一轮又退回到不完整代码
        best_complete_patch: Optional[str] = None

        # 消融模式：single_attempt=True 时最多只执行 1 轮（不进行 refine）
        effective_max = 1 if self.single_attempt else self.max_iterations

        for iteration in range(1, effective_max + 1):
            logger.info(f"Iteration {iteration}/{effective_max}")

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
                # 通过 completeness_check：记录为最优完整补丁
                best_complete_patch = draft_patch
                logger.debug(f"✓ 通过 completeness_check，更新 best_complete_patch（{len(draft_patch)} 字符）")
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

            # 🔧 防振荡：如果当前 draft_patch 不完整（completeness_check 失败），
            # 但之前有过完整的补丁，则以完整补丁为基础进行 refine，而非从不完整版本继续
            if missing_funcs and best_complete_patch is not None:
                refine_base = best_complete_patch
                logger.info(
                    f"⚠️  当前轮代码不完整（缺失 {len(missing_funcs)} 个函数），"
                    f"切换到 best_complete_patch（{len(best_complete_patch)} 字符）作为 refine 基础"
                )
            else:
                refine_base = draft_patch

            # 🔧 优化：更新滑动窗口（只保留当前轮）
            previous_patch = refine_base
            previous_error = error_log

            # Generate next patch based on feedback (只传递最近一轮)
            draft_patch = self._refine_patch(
                buggy_code, previous_patch, previous_error, iteration
            )

            self.repair_history.append(attempt)

        if not final_result["success"]:
            if self.single_attempt:
                logger.warning("Repair failed (single-attempt mode, no refinement performed)")
            else:
                logger.warning(f"Repair failed after {self.max_iterations} iterations")
            final_result["iterations"] = effective_max

            # ── 函数拼接后备（Stitch Fallback）────────────────────────────────────────
            # 如果最后失败原因是 completeness_check，尝试将缺失函数从原始代码中拼接进来
            last_attempt = self.repair_history[-1] if self.repair_history else {}
            last_vr = last_attempt.get("validation_result", {})
            last_stage = last_vr.get("stage", "")
            last_err_msg = last_vr.get("error_log", "") or last_vr.get("error_message", "")
            if last_stage == "completeness_check" or (
                "Missing functions" in last_err_msg and "completeness" in last_err_msg.lower()
            ):
                # 优先用 best_complete_patch（通过过 completeness_check），退而求其次用 draft_patch
                # 对于大文件（>20K字符），LLM 无法在网关超时内生成完整文件，draft_patch 包含
                # 已修复的函数，StitchFallback 负责从原始代码中补全其余函数
                stitch_base = best_complete_patch or draft_patch or ""
                if not stitch_base:
                    logger.warning("[StitchFallback] 无可用的 patch 基础，跳过拼接")
                elif best_complete_patch is None:
                    logger.info("[StitchFallback] 无 best_complete_patch，改用 draft_patch 作为拼接基础")
                else:
                    # 快速语法预检：确保 stitch_base 本身语法正确
                    lang_ext = (project_info or {}).get("buggy_code_file", "").rsplit(".", 1)[-1] if project_info else ""
                    syntax_ok = True
                    if lang_ext == "py":
                        try:
                            compile(stitch_base, "<stitch_base>", "exec")
                        except SyntaxError:
                            syntax_ok = False
                            logger.warning("[StitchFallback] best_complete_patch 存在 Python 语法错误，跳过拼接")
                    if syntax_ok:
                        logger.info("🔧 [StitchFallback] 尝试将缺失函数从原始代码拼接进补丁")
                        stitched = self._stitch_missing_functions(buggy_code, stitch_base)
                        if stitched and stitched != stitch_base:
                            stitch_missing = self._find_missing_functions(buggy_code, stitched)
                            if not stitch_missing:
                                # 对拼接结果也做语法预检（对 Python）
                                stitch_syntax_ok = True
                                if lang_ext == "py":
                                    try:
                                        compile(stitched, "<stitched>", "exec")
                                    except SyntaxError as se:
                                        stitch_syntax_ok = False
                                        logger.warning(f"[StitchFallback] 拼接结果语法错误: {se}，跳过验证")
                                if stitch_syntax_ok:
                                    logger.info("✅ [StitchFallback] 拼接成功，重新验证补丁")
                                    stitch_vr = self.validator.validate(
                                        buggy_code, stitched, project_info
                                    )
                                    self.repair_history.append({
                                        "iteration": effective_max + 1,
                                        "patch": stitched,
                                        "validation_result": stitch_vr,
                                        "strategy": "stitch_fallback",
                                    })
                                    if stitch_vr.get("success"):
                                        logger.info("🎉 [StitchFallback] 拼接后验证通过！")
                                        final_result = {
                                            "success": True,
                                            "patch": stitched,
                                            "iterations": effective_max + 1,
                                            "validation_result": stitch_vr,
                                        }
                            else:
                                logger.warning(f"[StitchFallback] 拼接后仍缺失 {len(stitch_missing)} 个函数，跳过")
                        else:
                            logger.warning("[StitchFallback] 拼接未产生新内容")

        return final_result

    def _generate_initial_patch(
        self, 
        buggy_code: str, 
        exemplars: List[Dict],
        diagnostic_report: Optional[Dict] = None,
        project_info: Optional[Dict] = None,
    ) -> str:
        """
        Generate initial patch using exemplars and diagnostic report.

        Args:
            buggy_code: Buggy code
            exemplars: Retrieved exemplars
            diagnostic_report: Diagnostic report from Phase 1
            project_info: Optional project information (used for language-aware filtering)

        Returns:
            Generated patch
        """
        logger.info(f"生成初始补丁（提供 {len(exemplars)} 个示例）")

        # Build prompt with exemplars and diagnostic report
        prompt = self._build_initial_prompt(buggy_code, exemplars, diagnostic_report, project_info)
        logger.debug(f"Prompt 长度: {len(prompt)} 字符")

        try:
            # 动态设置 max_tokens：根据代码长度调整
            # 估算：完整修复代码 ≈ buggy_code 长度 / 4 (chars→tokens) + overhead
            # 对超大文件（>20KB）提前使用更大的 max_tokens 避免初始生成就被截断
            buggy_chars = len(buggy_code)
            if buggy_chars > 40000:
                # 超大文件：网关超时阈值内最多安全生成 ~8192 tokens
                # LLM 只输出修复相关函数（部分），StitchFallback 负责补全其余原始函数
                max_tokens = 8192
                logger.info(f"🔧 超大文件（{buggy_chars}字符）→ 初始生成 max_tokens=8192（网关安全，StitchFallback 兜底）")
            elif buggy_chars > 15000:
                # 中大文件（15K-40K）：常见Python/JS文件，需要更大buffer
                estimated_output_tokens = (buggy_chars // 3) + 3000
                max_tokens = min(max(estimated_output_tokens, 8192), 32768)
                logger.info(f"🔧 中大文件（{buggy_chars}字符）→ 初始生成 max_tokens={max_tokens}")
            else:
                # 小文件（<15K）
                estimated_output_tokens = (buggy_chars // 3) + 1500
                max_tokens = min(max(estimated_output_tokens, 4096), 16384)
            
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

    @staticmethod
    def _detect_code_language(code: str, file_path: str = "") -> str:
        """
        从文件路径扩展名或代码内容启发式检测编程语言。

        返回值: 'go' | 'python' | 'javascript' | 'typescript' | 'c_cpp' | 'java' | 'unknown'
        """
        # 优先从文件扩展名判断（最可靠）
        if file_path:
            ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
            if ext == "go":
                return "go"
            if ext == "py":
                return "python"
            if ext in ("js", "mjs", "cjs"):
                return "javascript"
            if ext in ("ts", "tsx"):
                return "typescript"
            if ext in ("c", "h", "cc", "cpp", "cxx", "hpp"):
                return "c_cpp"
            if ext == "java":
                return "java"

        # 回退：关键词频率启发
        snippet = (code or "")[:3000]
        scores = {
            "go":         snippet.count("package ") + snippet.count("func ") + snippet.count(":= ") + snippet.count('import "'),
            "python":     snippet.count("def ") + snippet.count("import ") + snippet.count("class ") + snippet.count("    "),
            "javascript": snippet.count("function ") + snippet.count("require(") + snippet.count("const ") + snippet.count("=>"),
            "c_cpp":      snippet.count("#include") + snippet.count("void ") + snippet.count("int ") + snippet.count("static "),
            "java":       snippet.count("public ") + snippet.count("class ") + snippet.count("import ") + snippet.count("void "),
        }
        best = max(scores, key=scores.__getitem__)
        return best if scores[best] > 2 else "unknown"

    def _build_initial_prompt(
        self, 
        buggy_code: str, 
        exemplars: List[Dict],
        diagnostic_report: Optional[Dict] = None,
        project_info: Optional[Dict] = None,
    ) -> str:
        """Build prompt for initial patch generation with diagnostic report."""
        buggy_file_path = (project_info or {}).get("buggy_code_file", "")
        target_lang = self._detect_code_language(buggy_code, buggy_file_path)
        logger.info(f"🌐 目标语言检测: {target_lang}（文件: {buggy_file_path}）")

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
        import re as _re
        logger.debug(f"Refining patch (iteration {iteration})")
        
        # 🔧 优化：输出剥离 - 从 previous_patch 中移除 <THOUGHT> 标签
        cleaned_previous_patch = self._strip_thought_tags(previous_patch)
        if len(cleaned_previous_patch) < len(previous_patch):
            saved_chars = len(previous_patch) - len(cleaned_previous_patch)
            logger.debug(f"✂️  Stripped THOUGHT tags, saved {saved_chars} chars (~{saved_chars//4} tokens)")

        # ── 错误类型检测，驱动 max_tokens 和 error_log 截断策略 ──────────────────
        is_completeness_failure = (
            "Generated code is incomplete" in error_log
            or "Missing functions" in error_log
        )
        is_go_compile_error = _re.search(
            r"redeclared in this block|not enough arguments in call|too many arguments in call"
            r"|undefined: \w|mismatched types|build failed|invalid operation",
            error_log,
        ) is not None
        is_unused_import = _re.search(
            r'imported and not used|declared and not used',
            error_log,
        ) is not None
        is_truncated_output = _re.search(
            r"syntax error: unexpected EOF|unexpected end of file|unexpected EOF"
            r"|string literal not terminated|comment not terminated"
            r"|expected '\)', found 'EOF'|newline in string",
            error_log,
        ) is not None

        # completeness 失败时：对中等文件适当提升，对大文件（>20K字符）硬性限制到 8192
        # 原因：大文件生成 15K+ tokens 需要 5-6 分钟，会触发 Cloudflare 524 网关超时
        # 策略：宁可让 LLM 再次生成部分代码，由 StitchFallback 从原始代码中拼接缺失函数
        if is_completeness_failure:
            if len(buggy_code) > 20000:
                # 大文件：硬性限制 8192（约 2 分钟，在网关超时阈值内）
                max_tokens = 8192
                logger.info(f"🔧 completeness_check 失败（大文件 {len(buggy_code)}字符）→ max_tokens=8192（网关安全）")
            else:
                estimated_output_tokens = (len(buggy_code) // 3) + 3000
                max_tokens = min(max(estimated_output_tokens, 8192), 12288)
                logger.info(f"🔧 completeness_check 失败 → max_tokens={max_tokens}")
            # 对于 completeness 失败，error_log 中有用的信息是函数列表，保留更多
            error_log_clipped = error_log[:2000]
        elif is_truncated_output:
            # Output was truncated at the end → significantly boost max_tokens
            estimated_output_tokens = (len(buggy_code) // 3) + 4000
            max_tokens = min(max(estimated_output_tokens, 8192), 32768)
            logger.info(f"🔧 代码被截断（EOF/newline in string）→ 大幅提升 max_tokens={max_tokens}")
            error_log_clipped = error_log[:800]
        elif is_unused_import:
            estimated_output_tokens = (len(buggy_code) // 3) + 1500
            max_tokens = min(max(estimated_output_tokens, 4096), 16384)
            logger.info(f"🔧 Go unused_import/declared → max_tokens={max_tokens}")
            error_log_clipped = error_log[:1200]
        elif is_go_compile_error:
            estimated_output_tokens = (len(buggy_code) // 3) + 2000
            max_tokens = min(max(estimated_output_tokens, 4096), 24576)
            logger.info(f"🔧 Go 编译错误 → max_tokens={max_tokens}")
            # 编译错误：保留完整错误信息（含 have/want 类型行）
            error_log_clipped = error_log[:1500]
        else:
            # +800 buffer for potential <DEBUG_THOUGHT> overhead (we now suppress it,
            # but Gemini may still output a short preamble)
            estimated_output_tokens = (len(buggy_code) // 3) + 2000
            max_tokens = min(max(estimated_output_tokens, 4096), 16384)
            error_log_clipped = error_log[:1000]

        # ── 构建 size_hint：告知模型最低输出规模 ─────────────────────────────────
        buggy_lines = buggy_code.count('\n') + 1
        buggy_chars = len(buggy_code)
        if is_completeness_failure:
            # Extract missing function names from error_log for explicit reminder
            import re as _re2
            mf_match = _re2.search(r"Missing functions: \[([^\]]+)\]", error_log)
            missing_funcs_hint = ""
            if mf_match:
                missing_funcs_hint = (
                    f" You are SPECIFICALLY missing these functions: {mf_match.group(1)}. "
                    f"Copy them VERBATIM from the Original Buggy Code below — do NOT skip them."
                )
            # Build function skeleton as a checklist for the model
            func_signatures = _re2.findall(
                r'^[ \t]*(?:def |async def |function |async function |\w[\w\s*]+ \w+\s*\()',
                buggy_code, _re2.MULTILINE
            )
            skeleton_hint = ""
            if func_signatures:
                sig_list = "\n".join(f"  - {s.strip()}" for s in func_signatures[:40])
                skeleton_hint = (
                    f"\nThe original file contains these function definitions (CHECKLIST — all must appear in your output):\n"
                    f"{sig_list}"
                )
            size_hint = (
                f"The original Buggy Code is {buggy_lines} lines / {buggy_chars} characters. "
                f"Your output MUST be at least {buggy_lines} lines and {buggy_chars} characters long.{missing_funcs_hint} "
                f"DO NOT stop writing until every function from the original is included.{skeleton_hint}"
            )
        elif is_unused_import:
            # Extract unused import names for targeted hint
            import re as _re3
            unused_matches = _re3.findall(r'"([^"]+)" imported and not used', error_log)
            unused_decl = _re3.findall(r'(\w+) declared and not used', error_log)
            compile_hint = ""
            if unused_matches:
                pkgs = ", ".join(f'"{p}"' for p in unused_matches[:5])
                compile_hint = (
                    f" CRITICAL: The following imports are NOT used in your code and cause compile errors: {pkgs}. "
                    f"Remove these import lines entirely. Do NOT add import lines for packages you don't actually use."
                )
            elif unused_decl:
                vars_ = ", ".join(f'`{v}`' for v in unused_decl[:5])
                compile_hint = (
                    f" CRITICAL: The following variables are declared but never used: {unused_decl}. "
                    f"Either use them or remove the declaration."
                )
            size_hint = (
                f"The original Buggy Code is {buggy_lines} lines.{compile_hint}"
            )
        elif is_go_compile_error:
            # Extract specific compile error lines for targeted hints
            import re as _re3
            redecl_match = _re3.search(r'(\w+) redeclared in this block', error_log)
            sig_match = _re3.search(r'(assignment mismatch|too many arguments|not enough arguments|cannot use)', error_log)
            compile_hint = ""
            if redecl_match:
                sym = redecl_match.group(1)
                compile_hint = (
                    f" CRITICAL: The symbol '{sym}' is already defined in another file of this package. "
                    f"Do NOT redefine it — remove your duplicate definition entirely."
                )
            elif sig_match:
                compile_hint = (
                    f" CRITICAL: You changed a function's return type or parameter list. "
                    f"Restore the ORIGINAL function signatures exactly — only change the internal logic, "
                    f"not the function name, parameters, or return types."
                )
            size_hint = (
                f"The original Buggy Code is {buggy_lines} lines.{compile_hint} "
                f"Remember: helper functions/types NOT shown in the Buggy Code above are defined "
                f"in OTHER files of the same package — do NOT redefine them."
            )
        else:
            size_hint = f"The original Buggy Code is {buggy_lines} lines / {buggy_chars} characters."

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

Size Constraint:
{size_hint}

Output the complete fixed code below:"""
        
        # 🔧 优化：使用剥离后的完整 patch（不截断）
        prompt = prompt_template.format(
            buggy_code=buggy_code,
            previous_patch=cleaned_previous_patch,  # 使用剥离后的 patch
            error_log=error_log_clipped,
            size_hint=size_hint,
        )

        try:
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
        
        def _strip_inner_fences(code: str) -> str:
            """去除 <FIXED_CODE> 内部嵌套的 markdown 围栏（Claude 常见输出格式）"""
            code = re.sub(r'^```[a-zA-Z0-9_+-]*[ \t]*\n?', '', code.strip())
            code = re.sub(r'\n?```[ \t]*$', '', code)
            return code.strip()

        # Priority 1: Try to extract <FIXED_CODE> XML tag (with proper closing tag)
        xml_match = re.search(
            r'<FIXED_CODE>(.*?)</FIXED_CODE>',
            llm_response,
            re.DOTALL | re.IGNORECASE
        )
        if xml_match:
            logger.debug("✓ 使用 XML 标签 <FIXED_CODE> 提取代码")
            return _strip_inner_fences(xml_match.group(1))

        # Priority 1c: <FIXED_CODE> tag found but closing tag is MISSING (Claude truncation)
        # Move this BEFORE markdown code blocks to avoid picking up a short snippet
        partial_xml_early = re.search(
            r'<FIXED_CODE>(.*)',
            llm_response,
            re.DOTALL | re.IGNORECASE,
        )
        if partial_xml_early:
            candidate = partial_xml_early.group(1).strip()
            candidate = re.sub(r'</FIXED_CODE.*$', '', candidate, flags=re.IGNORECASE).strip()
            candidate = _strip_inner_fences(candidate)
            if len(candidate) >= 50:  # Must be meaningful code, not an empty block
                logger.debug("✓ 使用不完整 <FIXED_CODE> 标签提取代码（Priority 1c，提前于 markdown 块）")
                return candidate
        
        # Priority 1b: Has <THOUGHT>/<DEBUG_THOUGHT> but NO <FIXED_CODE> tag
        # Gemini often outputs: <DEBUG_THOUGHT>...</DEBUG_THOUGHT>\nraw code here
        # Also handles TRUNCATED thought block (no closing tag) — in that case there is NO code.
        # → strip the thought block (with or without closing tag) and treat the remainder as the patch
        # IMPORTANT: Only activate if THOUGHT tags were actually found and stripped.
        _orig_stripped = llm_response.strip()
        _has_thought_tag = bool(re.search(r'<(?:DEBUG_)?THOUGHT>', llm_response, re.IGNORECASE))
        if _has_thought_tag:
            thought_stripped = re.sub(
                r'<(?:DEBUG_)?THOUGHT>.*?</(?:DEBUG_)?THOUGHT>\s*',
                '',
                llm_response,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            # Handle truncated THOUGHT (no closing tag): remove from opening tag to end
            if not thought_stripped or thought_stripped == _orig_stripped:
                thought_stripped = re.sub(
                    r'<(?:DEBUG_)?THOUGHT>.*$',
                    '',
                    llm_response,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
            # Only proceed if actual content was stripped
            if thought_stripped and thought_stripped != _orig_stripped and len(thought_stripped) >= 10:
                # Re-run Priority 1 on the stripped version (maybe FIXED_CODE is after THOUGHT)
                xml_match2 = re.search(
                    r'<FIXED_CODE>(.*?)</FIXED_CODE>',
                    thought_stripped,
                    re.DOTALL | re.IGNORECASE,
                )
                if xml_match2:
                    logger.debug("✓ [1b-xml] THOUGHT 块后发现 <FIXED_CODE>，提取代码")
                    return xml_match2.group(1).strip()
                # Check if the remainder looks like actual code (starts with LINE 1 keyword)
                # Use MULTILINE but only check if the VERY FIRST non-empty line is a code keyword
                first_line = thought_stripped.lstrip().split('\n')[0].strip()
                code_start_re = re.compile(
                    r'^(package\s|import\s|from\s|class\s|def\s|async\s+def\s|function\s|const\s|let\s|var\s|//|#|/\*)',
                    re.IGNORECASE,
                )
                if code_start_re.match(first_line):
                    logger.debug("✓ [1b] 去除 THOUGHT 块后，首行识别为代码")
                    # Also strip any leading/trailing markdown fences
                    thought_stripped = re.sub(r'^```[a-zA-Z0-9_+-]*[ \t]*\n?', '', thought_stripped)
                    thought_stripped = re.sub(r'\n?```[ \t]*$', '', thought_stripped)
                    return thought_stripped.strip()

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
        
        # Priority 3b: (fallback, already handled by Priority 1c above)
        # Kept as a safety net with lower threshold in case Priority 1c was skipped
        partial_xml = re.search(
            r'<FIXED_CODE>(.*)',
            llm_response,
            re.DOTALL | re.IGNORECASE,
        )
        if partial_xml:
            candidate = partial_xml.group(1).strip()
            candidate = re.sub(r'</FIXED_CODE.*$', '', candidate, flags=re.IGNORECASE).strip()
            candidate = _strip_inner_fences(candidate)
            if len(candidate) >= 10:
                logger.debug("✓ 使用不完整 <FIXED_CODE> 标签提取代码（Priority 3b）")
                return candidate

        # Priority 4: Return the whole response (cleaned)
        # Remove common prose prefixes like "Here is the fixed code:"
        response = stripped
        response = re.sub(
            r'^(Here is|Here\'s|Below is|The fixed code is|以下是|修复后的代码)[:\s]*',
            '',
            response,
            flags=re.IGNORECASE,
        )
        # Remove trailing explanation after the last "}" or last code line
        # (Gemini sometimes appends explanation after the code)
        response = re.sub(
            r'\n{2,}(This (fix|patch|code)|The (fix|patch|change)|Note:|Explanation:).*$',
            '',
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        logger.warning("⚠️  未找到标准格式标记，返回清理后的原始响应")
        # 打印原始响应的前400字符和后200字符，帮助诊断 Gemini 输出格式
        raw_preview = llm_response[:400] + ("\n...[中间省略]...\n" + llm_response[-200:] if len(llm_response) > 600 else "")
        logger.warning(f"[原始响应样本]:\n{raw_preview}")
        # Safety net: strip any stray <FIXED_CODE> / </FIXED_CODE> tags from the result
        response = re.sub(r'^\s*</?FIXED_CODE[^>]*>\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\s*</?FIXED_CODE[^>]*>\s*$', '', response, flags=re.IGNORECASE)
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

    def _stitch_missing_functions(self, original_code: str, partial_patch: str) -> str:
        """
        将原始代码中缺失的函数体拼接到 partial_patch 末尾。
        适用于 completeness_check 失败但修复逻辑正确的场景。
        只提取顶级函数定义（Python def/async def，JS function/arrow）。

        Args:
            original_code: 原始（有漏洞）代码（用于提取缺失函数）
            partial_patch: LLM 生成的不完整补丁

        Returns:
            拼接后的补丁（在末尾添加缺失函数）
        """
        import re

        def _extract_func_names(code: str) -> set:
            names = set()
            for m in re.finditer(r'^\s*(?:async\s+)?def\s+(\w+)\s*\(', code, re.MULTILINE):
                names.add(m.group(1))
            for m in re.finditer(r'\bfunction\s+(\w+)\s*\(', code):
                names.add(m.group(1))
            for m in re.finditer(r'\b(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function', code):
                names.add(m.group(1))
            return names

        def _extract_python_func_body(code: str, func_name: str) -> str:
            """提取 Python 函数（含 decorator、body）直到下一个同级定义"""
            lines = code.split('\n')
            start = None
            for i, line in enumerate(lines):
                if re.match(rf'^\s*(?:async\s+)?def\s+{re.escape(func_name)}\s*\(', line):
                    # Look backwards for decorators
                    j = i - 1
                    while j >= 0 and re.match(r'^\s*@', lines[j]):
                        j -= 1
                    start = j + 1
                    break
            if start is None:
                return ""
            # Find the end: next top-level def/class or EOF
            base_indent = len(lines[start]) - len(lines[start].lstrip())
            end = len(lines)
            for i in range(start + 1, len(lines)):
                stripped = lines[i].lstrip()
                if stripped and not stripped.startswith('#'):
                    indent = len(lines[i]) - len(stripped)
                    if indent <= base_indent and re.match(r'(?:async\s+)?def\s+|class\s+', stripped):
                        end = i
                        break
            return '\n'.join(lines[start:end])

        def _extract_js_func_body(code: str, func_name: str) -> str:
            """提取 JS function 声明"""
            # Match: function funcName(...) { ... }
            pattern = rf'(?:^|\n)((?:\/\*.*?\*\/\s*|\/\/[^\n]*\n\s*)*function\s+{re.escape(func_name)}\s*\(.*?\)\s*\{{)'
            m = re.search(pattern, code, re.DOTALL)
            if not m:
                # Match: const funcName = function(...) { ... }
                pattern2 = rf'(?:^|\n)((?:const|let|var)\s+{re.escape(func_name)}\s*=\s*(?:async\s+)?function[^{{]*\{{)'
                m = re.search(pattern2, code, re.DOTALL)
            if not m:
                return ""
            # Find the matching closing brace
            brace_start = m.end() - 1  # position of '{'
            depth = 0
            in_str = None
            i = brace_start
            while i < len(code):
                c = code[i]
                if in_str:
                    if c == '\\':
                        i += 2
                        continue
                    if c == in_str:
                        in_str = None
                elif c in ('"', "'", '`'):
                    in_str = c
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return code[m.start():i + 1].lstrip('\n')
                i += 1
            return ""

        patch_funcs = _extract_func_names(partial_patch)
        orig_funcs = _extract_func_names(original_code)
        missing_names = sorted(orig_funcs - patch_funcs)

        if not missing_names:
            return partial_patch

        # Detect language
        is_python = bool(re.search(r'^\s*def\s+\w+\s*\(', original_code, re.MULTILINE))

        appended = []
        for fname in missing_names:
            if is_python:
                body = _extract_python_func_body(original_code, fname)
            else:
                body = _extract_js_func_body(original_code, fname)
            if body and len(body.strip()) > 5:
                appended.append(body.strip())
                logger.info(f"  [StitchFallback] 拼接函数: {fname} ({len(body)} 字符)")
            else:
                logger.warning(f"  [StitchFallback] 无法提取函数体: {fname}")

        if not appended:
            return partial_patch

        separator = "\n\n"
        stitched = partial_patch.rstrip() + separator + separator.join(appended)
        logger.info(f"[StitchFallback] 拼接 {len(appended)} 个函数，总长度 {len(stitched)} 字符")
        return stitched

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
