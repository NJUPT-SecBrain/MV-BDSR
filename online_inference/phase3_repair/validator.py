"""Validator for compilation and testing of patches."""

import subprocess
import tempfile
import shutil
import re
import difflib
import os
from pathlib import Path
from typing import Dict, Optional, Literal
from loguru import logger


class Validator:
    """Validator for compiling and testing patched code."""

    def __init__(
        self,
        compiler: str = "gcc",
        compile_timeout: int = 30,
        test_timeout: int = 600,  # 增加到10分钟（镜像拉取+测试执行）
        execution_mode: Literal["local", "docker"] = "local",
    ):
        """
        Initialize validator.

        Args:
            compiler: Compiler command (gcc, make, etc.)
            compile_timeout: Compilation timeout in seconds
            test_timeout: Test execution timeout in seconds (default 600s for docker pulls)
            execution_mode: Execution mode - "local" or "docker"
        """
        self.compiler = compiler
        self.compile_timeout = compile_timeout
        self.test_timeout = test_timeout
        self.execution_mode = execution_mode
        
        # Check Docker availability if docker mode
        if self.execution_mode == "docker":
            self._check_docker_available()

    def _ensure_temp_root(self) -> Optional[Path]:
        """
        Ensure TMPDIR exists and is usable.

        Returns:
            Resolved TMPDIR path when available, otherwise None.
        """
        tmpdir = os.environ.get("TMPDIR")
        if not tmpdir:
            return None

        try:
            tmp_path = Path(tmpdir).resolve()
            tmp_path.mkdir(parents=True, exist_ok=True)
            # Keep tempfile module aligned with runtime TMPDIR.
            tempfile.tempdir = str(tmp_path)
            return tmp_path
        except Exception as e:
            logger.warning(f"Failed to prepare TMPDIR {tmpdir}: {e}")
            return None

    def _make_temp_dir(self) -> Path:
        """
        Create temp dir with robust fallback.
        Prefer TMPDIR (used by run_online) and recreate it when missing.
        """
        preferred = self._ensure_temp_root()
        if preferred is not None:
            # mkdtemp can still fail if the TMPDIR root is deleted concurrently.
            # Retry once after re-creating TMPDIR, then fall back to system temp.
            try:
                return Path(tempfile.mkdtemp(dir=str(preferred)))
            except FileNotFoundError:
                self._ensure_temp_root()
                try:
                    return Path(tempfile.mkdtemp(dir=str(preferred)))
                except Exception as e:
                    logger.warning(f"Failed to create temp dir under TMPDIR={preferred}: {e}. Falling back to system temp.")
            except Exception as e:
                logger.warning(f"Failed to create temp dir under TMPDIR={preferred}: {e}. Falling back to system temp.")

        # Fallback: use system temp dir but make sure it exists.
        base = Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(dir=str(base)))

    def validate(
        self,
        buggy_code: str,
        patch: str,
        project_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Validate patch by compiling and testing.

        Args:
            buggy_code: Original buggy code
            patch: Generated patch
            project_info: Optional project info (test commands, paths, etc.)
                For Docker mode, should include:
                - docker_image: Docker image to use
                - repo: GitHub repo URL
                - commit: Git commit hash
                - file_path: Path to vulnerable file in repo
                - workdir: Working directory in container (default: /workspace)

        Returns:
            Validation result dictionary
        """
        mode = self.execution_mode
        # Recreate temp root proactively if users cleaned output/.tmp mid-run.
        self._ensure_temp_root()

        # Auto-detect PatchEval-style docker test scripts even when config is "local".
        # Typical signals:
        # - "# From ghcr.io/...." header
        # - "/workspace/..." paths
        # - explicit docker_image provided in project_info
        test_cmd = (project_info or {}).get("test_command") if project_info else None
        docker_image = (project_info or {}).get("docker_image") if project_info else None
        if mode == "local" and (docker_image or self._looks_like_docker_test_command(test_cmd)):
            mode = "docker"
            logger.info("检测到 Docker 风格测试脚本，自动切换到 docker 验证模式")

        logger.debug(f"Validating patch (mode: {mode})")

        if mode == "docker":
            # Ensure docker is available even if Validator was constructed in local mode.
            self._check_docker_available()

            # If docker_image missing, try to infer from test command header.
            if project_info is None:
                project_info = {}
            if not project_info.get("docker_image") and isinstance(test_cmd, str):
                inferred = self._extract_docker_image_from_test_cmd(test_cmd)
                if inferred:
                    project_info["docker_image"] = inferred
            return self._validate_in_docker(buggy_code, patch, project_info)

        return self._validate_local(buggy_code, patch, project_info)

    def _looks_like_docker_test_command(self, test_command: Optional[object]) -> bool:
        if not test_command:
            return False
        if not isinstance(test_command, str):
            return False
        s = test_command
        if "/workspace/" in s:
            return True
        if re.search(r"^\s*#\s*From\s+ghcr\.io/", s, flags=re.MULTILINE):
            return True
        return False

    def _extract_docker_image_from_test_cmd(self, test_cmd: str) -> Optional[str]:
        """Extract docker image from PatchEval header comment."""
        m = re.search(r"#\s*From\s+([\w./:_-]+)", test_cmd)
        if m:
            return m.group(1)
        return None
    
    def _detect_language(self, project_info: Optional[Dict] = None) -> str:
        """
        Detect programming language from project info.
        
        Args:
            project_info: Project info with buggy_code_file path
            
        Returns:
            Language name: 'c', 'cpp', 'python', 'javascript', 'go', etc.
        """
        if not project_info or "buggy_code_file" not in project_info:
            return "c"  # Default to C
        
        file_path = str(project_info["buggy_code_file"]).lower()
        
        if file_path.endswith((".py", ".pyw")):
            return "python"
        elif file_path.endswith((".js", ".mjs", ".cjs")):
            return "javascript"
        elif file_path.endswith((".cpp", ".cc", ".cxx", ".hpp", ".hxx")):
            return "cpp"
        elif file_path.endswith((".c", ".h")):
            return "c"
        elif file_path.endswith(".go"):
            return "go"
        elif file_path.endswith((".java",)):
            return "java"
        elif file_path.endswith((".rs",)):
            return "rust"
        elif file_path.endswith((".rb",)):
            return "ruby"
        elif file_path.endswith((".php",)):
            return "php"
        else:
            return "unknown"
    
    def _validate_local(
        self,
        buggy_code: str,
        patch: str,
        project_info: Optional[Dict] = None,
    ) -> Dict:
        """Validate patch locally (original logic)."""
        # Apply patch to get fixed code
        fixed_code = self._apply_patch(buggy_code, patch)

        if fixed_code is None:
            return {
                "success": False,
                "stage": "patch_application",
                "error_log": "Failed to apply patch",
            }

        # Detect language
        language = self._detect_language(project_info)
        logger.debug(f"Detected language: {language}")
        
        # 语言特定的验证
        if language == "python":
            # Python: 只做语法检查
            validation_result = self._validate_python(fixed_code)
        elif language == "javascript":
            # JavaScript: 只做语法检查（如果有 node）
            validation_result = self._validate_javascript(fixed_code)
        elif language in ["c", "cpp"]:
            # C/C++: 编译检查
            validation_result = self._compile(fixed_code, project_info)
        else:
            # 其他语言：跳过编译，只运行测试
            logger.warning(f"Unsupported language for compilation: {language}, skipping compile")
            validation_result = {"success": True, "output": "Compilation skipped for unsupported language"}
        
        if not validation_result["success"]:
            return {
                "success": False,
                "stage": "compilation",
                "error_log": validation_result["error_log"],
            }

        # Run tests (optional)
        # If no explicit test_command is provided, skip tests by default.
        if not (project_info and project_info.get("test_command")):
            return {
                "success": True,
                "compile_output": validation_result.get("output", ""),
                "test_output": "(skipped)",
            }

        test_result = self._run_tests(validation_result.get("executable"), project_info)

        if not test_result["success"]:
            return {
                "success": False,
                "stage": "testing",
                "error_log": test_result["error_log"],
            }

        # All passed
        return {
            "success": True,
            "compile_output": validation_result.get("output", ""),
            "test_output": test_result.get("output", ""),
        }

    def _apply_patch(self, buggy_code: str, patch: str) -> Optional[str]:
        """
        Apply patch to buggy code.

        Args:
            buggy_code: Original code
            patch: Patch to apply

        Returns:
            Fixed code or None if failed
        """
        # 检测补丁格式
        patch_stripped = patch.strip()
        
        # 检查是否是 unified diff 格式
        is_unified_diff = (
            patch_stripped.startswith("---") or 
            patch_stripped.startswith("diff --git") or
            patch_stripped.startswith("+++") or 
            "@@" in patch_stripped[:200]  # 检查前200个字符
        )
        
        if is_unified_diff:
            logger.debug("Detected unified diff format, attempting to apply")
            return self._apply_unified_diff(buggy_code, patch)
        else:
            # 假设 patch 就是完整的修复后代码
            logger.debug("Treating patch as complete fixed code")
            return patch
    
    def _normalize_patch_filename(self, patch: str, actual_relpath: str = "file.txt") -> str:
        """
        Normalize patch by replacing all filenames with the actual relpath.
        
        Handles cases where LLM generates patch with wrong filename from exemplars.
        
        Args:
            patch: Original patch string
            actual_relpath: The actual repo-relative path to use (can include directories)
            
        Returns:
            Normalized patch with corrected filenames
        """
        lines = patch.split('\n')
        normalized = []
        
        for line in lines:
            # Replace --- a/whatever with --- a/actual_filename
            if line.startswith('--- a/') or line.startswith('--- b/'):
                normalized.append(f"--- a/{actual_relpath}")
            elif line.startswith('+++ a/') or line.startswith('+++ b/'):
                normalized.append(f"+++ b/{actual_relpath}")
            elif line.startswith('diff --git'):
                # Replace: diff --git a/old.js b/old.js -> diff --git a/file.txt b/file.txt
                normalized.append(f"diff --git a/{actual_relpath} b/{actual_relpath}")
            else:
                normalized.append(line)
        
        return '\n'.join(normalized)

    def _infer_repo_relpath(self, project_info: Optional[Dict]) -> str:
        """Infer repo-relative file path for generating/applying patches.
        
        Expected CVEdataset layout:
          .../<CVE-XXXX-XXXX>/<commit_hash>/<repo_relpath>
          
        Examples:
          CVEdataset/CVE-2016-10548/da7bce7d/index.js
            → commit_dir = .../da7bce7d  → relpath = index.js
          CVEdataset/CVE-2015-8213/710e11d/django/utils/formats.py
            → commit_dir = .../710e11d   → relpath = django/utils/formats.py
          CVEdataset/CVE-2017-0360/5d7c4fa/trytond/tools/misc.py
            → commit_dir = .../5d7c4fa  → relpath = trytond/tools/misc.py
        """
        if not project_info:
            return "file.txt"

        # Prefer explicit repo-relative path if present.
        # file_path (from input.json / ground-truth metadata) takes priority over
        # the inferred repo_relpath, because input.json is authoritative.
        for k in ("file_path", "repo_relpath"):
            v = project_info.get(k)
            if isinstance(v, str) and v.strip():
                logger.debug(f"_infer_repo_relpath: using explicit '{k}' = '{v.strip()}'")
                return v.strip().lstrip("./")

        buggy = project_info.get("buggy_code_file")
        if isinstance(buggy, str) and buggy.strip():
            p = Path(buggy)
            try:
                parts = p.parts
                # Find the path component that looks like a CVE ID
                for i, part in enumerate(parts):
                    if part.upper().startswith("CVE-"):
                        # parts[i]   = CVE dir  (e.g. "CVE-2016-10548")
                        # parts[i+1] = commit dir (e.g. "da7bce7d...")
                        # relpath    = everything after parts[i+1]
                        if i + 2 < len(parts):
                            # Rebuild path up to and including the commit dir
                            # Path(*parts) fails on Windows with single-element; use os.path.join
                            commit_dir = Path(parts[0]).joinpath(*parts[1:i + 2])
                            rel = p.relative_to(commit_dir)
                            relpath = str(rel).replace("\\", "/")
                            logger.debug(
                                f"_infer_repo_relpath: CVE dir='{parts[i]}' "
                                f"commit='{parts[i+1]}' → relpath='{relpath}'"
                            )
                            return relpath
                # Fallback: use only the filename
                return p.name
            except Exception as e:
                logger.debug(f"_infer_repo_relpath fallback to filename: {e}")
                return p.name

        return "file.txt"

    def _ensure_unified_diff_patch(self, buggy_code: str, patch_or_code: str, project_info: Optional[Dict]) -> str:
        """Ensure the content mounted as fix.patch is a unified diff patch."""
        relpath = self._infer_repo_relpath(project_info)
        s = (patch_or_code or "").strip()

        # 如果 patch_or_code 为空或过短，直接失败（避免生成无意义的 diff）
        if not s or len(s) < 10:
            logger.error("LLM 输出为空或过短，无法生成有效补丁")
            raise ValueError("Generated patch is empty or too short")

        is_unified_diff = (
            s.startswith("---") or
            s.startswith("diff --git") or
            s.startswith("+++") or
            "@@" in s[:400]
        )

        if is_unified_diff:
            # Normalize filenames to the actual repo relpath to avoid mismatches.
            normalized = self._normalize_patch_filename(patch_or_code, relpath)
            # 验证补丁格式基本正确（至少包含 hunk header）
            if "@@" not in normalized:
                logger.error("Unified diff 缺少 hunk header (@@)")
                raise ValueError("Invalid unified diff format: missing hunk headers")
            return normalized

        # Treat as full fixed code, generate unified diff against buggy_code.
        logger.info(f"将完整修复代码转换为 unified diff（文件: {relpath}）")
        
        # 重要：不使用 keepends，让 difflib 自己处理换行符
        from_lines = buggy_code.splitlines(keepends=False)
        to_lines = patch_or_code.splitlines(keepends=False)
        
        # 检查是否真的有差异
        if from_lines == to_lines:
            logger.error("修复代码与原始代码完全相同，无需生成补丁")
            raise ValueError("Fixed code is identical to buggy code")
        
        # 使用 difflib 默认的换行符处理（lineterm=None 或不指定）
        diff_lines = list(difflib.unified_diff(
            from_lines,
            to_lines,
            fromfile=f"a/{relpath}",
            tofile=f"b/{relpath}",
            lineterm='',  # 让每行不带换行符
        ))
        
        if not diff_lines:
            logger.error("生成的 unified diff 为空")
            raise ValueError("Generated diff is empty")
        
        # 用 \n 连接所有行，并在末尾添加一个换行符
        diff_text = '\n'.join(diff_lines) + '\n'
        
        logger.debug(f"生成的 unified diff 长度: {len(diff_text)} 字符")
        logger.debug(f"Diff 行数: {len(diff_lines)}")
        
        # 验证格式：检查是否有异常的空行
        lines_preview = diff_text.split('\n')[:10]
        empty_count = sum(1 for line in lines_preview if line == '')
        if empty_count > 2:  # header 后可能有1-2个空行是正常的
            logger.warning(f"Diff 前10行中有 {empty_count} 个空行，可能格式有问题")
        
        return diff_text
    
    def _apply_unified_diff(self, original_code: str, patch: str) -> Optional[str]:
        """
        Apply unified diff patch to original code.
        
        Args:
            original_code: Original source code
            patch: Unified diff patch string
            
        Returns:
            Patched code or None if failed
        """
        import tempfile
        import subprocess
        import shutil
        
        try:
            # 创建临时目录
            temp_dir = self._make_temp_dir()
            
            # 从 patch 中提取文件名，如果提取不到就用 original.txt
            extracted_filename = self._extract_filename_from_patch(patch)
            if extracted_filename:
                # 如果补丁中有路径（如 lib/index.js），保留目录结构
                file_path = temp_dir / extracted_filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                file_path = temp_dir / "original.txt"
            
            # 写入原始代码
            file_path.write_text(original_code, encoding="utf-8")
            
            # 标准化补丁文件名（使用实际文件名）
            normalized_patch = self._normalize_patch_filename(patch, file_path.name)
            patch_file = temp_dir / "patch.diff"
            patch_file.write_text(normalized_patch, encoding="utf-8")
            
            # 使用 patch 命令应用补丁
            # --force: 强制应用，即使有冲突
            # --fuzz=3: 允许更宽松的上下文匹配
            # -p1: 剥离一层路径前缀（a/file -> file）
            result = subprocess.run(
                ["patch", "-p1", "--force", "--fuzz=3", "--no-backup-if-mismatch", str(file_path)],
                stdin=open(patch_file, 'r'),
                capture_output=True,
                text=True,
                timeout=30,  # 增加超时时间
                cwd=str(temp_dir),
            )
            
            # patch 命令即使有警告也可能返回非0，检查文件是否被修改
            if file_path.exists():
                patched_code = file_path.read_text(encoding="utf-8")
                
                # 如果代码有变化，认为应用成功
                if patched_code != original_code:
                    logger.debug("Patch applied successfully")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return patched_code
                elif result.returncode == 0:
                    # patch 命令成功但代码未变化（可能是空补丁）
                    logger.warning("Patch applied but code unchanged")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return patched_code
            
            # patch 失败，尝试 git apply
            if result.returncode != 0:
                logger.warning(f"patch command failed: {result.stderr.strip()[:200]}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return self._apply_patch_with_git(original_code, normalized_patch)
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
            
        except FileNotFoundError:
            logger.warning("`patch` command not found, trying git apply")
            normalized_patch = self._normalize_patch_filename(patch, "file.txt")
            return self._apply_patch_with_git(original_code, normalized_patch)
        except subprocess.TimeoutExpired:
            logger.error("patch command timed out (30s)")
            return None
        except Exception as e:
            logger.error(f"Failed to apply unified diff: {e}")
            return None
    
    def _apply_patch_with_git(self, original_code: str, patch: str, temp_dir: Optional[Path] = None) -> Optional[str]:
        """
        Fallback: use git apply to apply patch.
        
        Args:
            original_code: Original source code
            patch: Unified diff patch (should be normalized with correct filename)
            temp_dir: Optional existing temp directory
            
        Returns:
            Patched code or None if failed
        """
        import tempfile
        import subprocess
        import shutil
        
        cleanup_temp = False
        if temp_dir is None:
            temp_dir = self._make_temp_dir()
            cleanup_temp = True
        
        try:
            # 初始化 git repo
            subprocess.run(["git", "init"], cwd=str(temp_dir), capture_output=True, timeout=5)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(temp_dir), capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=str(temp_dir), capture_output=True)
            
            # 从 patch 中提取文件名
            file_name = self._extract_filename_from_patch(patch) or "file.txt"
            
            # 创建目录结构（如果文件名包含路径）
            original_file = temp_dir / file_name
            original_file.parent.mkdir(parents=True, exist_ok=True)
            original_file.write_text(original_code, encoding="utf-8")
            
            # git add
            subprocess.run(["git", "add", "-A"], cwd=str(temp_dir), capture_output=True, timeout=5)
            
            # git commit (需要一个初始提交)
            subprocess.run(
                ["git", "commit", "-m", "initial"],
                cwd=str(temp_dir),
                capture_output=True,
                timeout=5,
            )
            
            # 写入补丁文件
            patch_file = temp_dir / "patch.diff"
            patch_file.write_text(patch, encoding="utf-8")
            
            # git apply with multiple fallback options
            # 首先尝试严格模式
            result = subprocess.run(
                ["git", "apply", "--ignore-whitespace", str(patch_file)],
                cwd=str(temp_dir),
                capture_output=True,
                text=True,
                timeout=15,
            )
            
            if result.returncode != 0:
                # 尝试 3-way merge 模式（更宽容）
                logger.debug("Trying git apply with 3-way merge")
                result = subprocess.run(
                    ["git", "apply", "--3way", "--ignore-whitespace", str(patch_file)],
                    cwd=str(temp_dir),
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            
            if result.returncode != 0:
                logger.warning(f"git apply failed: {result.stderr.strip()[:200]}")
                return None
            
            # 读取修复后的代码
            if original_file.exists():
                patched_code = original_file.read_text(encoding="utf-8")
                return patched_code
            else:
                logger.error(f"File not found after git apply: {original_file}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to apply patch with git: {e}")
            return None
        finally:
            if cleanup_temp:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _extract_filename_from_patch(self, patch: str) -> Optional[str]:
        """
        Extract filename from unified diff patch.
        
        Args:
            patch: Unified diff patch string
            
        Returns:
            Filename or None
        """
        import re
        
        # 查找 --- a/filename 或 +++ b/filename
        match = re.search(r'(?:---|\+\+\+)\s+[ab]/(.+?)(?:\s|$)', patch)
        if match:
            return match.group(1)
        
        # 查找 diff --git a/filename b/filename
        match = re.search(r'diff --git a/(.+?)\s+b/', patch)
        if match:
            return match.group(1)
        
        return None

    def _compile(self, code: str, project_info: Optional[Dict] = None) -> Dict:
        """
        Compile code.

        Args:
            code: Source code to compile
            project_info: Optional project info

        Returns:
            Compilation result
        """
        logger.debug("Compiling code")

        # Determine compile mode
        # - "compile_only" (default): gcc -c code.c -o code.o   (no need for main)
        # - "syntax_only": gcc -fsyntax-only code.c             (fastest)
        # - "link": gcc code.c -o a.out                         (needs main)
        compile_mode = "compile_only"
        if project_info and project_info.get("compile_mode"):
            compile_mode = str(project_info["compile_mode"]).lower()

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write code to file
            code_file = temp_path / "code.c"
            code_file.write_text(code)

            output_file = temp_path / "a.out"
            obj_file = temp_path / "code.o"

            if compile_mode == "syntax_only":
                compile_cmd = [self.compiler, "-fsyntax-only", str(code_file)]
            elif compile_mode == "link":
                compile_cmd = [self.compiler, str(code_file), "-o", str(output_file)]
            else:
                # Default compile-only (produces object file, no linking)
                compile_cmd = [self.compiler, "-c", str(code_file), "-o", str(obj_file)]

            # Add compile flags if provided
            if project_info and "compile_flags" in project_info:
                compile_cmd.extend(project_info["compile_flags"])

            try:
                result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.compile_timeout,
                    cwd=temp_dir,
                )

                if result.returncode == 0:
                    return {
                        "success": True,
                        "output": result.stdout,
                        "executable": str(output_file) if compile_mode == "link" else None,
                        "object_file": str(obj_file) if compile_mode == "compile_only" else None,
                        "compile_mode": compile_mode,
                    }
                else:
                    return {
                        "success": False,
                        "error_log": result.stderr or result.stdout,
                    }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error_log": f"Compilation timeout ({self.compile_timeout}s)",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error_log": f"Compilation error: {str(e)}",
                }

    def _run_tests(self, executable: Optional[str], project_info: Optional[Dict] = None) -> Dict:
        """
        Run tests on compiled code.

        Args:
            executable: Path to compiled executable
            project_info: Optional project info

        Returns:
            Test result
        """
        logger.debug("Running tests")

        # Get test command from project_info or use default
        if project_info and "test_command" in project_info:
            test_cmd_raw = project_info["test_command"]
            if isinstance(test_cmd_raw, str):
                # 如果是字符串（可能是多行脚本），用 bash -c 执行
                test_cmd = ["bash", "-c", test_cmd_raw]
            else:
                # 如果已经是列表，直接使用
                test_cmd = test_cmd_raw
        else:
            # Default: just run the executable
            if executable is None:
                return {
                    "success": False,
                    "error_log": "No executable to test",
                }
            test_cmd = [executable]

        try:
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
            )

            # Assume exit code 0 means tests passed
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                }
            else:
                return {
                    "success": False,
                    "error_log": result.stderr or result.stdout or f"Test failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error_log": f"Test timeout ({self.test_timeout}s)",
            }
        except Exception as e:
            return {
                "success": False,
                "error_log": f"Test error: {str(e)}",
            }

    def quick_syntax_check(self, code: str) -> bool:
        """
        Quick syntax check without full compilation.

        Args:
            code: Source code

        Returns:
            True if syntax is valid
        """
        # Use compiler with syntax-only flag
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [self.compiler, "-fsyntax-only", temp_file],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def _quick_syntax_check(self, code: str, language: str) -> Dict:
        """快速语法检查（不需要应用补丁，直接检查生成的代码）"""
        if language == "python":
            return self._validate_python(code)
        elif language == "javascript":
            return self._validate_javascript(code)
        else:
            return {"success": True}
    
    def _validate_python(self, code: str) -> Dict:
        """
        Validate Python code using ast.parse for syntax checking.
        
        Args:
            code: Python source code
            
        Returns:
            Validation result dictionary
        """
        import ast
        
        try:
            ast.parse(code)
            return {
                "success": True,
                "output": "Python syntax check passed",
            }
        except SyntaxError as e:
            return {
                "success": False,
                "error_log": f"Python syntax error at line {e.lineno}: {e.msg}",
            }
        except Exception as e:
            return {
                "success": False,
                "error_log": f"Python validation error: {str(e)}",
            }
    
    def _validate_javascript(self, code: str) -> Dict:
        """
        Validate JavaScript code (basic check, requires node if available).
        
        Args:
            code: JavaScript source code
            
        Returns:
            Validation result dictionary
        """
        # 尝试使用 node 检查语法
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # node --check 只检查语法，不执行
                result = subprocess.run(
                    ["node", "--check", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "output": "JavaScript syntax check passed",
                    }
                else:
                    return {
                        "success": False,
                        "error_log": f"JavaScript syntax error: {result.stderr}",
                    }
            finally:
                Path(temp_file).unlink(missing_ok=True)
                
        except FileNotFoundError:
            # node 不可用，跳过语法检查
            logger.warning("Node.js not available, skipping JavaScript syntax check")
            return {
                "success": True,
                "output": "JavaScript validation skipped (node not available)",
            }
        except Exception as e:
            return {
                "success": False,
                "error_log": f"JavaScript validation error: {str(e)}",
            }
    
    def _check_docker_available(self):
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not available")
            logger.info(f"Docker is available: {result.stdout.decode().strip()}")
        except Exception as e:
            raise RuntimeError(f"Docker check failed: {e}")
    
    def _ensure_docker_image(self, image: str):
        """预拉取 Docker 镜像（如果本地不存在）以避免测试超时"""
        try:
            # 检查镜像是否已存在
            check = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=10,
            )
            
            if check.returncode == 0:
                logger.debug(f"Docker 镜像已存在: {image}")
                return
            
            # 镜像不存在，预拉取
            logger.info(f"预拉取 Docker 镜像: {image}（可能需要几分钟）")
            pull = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟拉取超时
            )
            
            if pull.returncode == 0:
                logger.info(f"Docker 镜像拉取成功: {image}")
            else:
                logger.warning(f"Docker 镜像拉取失败（将在运行时重试）: {pull.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Docker 镜像拉取超时（10分钟），将在运行时重试")
        except Exception as e:
            logger.warning(f"Docker 镜像拉取检查失败: {e}")
    
    def _validate_in_docker(
        self,
        buggy_code: str,
        patch: str,
        project_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Validate patch in Docker container.
        
        Workflow:
        1. Create temp directory for patch file
        2. Run a one-off container with:
           - Mount patch file into container (default: /workspace/fix.patch)
           - Execute test script provided by project_info["test_command"]
        3. Collect stdout/stderr and return result

        Notes:
        - PatchEval-style test scripts usually apply /workspace/fix.patch themselves.
          Therefore, we DO NOT apply the patch in this wrapper to avoid double-apply
          and path mismatches.
        """
        if not project_info:
            return {
                "success": False,
                "stage": "docker_setup",
                "error_log": "Docker mode requires project_info with docker_image and test_command",
            }
        
        docker_image = project_info.get("docker_image")
        test_command = project_info.get("test_command")
        workdir = project_info.get("workdir", "/workspace")
        patch_path_in_container = project_info.get("patch_path_in_container", f"{workdir.rstrip('/')}/fix.patch")
        
        if not docker_image:
            return {
                "success": False,
                "stage": "docker_setup",
                "error_log": "docker_image not provided in project_info",
            }
        
        if not test_command:
            return {
                "success": False,
                "stage": "docker_setup",
                "error_log": "test_command not provided in project_info",
            }

        # Note: test.patch is pre-baked inside PatchEval Docker images at /workspace/test.patch.
        # No need to reject commands that reference it - the file is already available in the container.
        # We only need to mount fix.patch (our generated patch).
        logger.info(f"Running validation in Docker: {docker_image}")
        
        # 预拉取镜像（避免每次迭代都超时等待拉取）
        self._ensure_docker_image(docker_image)
        
        # Create temp directory for patch
        temp_dir = self._make_temp_dir()
        patch_file = temp_dir / "fix.patch"
        
        try:
            # Write patch to file (must be unified diff for PatchEval scripts: git apply /workspace/fix.patch)
            try:
                patch_to_mount = self._ensure_unified_diff_patch(buggy_code, patch, project_info)
            except ValueError as e:
                return {
                    "success": False,
                    "stage": "patch_generation",
                    "error_log": f"补丁生成失败: {str(e)}",
                }
            
            # 在发送到 Docker 前，先在本地验证语法（Python/JavaScript）
            # 这可以快速拒绝明显不完整的代码，避免浪费 Docker 时间
            language = self._detect_language(project_info)
            if language in ["python", "javascript"]:
                syntax_check = self._quick_syntax_check(patch, language)
                if not syntax_check["success"]:
                    err_msg = syntax_check.get("error_log") or syntax_check.get("error") or str(syntax_check)
                    logger.warning(f"本地语法检查失败: {err_msg[:200]}")
                    return {
                        "success": False,
                        "stage": "syntax_check",
                        "error_log": f"生成的代码有语法错误（在发送到 Docker 前检测到）:\n{err_msg[:500]}",
                    }
            
            patch_file.write_text(patch_to_mount, encoding="utf-8")
            logger.debug(f"补丁文件已写入: {patch_file}（{len(patch_to_mount)} 字节）")
            
            # Build Docker command
            # Format: docker run --rm --platform linux/amd64 -v {patch}:{patch_path_in_container}:ro --workdir {workdir} {image} bash -lc "{cmd}"
            docker_cmd = [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",  # 明确指定平台，避免 ARM64 警告
                "-v", f"{patch_file.absolute()}:{patch_path_in_container}:ro",
                "--workdir", workdir,
                docker_image,
                "bash", "-lc",
            ]

            # Execute the provided test script as-is (it usually applies fix.patch itself).
            docker_cmd.append(str(test_command))
            
            logger.debug(f"Docker command: {' '.join(docker_cmd[:6])}...")
            
            # Execute
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=self.test_timeout + 60,  # Extra time for container startup
                text=True,
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            logger.debug(f"Docker exit code: {result.returncode}")
            logger.debug(f"Docker stdout: {stdout[:500]}")
            logger.debug(f"Docker stderr: {stderr[:500]}")
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "test_output": stdout,
                    "stderr": stderr,
                }
            else:
                return {
                    "success": False,
                    "stage": "docker_testing",
                    "error_log": f"Docker test failed (exit {result.returncode}):\n{stderr}\n{stdout}",
                }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stage": "docker_testing",
                "error_log": f"Docker test timeout ({self.test_timeout + 60}s)",
            }
        except Exception as e:
            logger.exception("Docker execution raised unexpected exception")
            return {
                "success": False,
                "stage": "docker_testing",
                "error_log": f"Docker execution error: {str(e)}",
            }
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
