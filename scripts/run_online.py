#!/usr/bin/env python3
"""批量在线修复：遍历 vul_files 目录下所有 CVE，依次执行在线修复流程。

用法：
  python scripts/run_online.py \\
    --vul-files-dir data/processed/vul_files \\
    --indices-dir data/indices/test_index \\
    --output-dir results/online_batch \\
    --limit 10

断点续跑：
  程序中断后重新运行相同命令即可自动跳过已处理的 CVE。
  使用 --no-resume 强制从头重跑所有 CVE。
  使用 --retry-failed 在已跳过成功 CVE 的同时重跑上次失败的 CVE。
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from config.paths import ProjectPaths
from utils.helpers import load_yaml, read_file, write_file
from utils.logger import setup_logger
from models import LLMInterface, EmbeddingModel, GraphCodeBERTModel
from models.llm_interface import LLMAuthError, LLMError
from static_analysis import (
    DataFlowAnalyzer,
    ControlFlowAnalyzer,
    ReachabilityChecker,
)
from offline_indexing import VectorStore
from online_inference.phase1_diagnosis import DiagnosticAgent, create_default_tool_registry
from online_inference.phase2_retrieval import QueryGenerator, Retriever, Reranker
from online_inference.phase3_repair import RepairAgent, Validator


def find_vul_files_in_cve(cve_dir: Path) -> List[Path]:
    """在 CVE 目录下递归查找所有源代码文件。
    
    返回按文件大小排序的代码文件列表（通常主文件较大）。
    """
    code_extensions = {".py", ".go", ".c", ".cpp", ".js", ".java", ".rs", ".rb", ".php"}
    files = []
    
    for ext in code_extensions:
        files.extend(cve_dir.rglob(f"*{ext}"))
    
    # 排除测试文件名常见模式
    filtered = [
        f for f in files
        if not any(pattern in f.name.lower() for pattern in ["test_", "_test.", "test."])
    ]
    
    # 按文件大小排序（通常主漏洞文件较大）
    if filtered:
        return sorted(filtered, key=lambda f: f.stat().st_size, reverse=True)
    return sorted(files, key=lambda f: f.stat().st_size, reverse=True)


def load_test_cmds(cve_dir: Path) -> Optional[Dict]:
    """从 CVE 目录加载 test_cmds.json（如果存在）"""
    test_cmds_file = cve_dir / "test_cmds.json"
    if test_cmds_file.exists():
        try:
            with test_cmds_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"无法读取 {test_cmds_file}: {e}")
    return None


def extract_docker_image_from_test_cmd(test_cmd: str) -> Optional[str]:
    """从测试命令注释中提取 Docker 镜像名称。
    
    示例注释格式：
    # From ghcr.io/anonymous2578-data/cve-2023-25173:latest
    """
    import re
    match = re.search(r'#\s*From\s+([\w./:_-]+)', test_cmd)
    if match:
        return match.group(1)
    return None


class CheckpointManager:
    """断点续跑管理器：记录每个 CVE 的处理状态，支持中断后自动恢复。

    checkpoint.json 结构：
    {
        "created_at": "...",
        "last_updated": "...",
        "vul_files_dir": "...",
        "processed": {
            "CVE-2015-1326": {"status": "failed",  "finished_at": "..."},
            "CVE-2015-3295": {"status": "success", "finished_at": "..."},
            "CVE-2016-0001": {"status": "skipped", "finished_at": "..."},
            "CVE-2016-0002": {"status": "error",   "finished_at": "...", "error": "..."}
        }
    }
    """

    CHECKPOINT_FILE = "checkpoint.json"

    def __init__(self, output_dir: Path, vul_files_dir: str):
        self.output_dir = output_dir
        self.path = output_dir / self.CHECKPOINT_FILE
        self._data: Dict = {}
        self._vul_files_dir = vul_files_dir

    # ──────────────────────────────────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """加载已有断点文件。返回 True 表示成功加载（续跑场景）。"""
        if not self.path.exists():
            self._data = {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "vul_files_dir": self._vul_files_dir,
                "processed": {},
            }
            return False

        try:
            with self.path.open("r", encoding="utf-8") as f:
                self._data = json.load(f)
            if "processed" not in self._data:
                self._data["processed"] = {}
            logger.info(
                f"✅ 加载断点文件: {self.path}，已记录 {len(self._data['processed'])} 个 CVE"
            )
            return True
        except Exception as e:
            logger.warning(f"断点文件读取失败（将重新开始）: {e}")
            self._data = {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "vul_files_dir": self._vul_files_dir,
                "processed": {},
            }
            return False

    def _save_atomic(self):
        """原子写入：先写临时文件再 rename，防止中断损坏。"""
        self._data["last_updated"] = datetime.now().isoformat()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            Path(tmp_path).replace(self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # 查询
    # ──────────────────────────────────────────────────────────────────────────

    def is_done(self, cve_id: str) -> bool:
        """是否已成功处理（success / skipped）过。"""
        rec = self._data["processed"].get(cve_id)
        return rec is not None and rec.get("status") in ("success", "skipped")

    def was_failed(self, cve_id: str) -> bool:
        """上次处理是否失败/出错。"""
        rec = self._data["processed"].get(cve_id)
        return rec is not None and rec.get("status") in ("failed", "error")

    def is_processed(self, cve_id: str) -> bool:
        """是否已有任何处理记录（含失败）。"""
        return cve_id in self._data["processed"]

    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self._data["processed"].values():
            s = rec.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    # ──────────────────────────────────────────────────────────────────────────
    # 更新
    # ──────────────────────────────────────────────────────────────────────────

    def mark(self, cve_id: str, status: str, error: Optional[str] = None):
        """记录某个 CVE 的处理结果并立即持久化。

        status 取值: "success" | "failed" | "skipped" | "error"
        """
        rec: Dict = {"status": status, "finished_at": datetime.now().isoformat()}
        if error:
            rec["error"] = error[:300]
        self._data["processed"][cve_id] = rec
        self._save_atomic()
        logger.debug(f"📌 CheckPoint [{status}] {cve_id}")

    def reset(self):
        """清空断点记录（--no-resume 时调用）。"""
        self._data = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "vul_files_dir": self._vul_files_dir,
            "processed": {},
        }
        if self.path.exists():
            self.path.unlink()
        logger.info("🗑️  已清空断点记录，从头开始")

    def clear_failed(self):
        """删除所有失败记录（--retry-failed 时调用，使这些 CVE 可被重跑）。"""
        before = len(self._data["processed"])
        self._data["processed"] = {
            k: v
            for k, v in self._data["processed"].items()
            if v.get("status") not in ("failed", "error")
        }
        after = len(self._data["processed"])
        cleared = before - after
        if cleared:
            self._save_atomic()
            logger.info(f"🔄 已清除 {cleared} 个失败记录，这些 CVE 将被重新处理")


def load_cve_metadata(cve_id: str, input_json_path: Path) -> Optional[Dict]:
    """从 input.json 加载 CVE 的元数据（repo, commit 等）"""
    if not input_json_path.exists():
        logger.warning(f"input.json 不存在: {input_json_path}")
        return None
    
    try:
        with input_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 查找对应的 CVE
        for item in data:
            if item.get("cve_id") == cve_id:
                return item
        
        logger.warning(f"未在 input.json 中找到 {cve_id}")
        return None
    except Exception as e:
        logger.warning(f"读取 input.json 出错: {e}")
        return None


def repair_single_cve(
    cve_id: str,
    buggy_code_file: Path,
    test_cmds: Optional[Dict],
    cve_metadata: Optional[Dict],
    models_dict: Dict,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """对单个 CVE 执行在线修复流程"""
    logger.info(f"\n{'='*60}")
    logger.info(f"修复 CVE: {cve_id}")
    logger.info(f"漏洞文件: {buggy_code_file}")
    logger.info(f"{'='*60}")
    
    # 读取漏洞代码
    buggy_code = buggy_code_file.read_text(encoding="utf-8", errors="replace")
    
    # Phase 1: Diagnosis
    logger.info("=== Phase 1: Diagnostic Analysis ===")
    diagnosis_result = models_dict["diagnostic_agent"].diagnose(buggy_code)
    enhanced_context = diagnosis_result["enhanced_context"]
    diagnostic_report = diagnosis_result.get("diagnostic_report", {})  # 新增：提取诊断报告
    logger.info("诊断分析完成")
    
    # Phase 2: Retrieval and Re-ranking
    logger.info("=== Phase 2: Retrieval & Re-ranking ===")
    queries = models_dict["query_generator"].generate_queries(enhanced_context)
    logger.info("生成多视角查询")
    
    candidates = models_dict["retriever"].retrieve(queries)
    logger.info(f"检索到 {len(candidates)} 个候选样本")
    
    top_exemplars = models_dict["reranker"].rerank(
        enhanced_context,
        buggy_code,
        candidates,
        top_k=config["online"]["phase2_retrieval"]["final_top_k"],
    )
    logger.info(f"重排后选择 top-{len(top_exemplars)} 样例")
    
    # Phase 3: Iterative Repair
    logger.info("=== Phase 3: Iterative Repair ===")
    
    # 构建 project_info（包含测试命令和 Docker 信息）
    # 构建 project_info（至少包含文件路径，用于语言检测）
    project_info = {
        "buggy_code_file": str(buggy_code_file),
        "cve_id": cve_id,
    }

    # 推断仓库内相对路径（PatchEval 容器内 git apply 依赖正确的 relpath）
    # 期望布局: .../<CVE-XXXX-XXXX>/<commit_hash>/<repo_relpath>
    # 例如:
    #   CVEdataset/CVE-2016-10548/da7bce7d/index.js          → index.js
    #   CVEdataset/CVE-2015-8213/710e11d/django/utils/formats.py → django/utils/formats.py
    #   CVEdataset/CVE-2017-0360/5d7c4fa/trytond/tools/misc.py   → trytond/tools/misc.py
    try:
        parts = buggy_code_file.parts
        repo_relpath = buggy_code_file.name  # fallback
        for i, part in enumerate(parts):
            if part.upper().startswith("CVE-"):
                # parts[i] = CVE 目录, parts[i+1] = commit 目录
                # relpath = commit 目录之后的所有部分
                if i + 2 < len(parts):
                    commit_dir = Path(parts[0]).joinpath(*parts[1 : i + 2])
                    repo_relpath = str(buggy_code_file.relative_to(commit_dir)).replace("\\", "/")
                break
        project_info["repo_relpath"] = repo_relpath
        logger.debug(f"推断 repo_relpath: {repo_relpath}")
    except Exception as e:
        project_info["repo_relpath"] = buggy_code_file.name
        logger.debug(f"repo_relpath fallback to filename: {e}")
    
    if test_cmds:
        # 优先使用 unit_test_cmd，fallback 到 poc_test_cmd
        test_cmd = test_cmds.get("unit_test_cmd") or test_cmds.get("poc_test_cmd")
        if test_cmd:
            project_info["test_command"] = test_cmd
            project_info["programming_language"] = test_cmds.get("programming_language")
            
            # 提取 Docker 镜像名（不依赖 execution_mode，Validator 会自动识别并切换）
            docker_image = extract_docker_image_from_test_cmd(test_cmd) if isinstance(test_cmd, str) else None
            if docker_image:
                project_info["docker_image"] = docker_image
                logger.info(f"Docker 镜像(从 test_cmd 提取): {docker_image}")

            # 从 metadata 中获取 repo/commit/file_path 信息（如存在）
            if cve_metadata:
                repo = cve_metadata.get("repo")
                if repo:
                    project_info["repo"] = repo
                
                vul_funcs = cve_metadata.get("vul_func", [])
                if vul_funcs:
                    commit = vul_funcs[0].get("commit")
                    file_path = vul_funcs[0].get("file_path")
                    if commit:
                        project_info["commit"] = commit
                    if file_path:
                        project_info["file_path"] = file_path
                
                logger.info(f"Repo: {repo}, Commit: {project_info.get('commit', 'N/A')}")
            
            logger.info(f"加载测试命令: {test_cmd[:80]}...")
    
    repair_result = models_dict["repair_agent"].repair(
        buggy_code, 
        top_exemplars,
        diagnostic_report=diagnostic_report,  # 新增：传递诊断报告
        project_info=project_info,
    )
    
    # 保存结果
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"{cve_id}_result.json"
    result_data = {
        "cve_id": cve_id,
        "buggy_code_file": str(buggy_code_file),
        "success": repair_result["success"],
        "iterations": repair_result["iterations"],
        "final_patch": repair_result.get("final_patch"),
        "repair_history": repair_result.get("history", []),
    }
    
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果: {'成功' if repair_result['success'] else '失败'}")
    logger.info(f"迭代次数: {repair_result['iterations']}")
    logger.info(f"结果已保存: {result_file}")
    
    # 如果成功，也保存补丁文件
    if repair_result["success"] and repair_result.get("final_patch"):
        patch_file = output_dir / f"{cve_id}_patch.diff"
        patch_file.write_text(repair_result["final_patch"], encoding="utf-8")
        logger.info(f"补丁已保存: {patch_file}")
    
    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="MV-BDSR 批量在线修复（支持断点续跑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
断点续跑示例：
  # 首次运行
  python scripts/run_online.py --vul-files-dir data/processed/CVEdataset --output-dir results/run1

  # 中断后续跑（直接重新运行相同命令即可，已处理 CVE 自动跳过）
  python scripts/run_online.py --vul-files-dir data/processed/CVEdataset --output-dir results/run1

  # 重跑所有失败的 CVE（跳过已成功的）
  python scripts/run_online.py --vul-files-dir data/processed/CVEdataset --output-dir results/run1 --retry-failed

  # 完全从头开始（忽略所有历史）
  python scripts/run_online.py --vul-files-dir data/processed/CVEdataset --output-dir results/run1 --no-resume
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--vul-files-dir",
        type=str,
        default="data/processed/vul_files",
        help="vul_files 根目录",
    )
    parser.add_argument(
        "--indices-dir",
        type=str,
        default=None,
        help="离线索引目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/online_batch",
        help="结果输出目录",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理多少个 CVE（用于测试）",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "mock"],
        help="LLM 提供商（覆盖配置文件）",
    )
    # ── 断点续跑控制 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="忽略断点文件，从头重新处理所有 CVE",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="重新处理上次运行中失败/出错的 CVE（跳过已成功的）",
    )

    args = parser.parse_args()

    # 加载配置
    config = load_yaml(args.config)
    
    # 设置日志
    setup_logger(
        log_file=config["logging"]["log_file"],
        level=config["logging"]["level"],
    )

    logger.info("启动批量在线修复流程")
    logger.info(f"vul_files 目录: {args.vul_files_dir}")
    logger.info(f"输出目录: {args.output_dir}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统一临时目录到输出目录下，避免 /tmp 分区空间不足导致运行中断
    tmp_root = output_dir / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_root.resolve())
    tempfile.tempdir = os.environ["TMPDIR"]
    logger.info(f"TEMP 目录: {os.environ['TMPDIR']}")

    # ── 断点续跑初始化 ────────────────────────────────────────────────────────
    checkpoint = CheckpointManager(output_dir, args.vul_files_dir)
    resuming = checkpoint.load()  # True = 续跑，False = 首次运行

    if args.no_resume:
        checkpoint.reset()
        resuming = False
    elif args.retry_failed:
        checkpoint.clear_failed()
        if resuming:
            logger.info("▶ 续跑模式（--retry-failed）：跳过已成功 CVE，重跑失败 CVE")
    elif resuming:
        stats = checkpoint.stats()
        logger.info(f"▶ 续跑模式：断点状态 → {stats}")
    else:
        logger.info("▶ 首次运行，断点文件已创建")

    # 初始化路径
    paths = ProjectPaths()
    indices_dir = args.indices_dir or paths.indices

    # 初始化模型（只初始化一次，所有 CVE 共享）
    logger.info("初始化模型...")
    
    provider = (args.llm_provider or config["models"]["llm"]["provider"]).lower()
    api_key = config["models"]["llm"].get("api_key") or os.getenv("OPENAI_API_KEY")
    base_url = config["models"]["llm"].get("base_url")
    
    if not api_key and provider in ["openai", "anthropic"]:
        logger.warning("⚠️  未找到 API Key，建议使用 --llm-provider mock 进行测试")
    
    llm = LLMInterface(
        provider=provider,
        model_name=config["models"]["llm"]["model_name"],
        api_key=api_key,
        base_url=base_url,
        temperature=config["models"]["llm"]["temperature"],
        max_tokens=config["models"]["llm"]["max_tokens"],
    )

    embedding_model = EmbeddingModel(
        model_name=config["models"]["embeddings"]["model_name"],
        device=config["experiment"]["device"],
    )

    graphcodebert = GraphCodeBERTModel(
        model_name=config["models"]["graphcodebert"]["model_name"],
        device=config["experiment"]["device"],
        max_length=config["models"]["graphcodebert"]["max_length"],
    )

    # 加载向量索引
    logger.info(f"加载索引: {indices_dir}")
    vector_store = VectorStore(
        dimension=config["models"]["embeddings"]["dimension"],
    )
    vector_store.load(indices_dir)

    # 初始化在线组件
    logger.info("初始化在线修复组件...")
    
    # Phase 1
    tool_registry = create_default_tool_registry(
        DataFlowAnalyzer(),
        ControlFlowAnalyzer(),
        ReachabilityChecker(),
    )
    diagnostic_agent = DiagnosticAgent(
        llm,
        tool_registry,
        max_iterations=config["online"]["phase1_diagnosis"]["max_tool_calls"],
    )
    
    # Phase 2
    query_generator = QueryGenerator(llm)
    retriever = Retriever(
        vector_store,
        embedding_model,
        top_k_per_view=config["online"]["phase2_retrieval"]["top_k_per_view"],
    )
    reranker = Reranker(
        graphcodebert,
        embedding_model,
        text_weight=config["online"]["phase2_retrieval"]["reranking"]["text_weight"],
        code_weight=config["online"]["phase2_retrieval"]["reranking"]["graphcodebert_weight"],
    )
    
    # Phase 3
    execution_mode = config["online"]["phase3_repair"]["validation"].get("execution_mode", "local")
    logger.info(f"验证模式: {execution_mode}")
    
    validator = Validator(
        compiler=config["online"]["phase3_repair"]["validation"]["compiler"],
        compile_timeout=config["online"]["phase3_repair"]["validation"]["compile_timeout"],
        test_timeout=config["online"]["phase3_repair"]["validation"]["test_timeout"],
        execution_mode=execution_mode,
    )
    repair_agent = RepairAgent(
        llm,
        validator,
        max_iterations=config["online"]["phase3_repair"]["max_iterations"],
    )
    
    models_dict = {
        "diagnostic_agent": diagnostic_agent,
        "query_generator": query_generator,
        "retriever": retriever,
        "reranker": reranker,
        "repair_agent": repair_agent,
    }

    # 遍历所有 CVE 目录
    vul_files_dir = Path(args.vul_files_dir)
    if not vul_files_dir.exists():
        logger.error(f"vul_files 目录不存在: {vul_files_dir}")
        return

    cve_dirs = sorted([d for d in vul_files_dir.iterdir() if d.is_dir() and d.name.startswith("CVE-")])
    logger.info(f"找到 {len(cve_dirs)} 个 CVE 目录")

    if args.limit:
        cve_dirs = cve_dirs[:args.limit]
        logger.info(f"限制处理前 {args.limit} 个")

    # 加载 input.json（用于获取 repo/commit 信息）
    input_json_path = paths.raw_data / "input.json"
    logger.info(f"input.json 路径: {input_json_path}")
    
    # 统计（本次运行）
    total = 0       # 本次实际处理数
    success = 0
    failed = 0
    skipped = 0     # 无源文件跳过
    resumed = 0     # 因断点跳过数

    # 处理每个 CVE
    for idx, cve_dir in enumerate(cve_dirs):
        cve_id = cve_dir.name

        # ── 断点检查：已成功 CVE 直接跳过 ──────────────────────────────────
        if checkpoint.is_done(cve_id):
            resumed += 1
            logger.debug(f"⏭ 跳过（已完成）: {cve_id}")
            continue
        
        try:
            # 查找漏洞代码文件
            vul_files = find_vul_files_in_cve(cve_dir)
            
            if not vul_files:
                logger.warning(f"跳过 {cve_id}: 未找到源代码文件")
                skipped += 1
                checkpoint.mark(cve_id, "skipped")
                continue
            
            # 使用第一个文件（通常是主漏洞文件）
            buggy_code_file = vul_files[0]
            
            # 加载测试命令
            test_cmds = load_test_cmds(cve_dir)
            
            # 加载 CVE 元数据（如果需要 Docker 模式）
            cve_metadata = None
            if execution_mode == "docker":
                cve_metadata = load_cve_metadata(cve_id, input_json_path)
            
            # 进度显示
            remaining = len(cve_dirs) - idx - resumed
            logger.info(
                f"[{total + 1}/{len(cve_dirs) - resumed}] 处理 {cve_id}"
                f"（进度: {idx + 1}/{len(cve_dirs)}，已跳过续跑: {resumed}）"
            )

            # 执行修复
            total += 1
            result = repair_single_cve(
                cve_id,
                buggy_code_file,
                test_cmds,
                cve_metadata,
                models_dict,
                config,
                output_dir,
            )
            
            # ── 断点更新 ────────────────────────────────────────────────────
            if result["success"]:
                success += 1
                checkpoint.mark(cve_id, "success")
            else:
                failed += 1
                checkpoint.mark(cve_id, "failed")
                
        except LLMAuthError as e:
            logger.error(f"LLM 认证失败: {e}")
            logger.error("请检查 API Key 或使用 --llm-provider mock")
            checkpoint.mark(cve_id, "error", str(e))
            break
        except Exception as e:
            logger.error(f"处理 {cve_id} 时出错: {e}")
            failed += 1
            checkpoint.mark(cve_id, "error", str(e))
            continue

    # ── 汇总结果 ──────────────────────────────────────────────────────────────
    all_stats = checkpoint.stats()
    total_done_ever = sum(all_stats.values())

    logger.info("\n" + "="*60)
    logger.info("批量修复完成")
    logger.info(f"本次运行: 处理={total}, 成功={success}, 失败={failed}, 无文件跳过={skipped}, 续跑跳过={resumed}")
    logger.info(f"累计全局: {all_stats}")
    logger.info(f"结果保存在: {output_dir}")
    logger.info("="*60)
    
    # 生成汇总报告
    summary_file = output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump({
            # 本次运行统计
            "this_run": {
                "total_processed": total,
                "success": success,
                "failed": failed,
                "skipped_no_files": skipped,
                "skipped_resumed": resumed,
            },
            # 累计统计（含历史运行）
            "cumulative": all_stats,
            "output_dir": str(output_dir),
            "checkpoint_file": str(checkpoint.path),
            "finished_at": datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"汇总报告: {summary_file}")
    logger.info(f"断点文件: {checkpoint.path}（下次运行可自动续跑）")


if __name__ == "__main__":
    main()
