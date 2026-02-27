"""多视角盲视生成器

API 调用点：
===========
在 generate_single_view() 方法中会调用：
  self.llm.generate(prompt, ...)  
  ↓
  models/llm_interface.py 的 generate() 方法
  ↓
  _generate_openai() 实际发送 HTTP 请求到 OpenAI
"""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class MultiViewGenerator:
    """多视角 RCA（根因分析）生成器"""

    VIEW_TYPES = ["data_flow", "control_flow", "api_semantic"]

    def __init__(self, llm_interface, view_types: Optional[List[str]] = None):
        """
        初始化多视角生成器
        
        Args:
            llm_interface: LLM 接口实例
            view_types: 可选的视角列表（用于只跑部分视角，如 ["data_flow"]）
        """
        self.llm = llm_interface
        if view_types is not None:
            # instance-level override (do not mutate the class constant globally)
            self.VIEW_TYPES = list(view_types)
        self.prompts = self._load_prompts()

    def _default_prompt_dir(self) -> Path:
        """获取默认 prompt 目录"""
        return Path(__file__).resolve().parent.parent / "prompts" / "offline" / "blind"

    def _read_prompt(self, path: Path) -> Optional[str]:
        """读取 prompt 模板文件"""
        try:
            if not path.exists():
                return None
            content = path.read_text(encoding="utf-8")
            return content.strip() or None
        except Exception as e:
            logger.warning(f"无法读取 prompt 文件 {path}: {e}")
            return None

    def _load_prompts(self) -> Dict[str, str]:
        """加载各视角的 prompt 模板"""
        prompt_dir = self._default_prompt_dir()
        prompts: Dict[str, str] = {}
        
        file_map = {
            "data_flow": prompt_dir / "data_flow.txt",
            "control_flow": prompt_dir / "control_flow.txt",
            "api_semantic": prompt_dir / "api_semantic.txt",
        }

        for view_type, path in file_map.items():
            content = self._read_prompt(path)
            if content is not None:
                prompts[view_type] = content

        # 内置 fallback prompts
        if len(prompts) != len(self.VIEW_TYPES):
            logger.warning("部分盲视 prompt 缺失，使用内置模板")
            prompts.setdefault(
                "data_flow",
                """分析以下有漏洞的代码，识别数据流问题。
关注点：
1. 变量定义和使用
2. 数据依赖
3. 错误的数据传播
4. 潜在的数据流漏洞

有漏洞的代码：
{buggy_code}

请提供详细的数据流分析：""",
            )
            prompts.setdefault(
                "control_flow",
                """分析以下有漏洞的代码，识别控制流问题。
关注点：
1. 条件分支
2. 循环结构
3. 控制依赖
4. 不可达代码或错误分支

有漏洞的代码：
{buggy_code}

请提供详细的控制流分析：""",
            )
            prompts.setdefault(
                "api_semantic",
                """分析以下有漏洞的代码，识别 API/语义问题。
关注点：
1. API 误用
2. 函数调用模式
3. 语义违规
4. 接口契约

有漏洞的代码：
{buggy_code}

请提供详细的 API 和语义分析：""",
            )

        return prompts

    def generate_blind_views(
        self, buggy_code: str, patch: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成所有视角的盲视
        
        Args:
            buggy_code: 有漏洞的源代码
            patch: 可选的真实补丁（盲视生成时通常不使用）
            
        Returns:
            视角类型到生成的 RCA 文本的映射
        """
        logger.info("为代码样本生成盲视")
        
        views = {}
        for view_type in self.VIEW_TYPES:
            views[view_type] = self.generate_single_view(buggy_code, view_type, patch)

        return views

    def generate_single_view(
        self, buggy_code: str, view_type: str, patch: Optional[str] = None
    ) -> str:
        """
        生成单个视角
        
        ★★★ 这里会调用 LLM API ★★★
        
        Args:
            buggy_code: 有漏洞的源代码
            view_type: 视角类型
            patch: 可选的真实补丁
            
        Returns:
            生成的 RCA 文本
        """
        if view_type not in self.prompts:
            raise ValueError(f"未知的视角类型: {view_type}")

        logger.debug(f"生成 {view_type} 视角")

        # 构造 prompt
        prompt = self.prompts[view_type].format(
            buggy_code=buggy_code,
            patch=patch or "",
            view_type=view_type,
        )

        # ★ 调用 LLM 生成盲视
        # 如果 API Key 无效，这里会抛出 LLMAuthError
        response = self.llm.generate(prompt, max_tokens=1024, temperature=0.3)
        return response

    def batch_generate(
        self, samples: List[Dict[str, str]], view_type: str
    ) -> List[str]:
        """
        批量生成视角
        
        Args:
            samples: 样本列表，每个包含 'buggy_code' 和可选的 'patch'
            view_type: 视角类型
            
        Returns:
            生成的视角文本列表
        """
        logger.info(f"为 {len(samples)} 个样本批量生成 {view_type} 视角")
        
        views = []
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"进度: {i}/{len(samples)}")

            view = self.generate_single_view(
                sample["buggy_code"], view_type, sample.get("patch")
            )
            views.append(view)

        return views

    def customize_prompt(self, view_type: str, new_prompt: str):
        """
        自定义视角的 prompt 模板
        
        Args:
            view_type: 视角类型
            new_prompt: 新的 prompt 模板
        """
        if view_type not in self.VIEW_TYPES:
            raise ValueError(f"未知的视角类型: {view_type}")

        self.prompts[view_type] = new_prompt
        logger.info(f"已更新 {view_type} 的 prompt")
