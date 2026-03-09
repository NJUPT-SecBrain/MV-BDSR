"""统一的 LLM 接口封装（支持 OpenAI/Anthropic/Gemini/Mock）

API 调用说明：
=============
OpenAI 接口通过官方 SDK 的 Chat Completions 发起 HTTP 请求：
  - 请求格式：POST https://api.openai.com/v1/chat/completions
  - 请求头：Authorization: Bearer sk-你的API密钥
  - 请求体：{"model":"gpt-4o-mini","messages":[...],"temperature":...}
  - 响应：{"choices":[{"message":{"content":"..."}}]}

Google Gemini 接口通过官方 google-genai SDK 调用：
  - 使用 GOOGLE_API_KEY 或 config.yaml 中的 api_key 进行认证
  - 支持 generate_content() 和多轮对话 chats.create()

API Key 来源：
  1. 优先使用初始化时传入的 api_key 参数
  2. OpenAI: 从环境变量 OPENAI_API_KEY 读取
  3. Anthropic: 从环境变量 ANTHROPIC_API_KEY 读取
  4. Gemini: 从环境变量 GOOGLE_API_KEY 读取
"""

from typing import List, Optional
from loguru import logger
import os
import re
import json
import time
import random


class LLMError(RuntimeError):
    """LLM 调用失败的基础异常"""


class LLMAuthError(LLMError):
    """认证失败（API Key 无效/过期）"""


class LLMInterface:
    """统一的 LLM 接口（OpenAI, Anthropic, Mock）"""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        """
        初始化 LLM 接口
        
        Args:
            provider: 提供商 (openai, anthropic, mock)
            model_name: 模型名称
            api_key: API 密钥（None 时从环境变量读取）
            base_url: API 端点 URL（可选）
                - OpenAI 默认: https://api.openai.com/v1
                - Azure OpenAI: https://{resource}.openai.azure.com
                - 自定义代理/兼容服务: 你的自定义 URL
            temperature: 采样温度
            max_tokens: 最大生成 token 数
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

        # 从环境变量获取 API Key（如果没有显式传入）
        if api_key is None:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")

        self.api_key = api_key
        self.client = self._initialize_client()

    def _is_auth_error(self, exc: Exception, msg: str) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return status in (401, 403) or "invalid_api_key" in msg or "Incorrect API key" in msg

    def _is_retryable_error(self, exc: Exception, msg: str) -> bool:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        m = (msg or "").lower()
        # Cloudflare/proxy HTML error pages, common in custom gateways
        if "<!doctype html" in m or "cloudflare" in m or "error 524" in m or " 524" in m:
            return True
        # Typical transient HTTP codes
        if status in (408, 409, 425, 429, 500, 502, 503, 504, 524):
            return True
        # Network/timeout strings
        transient = [
            "timeout",
            "timed out",
            "connection error",
            "connection reset",
            "connection aborted",
            "remote protocol error",
            "server disconnected",
            "eof",
            "temporarily unavailable",
        ]
        return any(t in m for t in transient)

    def _initialize_client(self):
        """初始化 provider 对应的客户端"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                # ★ 这里创建 OpenAI 客户端，后续所有请求都通过它发送
                # 如果指定了 base_url，使用自定义端点；否则使用默认端点
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                    logger.info(f"使用自定义 OpenAI 端点: {self.base_url}")
                return OpenAI(**client_kwargs)
            except ImportError:
                logger.error("openai 包未安装。请运行: pip install openai")
                return None

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                # Anthropic 也支持自定义 base_url
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                    logger.info(f"使用自定义 Anthropic 端点: {self.base_url}")
                return Anthropic(**client_kwargs)
            except ImportError:
                logger.error("anthropic 包未安装。请运行: pip install anthropic")
                return None

        elif self.provider == "gemini":
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                logger.info(f"使用 Google Gemini 原生 SDK，模型: {self.model_name}")
                return client
            except ImportError:
                logger.error("google-genai 包未安装。请运行: pip install google-genai")
                return None

        elif self.provider == "mock":
            # Mock 模式：用于离线测试，不发起真实 API 请求
            return None

        else:
            raise ValueError(f"未知的 provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        生成文本（核心调用入口）
        
        ★★★ 这是整个框架调用 LLM 的统一入口 ★★★
        所有地方（盲视生成、蒸馏、质量评估）都会调用这个方法
        
        Args:
            prompt: 输入 prompt
            max_tokens: 最大 token 数（None 时使用默认值）
            temperature: 温度（None 时使用默认值）
            stop: 停止序列
            
        Returns:
            生成的文本
            
        Raises:
            LLMAuthError: API Key 无效/过期
            LLMError: 其他 LLM 调用错误
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        if self.provider == "mock":
            return self._generate_mock(prompt)

        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.5"))

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt, max_tokens, temperature, stop)
                elif self.provider == "anthropic":
                    return self._generate_anthropic(prompt, max_tokens, temperature, stop)
                elif self.provider == "gemini":
                    return self._generate_gemini(prompt, max_tokens, temperature, stop)
                else:
                    raise ValueError(f"未知的 provider: {self.provider}")
            except Exception as e:
                msg = str(e)
                last_exc = e

                if self._is_auth_error(e, msg):
                    raise LLMAuthError(f"API Key 无效或过期: {msg}") from e

                if attempt < max_retries and self._is_retryable_error(e, msg):
                    sleep_s = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.6)
                    logger.warning(
                        f"LLM 调用失败（可重试，第 {attempt}/{max_retries} 次）: {msg[:200]}...；"
                        f"{sleep_s:.1f}s 后重试"
                    )
                    time.sleep(sleep_s)
                    continue

                raise LLMError(f"LLM 生成失败: {msg}") from e

        # Should not reach here
        raise LLMError(f"LLM 生成失败: {str(last_exc) if last_exc else 'unknown'}") from last_exc

    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
    ) -> str:
        """
        使用 OpenAI API 生成
        
        ★ 实际 HTTP 请求在这里发出 ★
        调用链：self.client.chat.completions.create() 
                → OpenAI SDK 内部
                → POST https://api.openai.com/v1/chat/completions
                → Headers: {"Authorization": "Bearer sk-..."}
                → Body: {"model":"...","messages":[...],"temperature":...}
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return response.choices[0].message.content

    def _generate_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
    ) -> str:
        """使用 Anthropic API 生成"""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=stop,
        )
        return response.content[0].text

    def _generate_gemini(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
    ) -> str:
        """
        使用 Google Gemini 原生 SDK 生成
        
        ★ 实际 HTTP 请求在这里发出 ★
        调用链：self.client.models.generate_content()
                → google-genai SDK 内部
                → POST https://generativelanguage.googleapis.com/v1beta/models/...
                → Headers: {"x-goog-api-key": "AIza..."}
        """
        from google import genai
        from google.genai import types

        config_kwargs = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            config_kwargs["stop_sequences"] = stop

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text

    def _extract_view_type(self, prompt: str) -> str:
        """从 prompt 中提取视角类型（用于 mock 模式）"""
        m = re.search(r'【View Type】\s*\n\s*([a-zA-Z_]+)\s*', prompt)
        if m:
            return m.group(1).strip()
        m = re.search(r'View Type:\s*([a-zA-Z_]+)', prompt)
        if m:
            return m.group(1).strip()
        m = re.search(r'"view_type"\s*:\s*"([^"]+)"', prompt)
        if m:
            return m.group(1).strip()
        return "unknown"

    def _generate_mock(self, prompt: str) -> str:
        """
        Mock 模式：生成符合格式的假数据（用于离线测试）
        不发起任何网络请求，返回结构正确的测试数据
        """
        view_type = self._extract_view_type(prompt)

        # 判断请求类型
        wants_quality = ("missing_aspects" in prompt) and ("confidence" in prompt) and ("accurate" in prompt)
        wants_distill = ('"view_type"' in prompt) and ("fix_hints" in prompt) and ("constraints" in prompt)

        if wants_quality:
            # 质量评估请求
            return json.dumps(
                {
                    "accurate": True,
                    "confidence": 0.7,
                    "reasoning": "Mock 模式：默认认为盲视蒸馏输出可用（仅用于管线自测）。",
                    "missing_aspects": [],
                },
                ensure_ascii=False,
            )

        if wants_distill:
            # 蒸馏请求
            return json.dumps(
                {
                    "view_type": view_type,
                    "root_cause": "Mock 模式：可能缺少边界检查/空指针检查或输入验证导致漏洞。",
                    "signals": ["lack_of_validation", "unsafe_api_or_boundaries"],
                    "entities": {"vars": [], "funcs": [], "apis": []},
                    "constraints": ["添加必要的输入校验/边界检查"],
                    "fix_hints": ["在关键分支前增加判空/范围检查", "确保长度/索引不越界"],
                    "patch_mechanism": "Mock 模式：通过增加检查/调整条件修复。",
                },
                ensure_ascii=False,
            )

        # 盲视生成请求
        return (
            f"root_cause: Mock 模式（{view_type}）：疑似缺少校验/边界检查导致不安全数据传播或错误分支。\n"
            f"evidence: 见 Buggy Code 中与输入处理/索引/指针相关的语句。\n"
            f"constraints: 修复需保持原有语义，增加必要检查。\n"
            f"fix_hints: 1) 判空/范围检查 2) 修正条件分支 3) 处理返回值/错误码。\n"
        )

    def chat(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        多轮对话接口
        
        Args:
            messages: 消息列表 [{"role":"user","content":"..."},...]
            max_tokens: 最大 token 数
            temperature: 温度
            
        Returns:
            助手回复
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.5"))

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return response.choices[0].message.content
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                    )
                    return response.content[0].text
                elif self.provider == "gemini":
                    from google import genai
                    from google.genai import types
                    # 转换消息格式：OpenAI 的 "assistant" → Gemini 的 "model"
                    gemini_history = []
                    for msg in messages[:-1]:
                        role = "model" if msg["role"] == "assistant" else "user"
                        gemini_history.append(
                            types.Content(role=role, parts=[types.Part(text=msg["content"])])
                        )
                    last_content = messages[-1]["content"]
                    chat_session = self.client.chats.create(
                        model=self.model_name,
                        history=gemini_history,
                        config=types.GenerateContentConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                        ),
                    )
                    response = chat_session.send_message(last_content)
                    return response.text
                else:
                    raise NotImplementedError(f"{self.provider} 的 chat 接口未实现")
            except Exception as e:
                msg = str(e)
                last_exc = e
                if self._is_auth_error(e, msg):
                    raise LLMAuthError(f"API Key 无效或过期: {msg}") from e
                if attempt < max_retries and self._is_retryable_error(e, msg):
                    sleep_s = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.6)
                    logger.warning(
                        f"Chat 调用失败（可重试，第 {attempt}/{max_retries} 次）: {msg[:200]}...；"
                        f"{sleep_s:.1f}s 后重试"
                    )
                    time.sleep(sleep_s)
                    continue
                logger.error(f"Chat 失败: {msg}")
                raise LLMError(f"Chat 失败: {msg}") from e

        raise LLMError(f"Chat 失败: {str(last_exc) if last_exc else 'unknown'}") from last_exc

    def count_tokens(self, text: str) -> int:
        """
        估算 token 数量（简单估算，生产环境应使用 tiktoken）
        
        Args:
            text: 输入文本
            
        Returns:
            估算的 token 数
        """
        return len(text) // 4

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        批量生成
        
        Args:
            prompts: prompt 列表
            max_tokens: 最大 token 数
            temperature: 温度
            
        Returns:
            生成文本列表
        """
        responses = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"生成 {i+1}/{len(prompts)}")
            response = self.generate(prompt, max_tokens, temperature)
            responses.append(response)
        return responses
