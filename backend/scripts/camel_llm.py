"""
OASIS/camel-ai 模型创建（模拟子进程共用）

模型路由：
- 模拟轮次中每个 Agent 的动作调用量最大，优先使用加速模型 LLM_BOOST_*；
- 任一 LLM_BOOST_* 为空时回退到对应的 LLM_*（全部为空即完全使用主模型）。

camel-ai 的 OpenAIModel 读取 OPENAI_API_KEY / OPENAI_API_BASE_URL；这里显式传入
api_key/url（避免并行模拟中两个平台互相覆盖全局环境变量），同时仍把它们写入环境
变量，保证 camel 内部其他读取点也来自 .env。

每次模型调用共享 LLM_MAX_CONCURRENCY 并发上限，并对 429/529 过载错误做指数退避重试。
"""

import asyncio
import functools
import os
import random
from typing import Any, Dict, Optional

from camel.models import ModelFactory
from camel.types import ModelPlatformType

from app.utils.openai_chat_compat import (
    LLM_RETRY_INITIAL_DELAY,
    LLM_RETRY_MAX_ATTEMPTS,
    LLM_RETRY_MAX_DELAY,
    get_max_concurrency,
    is_retryable_llm_error,
)


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def resolve_llm_settings(use_boost: bool = True) -> Dict[str, Any]:
    """返回 {api_key, base_url, model, is_boost}；加速配置缺失或为空时回退主配置"""
    main = {
        "api_key": _env("LLM_API_KEY"),
        "base_url": _env("LLM_BASE_URL"),
        "model": _env("LLM_MODEL_NAME"),
    }
    boost = {
        "api_key": _env("LLM_BOOST_API_KEY"),
        "base_url": _env("LLM_BOOST_BASE_URL"),
        "model": _env("LLM_BOOST_MODEL_NAME"),
    }
    is_boost = use_boost and any(boost.values())
    if not is_boost:
        return {**main, "is_boost": False}
    return {key: boost[key] or main[key] for key in main} | {"is_boost": True}


_llm_semaphore: Optional[asyncio.Semaphore] = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(get_max_concurrency())
    return _llm_semaphore


def _wrap_arun(model):
    """为 model.arun 加上共享并发上限和 429/529 退避重试"""
    original_arun = model.arun

    @functools.wraps(original_arun)
    async def arun(*args, **kwargs):
        delay = LLM_RETRY_INITIAL_DELAY
        for attempt in range(1, LLM_RETRY_MAX_ATTEMPTS + 1):
            try:
                async with _get_llm_semaphore():
                    return await original_arun(*args, **kwargs)
            except Exception as error:
                if attempt >= LLM_RETRY_MAX_ATTEMPTS or not is_retryable_llm_error(error):
                    raise
                wait = min(delay, LLM_RETRY_MAX_DELAY) * (0.5 + random.random())
                print(
                    f"LLM限流/过载 (status={getattr(error, 'status_code', None)}), "
                    f"第 {attempt}/{LLM_RETRY_MAX_ATTEMPTS} 次, {wait:.1f}秒后重试"
                )
                await asyncio.sleep(wait)
                delay *= 2

    model.arun = arun
    return model


def create_camel_model(
    config: Optional[Dict[str, Any]] = None,
    use_boost: bool = True,
    label: str = "",
):
    """创建 Agent 动作使用的 camel 模型"""
    settings = resolve_llm_settings(use_boost)
    model_name = settings["model"] or (config or {}).get("llm_model", "gpt-4o-mini")

    if not settings["api_key"]:
        raise ValueError("缺少 API Key 配置，请在项目根目录 .env 文件中设置 LLM_API_KEY")

    # 设置 camel-ai 读取的环境变量
    os.environ["OPENAI_API_KEY"] = settings["api_key"]
    if settings["base_url"]:
        os.environ["OPENAI_API_BASE_URL"] = settings["base_url"]

    config_label = "[加速LLM]" if settings["is_boost"] else "[通用LLM]"
    base_url = settings["base_url"]
    print(
        f"{label}{config_label} model={model_name}, "
        f"base_url={base_url[:40] if base_url else '默认'}..., "
        f"max_concurrency={get_max_concurrency()}"
    )

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_name,
        api_key=settings["api_key"],
        url=base_url or None,
    )
    return _wrap_arun(model)
