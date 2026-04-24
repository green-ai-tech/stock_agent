"""
模型的封装：
    1. 通用的模型封装：get_chat_model
    2. 简易模型封装：get_model_by_preset
"""

from typing import Optional,Dict,Any
from langchain.chat_models import init_chat_model,BaseChatModel #类型约束
from langchain_ollama import ChatOllama
from utils import get_logger,settings

logger = get_logger(__name__)

def get_chat_model(
    *,   # 强制后面参数都使用关键字参数
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: Optional[bool] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    获取配置好的聊天模型实例
    Args:
        model_name: 模型名称，默认使用配置中的 openai_model
        temperature: 温度参数 (0.0-2.0)，控制输出随机性，默认使用配置值
        max_tokens: 最大生成 token 数，默认使用配置值
        streaming: 是否启用流式输出，默认使用配置值
        **kwargs: 其他传递给模型的参数
    Returns:
        配置好的 ChatModel 实例
    """
    # 使用配置中的设置值
    model_name  = model_name or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature
    streaming   = streaming if streaming is not None else settings.llm_streaming
    max_tokens  = max_tokens if max_tokens is not None else settings.llm_max_tokens
    
    # 构建模型配置
    model_config: Dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "streaming": streaming,
        "max_tokens": max_tokens,
        "base_url": settings.llm_base_url,
    }
    
    # 合并其他的参数
    model_config.update(kwargs)
    
    logger.info(
        f"创建聊天模型: {model_name} "
        f"(temperature={temperature}, streaming={streaming}, max_tokens={max_tokens})"
    )
    # 创建 Agent 实例
    try:
        if isinstance(model_name, str) and model_name.startswith("ollama:"):
            ollama_model_name = model_name.split("ollama:", 1)[1]
            model = ChatOllama(
                model=ollama_model_name,
                temperature=temperature,
                base_url=settings.llm_base_url,
                num_predict=max_tokens,
                **kwargs,
            )
        else:
            model = init_chat_model(**model_config)

        logger.success(f"模型创建成功: {model_name}")
        return model
    except Exception as e:
        logger.error(f"❌ 模型创建失败: {e}")
        raise  # 转移异常，交给其他调用者继续处理异常

# 预定义的模型配置
from utils.setting import settings

PRESET_CONFIGS = {
    "default": {
        "model_name": settings.llm_model,   # 动态读取
        "temperature": 0.7,
        "description": "默认模型",
    },
    "fast": {
        "model_name": settings.llm_model,   # 或单独指定一个更快的模型
        "temperature": 0.7,
        "description": "快速模型",
    },
    "precise": {
        "model_name": settings.llm_model,
        "temperature": 0.3,
        "description": "精确模型",
    },
    "creative": {
        "model_name": settings.llm_model,
        "temperature": 1.0,
        "description": "创意模型",
    },
    "structure": {
        "model_name": settings.llm_model,
        "temperature": 0.0,
        "description": "结构化输出模型",
    },
}

# 提供一个函数，使用预制模版创建模型
def get_model_by_preset(preset: str = "default", **kwargs: Any) -> BaseChatModel:
    """
    获取预设模型
    Args:
        preset: 预设名称，可选值: default, fast, precise, creative，structure
        **kwargs: 覆盖预设的参数
    Returns:
        配置好的 ChatModel 实例
    Raises:
        ValueError: 如果预设名称不存在
    """
    if preset not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"未知的预设: {preset}. 可用预设: {available}")
    
    config = PRESET_CONFIGS[preset].copy()
    config.pop("description", None)  # 移除描述字段
    config.update(kwargs)  # 用户参数覆盖预设
    
    logger.info(f"📋 使用预设模型配置: {preset}")
    return get_chat_model(**config)