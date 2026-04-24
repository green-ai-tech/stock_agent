"""
Embedding 模型封装
使用 Ollama 本地 embedding 模型
"""
from typing import List
from langchain_ollama import OllamaEmbeddings
from utils.setting import settings
from utils.logger import logger


def get_embedding_model() -> OllamaEmbeddings:
    """获取 Ollama embedding 模型实例"""
    model_name = settings.embedding_model  # e.g. "qwen3-embedding:4b"
    base_url = settings.embedding_base_url

    logger.info(f"初始化 Embedding 模型: {model_name} @ {base_url}")
    return OllamaEmbeddings(
        model=model_name,
        base_url=base_url,
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """批量文本向量化"""
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """单条查询向量化"""
    model = get_embedding_model()
    return model.embed_query(text)
