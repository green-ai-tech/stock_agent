"""
Chroma 向量数据库管理
- get_vector_store: 获取/初始化 Chroma 实例
- add_documents: 添加文档到向量库
- list_documents: 列出已入库的文档来源
- delete_document: 删除指定来源的所有文档
"""
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from utils.setting import settings
from utils.paths import PROJECT_ROOT
from utils.logger import logger
from .embeddings import get_embedding_model


CHROMA_DIR = str(Path(PROJECT_ROOT) / "data" / "chroma")
COLLECTION_NAME = "knowledge_base"


def _get_chroma_client() -> chromadb.PersistentClient:
    """获取 Chroma 持久化客户端"""
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    """获取或创建知识库集合"""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(documents: List[dict]) -> int:
    """
    将分块后的文档添加到向量库
    Args:
        documents: [{"content": "...", "metadata": {"source": "...", ...}}, ...]
    Returns:
        添加的文档数量
    """
    if not documents:
        return 0

    collection = get_collection()
    embedding_model = get_embedding_model()

    # 批量向量化
    texts = [doc["content"] for doc in documents]
    embeddings = embedding_model.embed_documents(texts)

    # 生成唯一 ID
    existing_count = collection.count()
    ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

    # 入库
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[doc["metadata"] for doc in documents],
    )

    logger.info(f"向量库添加 {len(documents)} 条文档, 当前总量: {collection.count()}")
    return len(documents)


def list_documents() -> List[dict]:
    """
    列出已入库的文档来源（去重）
    Returns:
        [{"source": "xxx.pdf", "chunks": 5}, ...]
    """
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    # 获取所有元数据
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    # 按 source 分组计数
    source_counts = {}
    for meta in metadatas:
        source = meta.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    return [
        {"source": source, "chunks": count}
        for source, count in source_counts.items()
    ]


def delete_document(source_name: str) -> bool:
    """
    删除指定来源的所有文档
    Args:
        source_name: 文件名
    Returns:
        是否成功
    """
    collection = get_collection()
    try:
        collection.delete(where={"source": source_name})
        logger.info(f"已删除文档: {source_name}")
        return True
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        return False


def clear_all():
    """清空整个知识库"""
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("知识库已清空")
    except Exception:
        pass
