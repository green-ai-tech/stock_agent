"""
RAG 检索器
提供 Agent 可调用的知识库检索工具
"""
from typing import List, Annotated
from utils.setting import settings
from utils.logger import logger
from .vector_store import get_collection
from .embeddings import get_embedding_model


def retrieve(query: str, top_k: int = None) -> List[dict]:
    """
    从知识库中检索相关文档
    Args:
        query: 查询文本
        top_k: 返回数量，默认使用配置值
    Returns:
        [{"content": "...", "score": 0.85, "source": "xxx.pdf"}, ...]
    """
    top_k = top_k or settings.retriever_k
    collection = get_collection()

    if collection.count() == 0:
        return []

    # 向量化查询
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)

    # 检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    hits = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # cosine distance → similarity (越小越相似)
        similarity = 1 - dist
        if similarity >= settings.retriever_score_threshold:
            hits.append({
                "content": doc,
                "score": round(similarity, 4),
                "source": meta.get("source", "unknown"),
            })

    logger.info(f"检索完成: query='{query[:30]}...', 返回 {len(hits)} 条结果")
    return hits


def format_results(hits: List[dict]) -> str:
    """将检索结果格式化为文本，供 LLM 使用"""
    if not hits:
        return "未找到相关知识。"

    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"[片段 {i}] (来源: {hit['source']}, 相关度: {hit['score']})\n{hit['content']}"
        )
    return "\n\n".join(parts)


def search_knowledge_base(
    query: Annotated[str, "搜索查询，描述你想了解的知识内容"],
    top_k: Annotated[int, "返回结果数量，默认3"] = 3,
) -> str:
    """
    搜索投资知识库，获取相关专业知识。
    当用户询问投资概念、技术指标、财务分析方法、K线形态等专业知识时使用此工具。
    """
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return "知识库中未找到相关信息。请尝试更换关键词，或确认知识库中已导入相关文档。"

    return format_results(hits)
