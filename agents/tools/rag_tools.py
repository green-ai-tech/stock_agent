"""
RAG 检索工具（供 Agent 调用）
"""
from typing import Annotated
from langchain_core.tools import tool
from rag.retriever import retrieve, format_results
from utils.logger import logger


@tool
def search_knowledge_base(
    query: Annotated[str, "搜索查询，描述你想了解的投资知识内容"],
    top_k: Annotated[int, "返回结果数量，默认3"] = 3,
) -> str:
    """
    搜索投资知识库，获取相关专业知识。
    当用户询问投资概念、技术指标、财务分析方法、K线形态、基本面分析等专业知识时，使用此工具从知识库中检索参考内容。

    使用场景示例：
    - "什么是MACD指标？"
    - "头肩顶形态怎么判断？"
    - "ROE是什么意思？"
    - "如何分析一只股票的基本面？"
    """
    try:
        hits = retrieve(query, top_k=top_k)
        if not hits:
            return "知识库中未找到相关信息。请尝试更换搜索关键词。"

        result = format_results(hits)
        logger.info(f"RAG检索: query='{query[:30]}...', 命中 {len(hits)} 条")
        return result
    except Exception as e:
        logger.error(f"知识库检索出错: {e}")
        return f"知识库检索出错: {str(e)}"
