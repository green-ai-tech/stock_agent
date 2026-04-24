"""
文本分块器
基于 langchain-text-splitters
"""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.setting import settings
from utils.logger import logger


def split_text(text: str, source_name: str = "unknown") -> List[dict]:
    """
    将文本分块，返回带元数据的文档列表
    Args:
        text: 原始文本
        source_name: 来源文件名
    Returns:
        [{"content": "...", "metadata": {"source": "...", "chunk_index": 0}}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    chunks = splitter.split_text(text)
    logger.info(f"文本分块完成: {source_name}, 共 {len(chunks)} 块")

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "content": chunk,
            "metadata": {
                "source": source_name,
                "chunk_index": i,
            }
        })
    return documents
