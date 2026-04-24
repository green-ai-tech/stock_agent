"""
RAG (Retrieval-Augmented Generation) 模块
"""
from .retriever import search_knowledge_base
from .vector_store import get_collection, add_documents, list_documents, delete_document

__all__ = [
    "search_knowledge_base",
    "get_collection",
    "add_documents",
    "list_documents",
    "delete_document",
]
