from .time_tools import get_current_date,get_current_time
from .rag_tools import search_knowledge_base

BASIC_TOOLS = [
    get_current_time,
    get_current_date,
    search_knowledge_base,
]

__all__ = [
    "get_current_time",
    "get_current_date",
    "search_knowledge_base",
    #工具分组
    "BASIC_TOOLS"
]