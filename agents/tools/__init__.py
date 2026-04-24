from .time_tools import get_current_date,get_current_time

BASIC_TOOLS = [
    get_current_time,
    get_current_date
]

__all__ = [
    "get_current_time",
    "get_current_date",
    #工具分组
    "BASIC_TOOLS"
]