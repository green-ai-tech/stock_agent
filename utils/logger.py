"""
日志工具模块
===================================
项目统一日志管理工具，基于 loguru 实现
提供控制台彩色输出 + 文件持久化 + 自动切割 + 自动清理功能

功能说明：
    1. 自动读取配置文件中的日志级别、路径、大小、保留时间
    2. 支持多模块统一调用，全局单例日志对象
    3. 日志格式：时间 | 级别 | 模块 | 消息
    4. 自动按大小切割，按时间保留，防止日志膨胀

使用示例：
    from utils.logger import logger

    logger.info("服务启动成功")
    logger.warning("配置文件未找到，使用默认值")
    logger.error("数据库连接失败")
    logger.exception("捕获异常信息")
"""

import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger
from .paths import PROJECT_ROOT,LOG_PATH
from .setting import settings


#==================日志配置====================
def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
) -> None:
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    rotation = rotation or settings.log_rotation
    retention = retention or settings.log_retention

    logger.remove()

    # 控制台日志
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 文件日志
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

def get_logger(name: str):
    return logger.bind(name=name)

# 自动初始化
setup_logging()

__all__ = ["logger", "get_logger"]

# 测试
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("调试信息")
    logger.info("提示信息")
    logger.success("成功信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    logger.critical("紧急信息")
