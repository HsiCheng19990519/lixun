# devmate/logging_utils.py
"""
日志工具模块。

目标：
- 在程序启动时，统一配置 logging；
- 把日志输出到控制台和文件（logs/devmate.log）；
- 使用滚动日志文件，避免单个日志文件过大。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import Settings


def setup_logging(settings: Settings) -> logging.Logger:
    """
    根据 Settings 初始化日志系统，并返回根 logger。

    调用约定：
    - 在 CLI 入口（devmate/cli.py）中，程序开始时调用一次即可。
    """
    logger = logging.getLogger()

    # 如果已经配置过 handler，就不重复配置，避免日志重复打印
    if logger.handlers:
        return logger

    # 1. 设置日志等级
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # 2. 定义日志格式：时间 + 等级 + 模块名 + 消息
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 3. 控制台输出 handler
    # Use stderr for console logs to avoid polluting stdout (important for stdio protocols like MCP).
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 4. 文件输出 handler（带滚动）
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 单个日志文件最大 5 MB
        backupCount=3,             # 最多保留 3 个历史文件
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(
        "日志系统初始化完成: level=%s, file=%s",
        settings.log_level,
        log_path,
    )

    return logger
