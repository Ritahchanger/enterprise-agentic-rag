"""
Shared application logger using loguru.
Import `logger` anywhere: `from src.utils.logger import logger`
"""

import sys
from loguru import logger

from src.config.settings import settings

# Remove default handler and configure a clean, leveled console sink.
logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

__all__ = ["logger"]
