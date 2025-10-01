from loguru import logger
import sys



def set_log_level(level: str="INFO", disable: bool=False):
    logger.remove(0)
    logger.add(
        sys.stderr,
        colorize=True,
        format="<blue>|MorphZ|</blue>{level}| <level>{message}</level>",
        level="INFO"  # Or your desired default level
    )
    if disable:
        logger.disable("morphZ")

    return logger

logger = set_log_level()
