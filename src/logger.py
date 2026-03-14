import sys

import logging
from datetime import datetime


class CenterAlignedFormatter(logging.Formatter):
    """Custom formatter that center-aligns the log level - NO COLORS"""
    
    def __init__(self, fmt=None, datefmt=None, level_width=8):
        super().__init__(fmt, datefmt)
        self.level_width = level_width
    
    def format(self, record):
        record.levelname = record.levelname.center(self.level_width)
        return super().format(record)


def create_jupyter_logger(name=__name__, level=logging.INFO, level_width=8):
    """Create a logger with center-aligned log levels for JupyterLab - NO COLORS"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Use CenterAlignedFormatter instead of ColorCenterAlignedFormatter
    formatter = CenterAlignedFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level_width=level_width
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger
