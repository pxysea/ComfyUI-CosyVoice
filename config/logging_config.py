# logging_config.py

import logging

def setup_logging(log_file="app.log",lvl = logging.INFO):
    """Configure logging for the entire application."""
    logging.basicConfig(
        level=lvl,  # 设置全局日志级别
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),  # 输出到文件
        ],
    )