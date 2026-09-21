# llm_market_sim/utils/logger.py
import os
import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_file: Optional[str] = None, 
                 log_level: int = logging.INFO,
                 console_output: bool = True) -> None:
    """
    设置日志记录。
    
    Args:
        log_file: 日志文件路径，如果为None则只输出到控制台
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        console_output: 是否同时输出到控制台
    """
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除已存在的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台输出
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件输出
    if log_file:
        # 确保日志文件目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info(f"日志系统初始化完成，级别: {logging.getLevelName(log_level)}")
    if log_file:
        logging.info(f"日志文件: {log_file}") 