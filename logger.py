import logging
import sys
from typing import Optional

def get_logger(name: str = "stock_scanner", level: Optional[int] = None) -> logging.Logger:
    """
    获取配置好的日志器
    
    Args:
        name: 日志器名称
        level: 日志级别，默认为INFO
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    
    # 如果日志器已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    if level is None:
        level = logging.INFO
    logger.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志器
    logger.addHandler(console_handler)
    
    # 防止日志重复输出
    logger.propagate = False
    
    return logger

def setup_file_logger(name: str = "stock_scanner", 
                     filename: str = "stock_scanner.log",
                     level: Optional[int] = None) -> logging.Logger:
    """
    设置文件日志器
    
    Args:
        name: 日志器名称
        filename: 日志文件名
        level: 日志级别
        
    Returns:
        配置好的日志器
    """
    logger = get_logger(name, level)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 添加文件处理器
    logger.addHandler(file_handler)
    
    return logger 