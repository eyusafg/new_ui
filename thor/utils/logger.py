import logging
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

def setup_logging(log_file, logger_name):
    """配置异步文件日志系统（生产消费模式）"""
    # 创建线程安全队列（建议根据业务量设置合理容量）
    log_queue = Queue(maxsize=10000)  # 最多缓冲10000条日志

    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器（消费者）
    file_handler = logging.FileHandler(
        filename=log_file,
        mode='a',
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 队列监听器（异步写入）
    listener = QueueListener(
        log_queue,
        file_handler,
        respect_handler_level=True
    )
    listener.start()

    # 配置根日志记录器（生产者）
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # 清理旧处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 添加非阻塞队列处理器
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # # 确保程序退出时刷新日志
    # def exit_handler():
    #     listener.stop()
    
    # # 注册退出处理
    # import atexit
    # atexit.register(exit_handler)

    return logger,listener



if __name__ == "__main__":
    # 初始化日志系统
    logger = setup_logging()

    logger.info("程序启动")
    # 测试写入
    try:
        logger.debug("调试信息")
        logger.info("常规操作")
    except Exception as e:
        logger.error("发生异常", exc_info=True)
    logger.info("程序正常退出")
