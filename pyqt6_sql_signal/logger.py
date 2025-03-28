import logging
import os
import ctypes
import atexit
from logging import FileHandler

# Windows隐藏文件属性常量
FILE_ATTRIBUTE_HIDDEN = 0x02
FILE_ATTRIBUTE_READONLY = 0x01

def setup_logging(log_file):
    # 获取exe或脚本所在目录
    # if getattr(sys, 'frozen', False):
    #     base_dir = os.path.dirname(sys.executable)
    # else:
    #     base_dir = os.path.dirname(os.path.abspath(__file__))

    # 日志文件路径
    # log_file = ".app_log.log"
    # log_file = os.path.join(base_dir, ".app_log.log")

    # 初始化时设置文件可写
    set_file_writable(log_file)

    # 配置日志系统
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建文件处理器
    file_handler = FileHandler(
        log_file,
        mode='a',
        encoding='utf-8'
    )

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # 添加到日志系统
    logger.addHandler(file_handler)

    # 注册退出处理函数
    atexit.register(set_file_readonly, log_file)

    # 设置文件隐藏属性（仅Windows）
    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetFileAttributesW(log_file, FILE_ATTRIBUTE_READONLY)
        except Exception as e:
            logging.warning(f"无法设置隐藏属性: {str(e)}")

    return logger

def set_file_readonly(file_path):
    import win32api, win32con
    """设置文件为只读"""
    if os.path.exists(file_path):
        try:
            if os.name == 'nt':
                win32api.SetFileAttributes(file_path,win32con.FILE_ATTRIBUTE_READONLY)
                # Windows系统
                # ctypes.windll.kernel32.SetFileAttributesW(file_path, FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN)
            else:
                # Linux/macOS系统
                os.chmod(file_path, 0o444)
        except Exception as e:
            logging.error(f"设置只读属性失败: {str(e)}")

def set_file_writable(file_path):
    """设置文件可写"""
    if os.path.exists(file_path):
        try:
            if os.name == 'nt':
                # Windows系统
                ctypes.windll.kernel32.SetFileAttributesW(file_path, FILE_ATTRIBUTE_HIDDEN)
            else:
                # Linux/macOS系统
                os.chmod(file_path, 0o666)
        except Exception as e:
            logging.error(f"设置可写属性失败: {str(e)}")
    else:
        # 文件不存在时创建空文件并设置初始权限
        open(file_path, 'a').close()
        if os.name == 'nt':
            ctypes.windll.kernel32.SetFileAttributesW(file_path, FILE_ATTRIBUTE_HIDDEN)
        else:
            os.chmod(file_path, 0o644)


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
