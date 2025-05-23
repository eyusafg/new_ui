import os
import time
import numpy as np
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
import serial
import serial.tools.list_ports as list_ports
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QRunnable, QThreadPool,QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QCheckBox, QRadioButton, QMenu,
                            QMessageBox,QApplication, QSizePolicy, QLineEdit, QDialog,QInputDialog)
import onnxruntime as rt
import yaml
import cv2
from utils.detect_circle import detect_circle
from utils.irregular_im_de_simple import get_kde_x, get_kde_y, get_y, find_max_min_y_for_x
from utils.get_max_contour import get_max_contour
from utils.get_contour_corner import get_contour_corner
from utils.logger import setup_logging
from utils.calculate_distance import calculate_distance
from utils.alingning_offset import AlignmentOptimizer
import struct
import shutil
from PyCameraList.camera_device import list_video_devices
import json 
from datetime import datetime, timedelta
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import uuid
import psutil
import hmac
import wmi 
from pathlib import Path
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# from scipy.ndimage import uniform_filter1d
from threading import Thread
import pandas as pd 
from utils.serial_listener import SerialListener
from utils.serial_listener_crc import SerialListener_crc
import logging
import atexit


VAIL_MODEL = False
IS_ENCRYPT = False

SECRET_KEY = hashlib.pbkdf2_hmac('sha256', b'03106666', b'salt', 100000)
IV = os.urandom(16)

class LicenseManager:
    def __init__(self, app_data_dir):
        self.license_path = os.path.join(app_data_dir, 'license.dat')
        self.install_time_path = os.path.join(app_data_dir, '.install')
        self.app_data_dir = app_data_dir
        
        os.makedirs(app_data_dir, exist_ok=True)
        
        self.master_key = SECRET_KEY

    def _encrypt(self, data):
        """AES加密"""
        cipher = AES.new(SECRET_KEY, AES.MODE_CBC, IV)
        return IV + cipher.encrypt(pad(data.encode(), AES.block_size))

    def _decrypt(self, data):
        """AES解密"""
        iv = data[:16]
        cipher = AES.new(SECRET_KEY, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(data[16:]), AES.block_size).decode()
        
    # def get_client_id(self):
    #     """生成可靠的硬件唯一标识"""
    #     def get_mac():
    #         """获取物理MAC地址（兼容多平台）"""
    #         try:
    #             # 方法1：通过uuid库获取
    #             mac = uuid.getnode()
    #             if (mac >> 40) & 0xff == 0x00:
    #                 # 方法2：通过netifaces获取真实MAC
    #                 import netifaces
    #                 for interface in netifaces.interfaces():
    #                     addr = netifaces.ifaddresses(interface).get(netifaces.AF_LINK)
    #                     if addr and addr.get('addr'):
    #                         return addr['addr'].replace(':', '')
    #             return ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
    #         except:
    #             return "unknown-mac"

    #     def get_disk():
    #         """获取系统盘序列号（兼容多平台）"""
    #         try:
    #             import psutil
    #             # Windows取C盘，Linux/macOS取根分区
    #             target_disk = 'C:' if sys.platform == 'win32' else '/'
    #             for part in psutil.disk_partitions():
    #                 if part.mountpoint == target_disk:
    #                     disk = part.device
    #                     if sys.platform == 'win32':
    #                         # Windows需要转换路径格式
    #                         disk = disk.replace('\\', '')
    #                     return psutil.disk_usage(disk).serial
    #             return "unknown-disk"
    #         except:
    #             return "unknown-disk"

    #     try:
    #         # 组合多个硬件特征
    #         components = [
    #             get_mac(),
    #             get_disk(),
    #             # 添加更多硬件信息（可选）
    #             str(psutil.cpu_count()),
    #             str(psutil.virtual_memory().total)
    #         ]
    #         return hashlib.sha256('|'.join(components).encode()).hexdigest()[:16]
    #     except Exception as e:
    #         print(f"硬件标识生成失败: {str(e)}")
    #         return "default-client-id"
        
    def get_client_id(self):
        """Windows 11专用硬件标识生成"""
        def get_physical_mac():
            """获取物理网卡MAC地址"""
            try:
                c = wmi.WMI()
                for interface in c.Win32_NetworkAdapterConfiguration(IPEnabled=True):
                    if not interface.MACAddress: continue
                    # 过滤虚拟网卡（根据名称判断）
                    if "Virtual" not in interface.Description:
                        return interface.MACAddress.replace(':', '')
            except Exception as e:
                print(f"MAC获取失败: {str(e)}")
            return "unknown-mac"

        def get_disk_serial():
            """获取系统盘序列号"""
            try:
                c = wmi.WMI()
                for disk in c.Win32_DiskDrive():
                    if "Partition0" in disk.DeviceID:  # 系统盘通常是第一个磁盘
                        return disk.SerialNumber.strip()
            except Exception as e:
                print(f"磁盘序列号获取失败: {str(e)}")
            return "unknown-disk"

        def get_baseboard_id():
            """获取主板ID"""
            try:
                c = wmi.WMI()
                for board in c.Win32_BaseBoard():
                    return board.Product.strip() + board.SerialNumber.strip()
            except:
                return "unknown-board"

        try:
            # 组合多种硬件信息
            components = [
                get_physical_mac(),
                # get_disk_serial(),
                get_baseboard_id(),
                str(psutil.cpu_count(logical=False)),  # 物理核心数
                hex(uuid.getnode())[2:].zfill(12)      # 备用网络标识
            ]
            print(f"硬件组件信息: {components}")  # 调试用
            
            # 生成哈希标识
            combined = "|".join(components).encode('utf-16le')
            return hashlib.sha256(combined).hexdigest()[:20].upper()
        except Exception as e:
            print(f"硬件标识生成异常: {str(e)}")
            return "DEFAULT-ID"
    
    def _validate_license(self, key):
        # """测试用验证逻辑（接受任意THOR-开头的激活码）"""
        # # 正式环境需删除此测试逻辑
        # if key.startswith("THOR-TEST-"):
        #     return True
        
        """离线验证许可证"""
        try:
            # 解析密钥结构：THOR-{client_id}-{days}D-{signature}
            parts = key.split('-')
            if len(parts) != 4 or parts[0] != "THOR":
                print(f"license key format error: {key}")
                return False
                
            client_id = parts[1]
            days = int(parts[2][:-1])
            received_sig = parts[3]
            
            # 验证本地客户端ID
            if client_id != self.get_client_id():
                return False
                
            # 重新计算签名
            data = f"{client_id}|{days}".encode()
            expected_sig = hmac.new(self.master_key, data, hashlib.sha256).hexdigest()[:16]
            
            return hmac.compare_digest(received_sig.upper(), expected_sig.upper())
        except:
            return False

    def check_license(self):
        """检查许可证状态"""
        # 正式许可证检查
        if os.path.exists(self.license_path):
            try:
                with open(self.license_path, 'rb') as f:
                    encrypted = f.read()
                    license_info = json.loads(self._decrypt(encrypted))
                    expire_date = datetime.fromisoformat(license_info['expire'])
                    return datetime.now() < expire_date
            except:
                QMessageBox.warning(None, '错误', 
                                  '许可路径不存在')
        
        # 试用期检查
        install_time = self._get_install_time()
        if not install_time:
            self._save_install_time()
            return True
            
        trial_end = install_time + timedelta(days=1)
        remaining_days = (trial_end - datetime.now()).days
        
        if remaining_days > 0:
            if remaining_days <= 1:
                QMessageBox.warning(None, '试用提醒', 
                                  f'剩余试用天数：{remaining_days}天')
            return True
        return False

    def activate_license(self, key):
        """激活许可证"""
        if not self._validate_license(key):
            return False
        
        # 解析有效期
        days = int(key.split('-')[2][:-1])
        
        # 存储许可证信息
        license_info = {
            'expire': (datetime.now() + timedelta(days=days)).isoformat(),
            'client_id': self.get_client_id(),
            'signature': key.split('-')
        }
        
        try:
            encrypted = self._encrypt(json.dumps(license_info))
            with open(self.license_path, 'wb') as f:
                f.write(encrypted)
            return True
        except:
            return False
        
    def _get_install_time(self):
        try:
            with open(self.install_time_path, 'rb') as f:
                encrypted = f.read()
                return datetime.fromisoformat(self._decrypt(encrypted))
        except:
            return None

    def _save_install_time(self):
        encrypted = self._encrypt(datetime.now().isoformat())
        with open(self.install_time_path, 'wb') as f:
            f.write(encrypted)
        if sys.platform == 'win32':
            os.system(f'attrib +h "{self.install_time_path}"')

class LicenseDialog(QDialog):
    def __init__(self, client_id):
        super().__init__()
        self.setWindowTitle('许可证激活')
        self.setFixedSize(500, 200)
        
        layout = QVBoxLayout()
        
        # 显示客户端ID
        lbl_info = QLabel(f"您的设备ID：{client_id}\n请输入从供应商获取的激活码：")
        layout.addWidget(lbl_info)
        
        lbl_info.setContextMenuPolicy(Qt.CustomContextMenu)
        lbl_info.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(lbl_info)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText('格式：THOR-XXXX-XXD-XXXX')
        layout.addWidget(self.key_input)
        
        self.btn_activate = QPushButton('激活')
        self.btn_activate.clicked.connect(lambda: self._activate())
        layout.addWidget(self.btn_activate)
        
        self.setLayout(layout)
        self.client_id = client_id

    def _show_context_menu(self, pos):
        context_menu = QMenu(self)
        copy_action = context_menu.addAction("复制设备ID")
        action = context_menu.exec_(self.mapToGlobal(pos))
        if action == copy_action:
            self._copy_client_id()

    def _copy_client_id(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.client_id)

    def _activate(self):
        key = self.key_input.text().strip()
        if not key:
            QMessageBox.warning(self, '错误', '请输入有效的激活码')
            return
            
        manager = LicenseManager(os.path.join(os.path.expanduser("~"), ".thor_data"))
        if manager.activate_license(key):
            QMessageBox.information(self, "成功", "产品已激活！")
            self.accept()
        else:
            QMessageBox.critical(self, "失败", "激活码无效或设备不匹配")


class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

class SafeRunnable(QRunnable):
    def __init__(self, parent=None):
        super().__init__()
        self.signals = WorkerSignals()
        if parent:
            self.signals.setParent(parent) # 绑定父对象生命周期
        self.mutex = QMutex()
        self._active  = True
    def is_active(self):
        with QMutexLocker(self.mutex):
            return self._active
    def cancel(self):
        with QMutexLocker(self.mutex):
            self._active = False

class InitWorker(SafeRunnable):
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        # self.signals = WorkerSignals()
        # self.setAutoDelete(True)

    def run(self):
        try:
            if not self.is_active(): return

            # 加载配置
            config = self.load_config()
            model_path = config['model_path']
            if 'thor_segm0103' in model_path:
                self.is_640 = True
            # 并行加载模型
            sess = self.load_model(config['model_path'])
            segm_sess = self.load_model(config['segm_model_path'], delay=False)

            # 返回结果
            result = {
                'config': config,
                'sess': sess,
                'input_name': sess.get_inputs()[0].name, 
                'output_names': [out.name for out in sess.get_outputs()],
                'segm_sess': segm_sess,
                'segm_input_name': segm_sess.get_inputs()[0].name,  
                'segm_output_names': [out.name for out in segm_sess.get_outputs()]               
            }
            self.signals.finished.emit(result)
        except Exception as e:
            if self.is_active():    
                self.signals.error.emit(f"配置文件初始化失败: {str(e)}")

    def load_config(self):
        with open(self.config_path, "r",encoding='utf-8') as f:
            config = yaml.safe_load(f)

            if 'cached_camera_index' not in config:
                # 初始化缓存字段
                config.setdefault('cached_camera_index', None)
                config.setdefault('cached_serial_port', None)

            return config
    
    def load_model(self, model_path, delay=False):
        if delay: return None # 延迟加载分割模型
        
        so = rt.SessionOptions()
        so.intra_op_num_threads = 2
        # return rt.InferenceSession(model_path, so)
        return rt.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    
class CameraWorker(SafeRunnable):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        # self.signals = WorkerSignals()
        # self.setAutoDelete(True)

    def run(self):
        try:
            if not self.is_active(): return
            camera = self.init_camera()
            self.signals.finished.emit(camera)
        except Exception as e:
            if self.is_active():    
                self.signals.error.emit(f"相机初始化失败: {str(e)}")

    def init_camera(self):
        # 优先使用缓存索引
        if self.config['cached_camera_index'] is not None:
            camera = cv2.VideoCapture(self.config['cached_camera_index'])
            if camera.isOpened():
                self.set_camera_props(camera)
                return camera
        
        # 缓存失效重新检测
        devices = list_video_devices()
        valid_devices = [(idx, name) for idx, name in devices 
                        if 'obs' not in name.lower() and 'webcam' not in name.lower()]
        print(valid_devices)
        if len(valid_devices) == 1:
            index, _ = valid_devices[0]
            # print('camera index:', index)
            self.config['cached_camera_index'] = index
            camera = cv2.VideoCapture(index)
            self.set_camera_props(camera)
            return camera
        raise RuntimeError("找不到唯一相机设备" if len(valid_devices)==0 else "检测到多个相机")
    
    def set_camera_props(self, camera):
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2880)
        camera.set(cv2.CAP_PROP_FPS, 20)
        
class SerialWorker(SafeRunnable):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        # self.signals = WorkerSignals()
        # self.setAutoDelete(True)

    def run(self):
        try:
            if not self.is_active(): return
            ser = self.init_serial()
            self.signals.finished.emit(ser)
        except Exception as e:
            if self.is_active(): 
                print(f"串口初始化失败: {str(e)}")   
                self.signals.error.emit(f"串口初始化失败: {str(e)}")
    def init_serial(self):
        # 优先使用缓存端口
        if self.config['cached_serial_port']:
            try:
                ser = serial.Serial(self.config['cached_serial_port'], 115200, timeout=0.005)
                if ser.is_open:
                    return ser
            except: pass
        
        ports = list_ports.comports()
        if len(ports) == 1:
            port = ports[0].device
            self.config['cached_serial_port'] = port
            return serial.Serial(port, 115200, timeout=0.005)
        raise RuntimeError('找不到唯一串口设备' if len(ports) == 0 else '检测到多个串口设备')

def infer_frame(image, sess, input_name, output_names):
    img = np.asarray([image]).astype(np.float32)
    detection = sess.run(output_names, {input_name: img})
    return detection

def infer_segm(image, sess, input_name, output_names, is_640):
    if is_640:
        frame_se = cv2.resize(image, (640, 640))
    frame_se = cv2.resize(image, (384, 384))
    frame_se = np.ascontiguousarray(frame_se[:, :, ::-1].transpose(2, 0, 1))
    img = np.asarray([frame_se]).astype(np.float32)
    detection = sess.run(output_names, {input_name: img})[0]
    return detection

def calculate_average_excluding_extremes(offset_list):
    if len(offset_list) <= 2:
        raise ValueError("列表中至少需要三个元素才能去除最大值和最小值并计算平均值")
    
    # 转换为 numpy 数组
    offset_array = np.array(offset_list)
    
    # 去除最大值和最小值
    offset_array_excluding_extremes = np.delete(offset_array, [np.argmax(offset_array), np.argmin(offset_array)])
    
    # 计算剩余元素的平均值
    average_value = np.mean(offset_array_excluding_extremes)
    
    return average_value

def filter_outliers(arr):
    """通过IQR方法生成正常值的布尔掩码"""
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (arr >= lower) & (arr <= upper)


def check_condition(pts, head_start_x_border, head_end_x_border, x_diff, loger):
    if pts.ndim != 2:
        loger.error('条件不满足: pts.ndim != 2')
        return False
    if not np.all(pts > 0):
        loger.error('条件不满足: np.all(pts > 0)')
        return False
    if not np.all(pts[:, 0] > head_start_x_border):
        loger.error('条件不满足: np.all(pts[:, 0] > head_start_x_border)')
        return False
    if not np.all(pts[:, 0] < head_end_x_border):
        loger.error(f'条件不满足: np.all(pts[:, 0] < head_end_x_border)')
        return False
    if abs(pts[0][0] - pts[1][0]) >= x_diff:
        loger.error(f'条件不满足: {abs(pts[0][0] - pts[1][0])} < {x_diff}')
        return False
    return True

def check_tail_condition(pts, pixel_d, end_p, x_diff, logger):
    if pts.ndim != 2:
        logger.error('条件不满足: pts.ndim != 2')
        return False
    if not np.all(pts > 0):
        logger.error('条件不满足: np.all(pts > 0)')
        return False
    if not np.all(pixel_d < pts[:, 0]):
        logger.error('条件不满足: np.all(pixel_d < pts[:, 0])')
        return False
    if not np.all(pts[:, 0] < end_p):
        logger.error('条件不满足: np.all(pts[:, 0] < end_p)')
        return False
    if abs(pts[0][0] - pts[1][0]) >= x_diff:
        logger.error(f'条件不满足: 布料翘起或者有缺口 实际: {abs(pts[0][0] - pts[1][0])}')
        logger.error('条件不满足: 布料翘起或者有缺口 阈值120')
        return False
    return True

def crc8_rohc(data):
    """CRC-8/ROHC算法实现"""
    crc = 0xFF  # 初始值
    polynomial = 0x07  # 多项式
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
            crc &= 0xFF
    crc = struct.pack('B', crc)
    return crc
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化许可证管理器
        if IS_ENCRYPT:
            app_data_dir = os.path.join(os.path.expanduser("~"), ".thor_data")
            self.license_mgr = LicenseManager(app_data_dir)
            
            if not self.license_mgr.check_license():
                self._show_activation_dialog()
                sys.exit()  # 如果取消激活则退出

        # self.log_file = 'thor_log.log'
        info_log_dir = 'thor_log//info'
        os.makedirs(info_log_dir, exist_ok=True)
        error_log_dir = 'thor_log//error'
        os.makedirs(error_log_dir, exist_ok=True)

        self.log_file_name = datetime.now().strftime('%Y_%m_%d.log')
        self.info_log_file = os.path.join(info_log_dir, self.log_file_name)
        self.error_log_file = os.path.join(error_log_dir, self.log_file_name)
        self.loger, self.info_listener = setup_logging(self.info_log_file, 'info_logger')
        self.error_loger, self.error_listener = setup_logging(self.error_log_file, 'error_logger')

        # 设置过滤器
        self.loger.addFilter(lambda record: record.levelno < logging.ERROR)
        self.error_loger.addFilter(lambda record: record.levelno >= logging.ERROR)
        # 注册退出处理
        atexit.register(self.exit_handler)
    
        self.config_path = 'profiles//config.yaml'
        self.backup_dir = Path('profiles') / 'backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_config()
        self.delete_old_backups()

        self.camera = None
        self.ser = None
        self.camera_ready = False
        self.serial_ready = False
        self.segm_infer = True
        self.tail_count = 0
        self.total_ = 0
        self.t_ = 0
        self.head_de_is_ok = True
        self.cloth_w = 0.0
        self.cloth_d = 0.0
        self.is_640 = False 
        
        # 初始化线程池
        self.thread_pool = QThreadPool.globalInstance()

        # 并行启动初始化任务
        self.start_paraller_init()
        

        # 初始化ui
        # self.button_clicked = False
        if VAIL_MODEL:
            self.inference_count = 0
            self.ser_infer_count = 0
            self.excel_data = []

        self.init_ui()

        self.head_frame = None
        # self.circle_head_y = None
        # self.head_x_coords_ = None
        self.head_x_coords = None

    def exit_handler(self):
        # 确保程序退出时刷新日志
        if self.info_listener is not None:
            self.info_listener.stop()
        if self.error_listener is not None:
            self.error_listener.stop()

    def backup_config(self):
        # 获取当前日期
        current_date = datetime.now().strftime('%Y-%m-%d')
        # 构建备份文件路径
        backup_path = self.backup_dir / f'config_{current_date}.yaml'
        
        # 备份文件
        shutil.copy2(self.config_path, backup_path)
        self.loger.info(f'Config file backed up to {backup_path}')
        print(f'Config file backed up to {backup_path}')

    def delete_old_backups(self):
        # 获取当前日期
        current_date = datetime.now()
        
        # 遍历备份文件
        for backup_file in self.backup_dir.glob('config_*.yaml'):
            # 提取备份文件的日期
            file_date_str = backup_file.stem.split('_')[1].split('.')[0]
            file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
            
            # 如果备份文件超过7天，则删除
            if (current_date - file_date).days > 7:
                backup_file.unlink()
                self.loger.info(f'Deleted old backup: {backup_file}')
                print(f'Deleted old backup: {backup_file}')

    def _show_activation_dialog(self):
        client_id = self.license_mgr.get_client_id()
        print('client_id', client_id)
        dialog = LicenseDialog(client_id)
        if dialog.exec_() != QDialog.Accepted:
            QMessageBox.critical(self, "错误", "必须激活才能使用")
            sys.exit()

    def init_ui(self):
        self.setWindowTitle('Thor Visin System')
        self.resize(400, 200)
        
        self.create_widgets()
        self.setup_layout()
        self.connect_signals()
        self.disable_all_buttons()

    def create_widgets(self):
        """创建界面控件"""
        # 图像显示
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.image_label.setMinimumSize(320, 240)
        
        self.label_head = QLabel(self)
        self.label_head.setAlignment(Qt.AlignCenter)
        self.label_head.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.label_head.setMinimumSize(320, 240)
        self.label_head.setVisible(False)

        self.label_tail = QLabel(self)
        self.label_tail.setAlignment(Qt.AlignCenter)
        self.label_tail.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.label_tail.setMinimumSize(320, 240)
        self.label_tail.setVisible(False)

        if VAIL_MODEL:
            self.label_align = QLabel(self)
            self.label_align.setAlignment(Qt.AlignCenter)
            self.label_align.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.label_align.setMinimumSize(320, 240)
            self.label_align.setVisible(False)
            self.nametext = QLineEdit(self)

        # 按钮
        self.save_button = QPushButton("保存图片")
        self.p_detect_button = QPushButton("基点检测")
        self.factor_detect_button = QPushButton("转换系数检测")
        self.inference_button = QPushButton("单独推理")
        # self.ser_inference_button = QPushButton("通信推理")
        # self.ser_inference_button.clicked.connect(self.toggle_button_color)
        self.advanced_options_button = QPushButton("高级选项")

        # 单选/复选框
        self.front_radio = QRadioButton('前部推理')
        self.rear_radio = QRadioButton('尾部推理')
        self.front_radio.setChecked(True)
        
        self.checkbox_infer = QCheckBox("开启推理界面")
        # self.checkbox_segm = QCheckBox("使用分割")
        
        self.checkbox = QCheckBox("开启更新画面")
        self.checkbox_test = QCheckBox("测试模式")
        self.reload_buttons = [
            QRadioButton(text) for text in [
                "重新加载配置文件",
                "重新加载相机",
                "重新加载串口"
            ]
        ]

        # 添加高级选项
        self.advanced_options_button.clicked.connect(self.show_advanced_options)

    def setup_layout(self):
        """布局设置"""
        # 图像布局
        self.hbox_labels = QHBoxLayout()
        self.hbox_labels.addWidget(self.image_label)
        self.hbox_labels.addWidget(self.label_head)
        self.hbox_labels.addWidget(self.label_tail)
        if VAIL_MODEL:
            self.hbox_labels.addWidget(self.label_align)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(self.hbox_labels)
        # main_layout.addWidget(self.checkbox_segm)
        main_layout.addWidget(self.checkbox_infer)
        main_layout.addWidget(self.checkbox)

        # 添加重新加载按钮
        for btn in self.reload_buttons:
            main_layout.addWidget(btn)
        
        main_layout.addWidget(self.checkbox_test)

        # 添加单选按钮组
        main_layout.addWidget(self.front_radio)
        main_layout.addWidget(self.rear_radio)
        
        if VAIL_MODEL:
            main_layout.addWidget(self.nametext)

        # 添加功能按钮
        button_group = [
            self.save_button,
            self.inference_button,
            self.p_detect_button,
            self.factor_detect_button,
            # self.ser_inference_button
        ]
        for btn in button_group:
            main_layout.addWidget(btn)

        # 默认设置为不可见
        self.inference_button.setVisible(False)
        self.front_radio.setVisible(False)
        self.rear_radio.setVisible(False)
        self.save_button.setVisible(False)
        self.p_detect_button.setVisible(False)
        self.factor_detect_button.setVisible(False)
        self.checkbox_test.setVisible(False)
        self.reload_buttons[0].setVisible(False)
        self.reload_buttons[1].setVisible(False)
        self.reload_buttons[2].setVisible(False)

        main_layout.addStretch(1)  # 添加伸缩项确保按钮在底部
        main_layout.addWidget(self.advanced_options_button)

        # 设置中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


    def show_advanced_options(self):
        """显示高级选项"""
        password, ok = QInputDialog.getText(self, '输入密码', '请输入密码:', QLineEdit.Password)
        if ok:
            correct_password = "03106666"  
            if password == correct_password:
                # 显示高级选项
                # self.inference_button.setVisible(True)
                # self.front_radio.setVisible(True)
                # self.rear_radio.setVisible(True)
                self.save_button.setVisible(True)
                self.p_detect_button.setVisible(True)
                self.factor_detect_button.setVisible(True)
                self.checkbox_test.setVisible(True)
                self.reload_buttons[0].setVisible(True)
                self.reload_buttons[1].setVisible(True)
                self.reload_buttons[2].setVisible(True)
                if VAIL_MODEL:
                    self.nametext.setVisible(True)
            else:
                QMessageBox.warning(self, '警告', '密码错误，请重试。')


    def connect_signals(self):
        """连接信号槽"""
        # 按钮点击
        self.save_button.clicked.connect(lambda: self.save_image())
        self.p_detect_button.clicked.connect(lambda: self.start_corner_p_detection())
        self.factor_detect_button.clicked.connect(lambda: self.calib_pixel_to_real())
        self.inference_button.clicked.connect(lambda: self.toggle_inference())
        # self.ser_inference_button.clicked.connect(self.ser_infer)
        
        # 复选框
        self.checkbox_infer.stateChanged.connect(lambda state: self.switch_mode(state))
        # self.checkbox_segm.stateChanged.connect(self.is_segm)
        self.checkbox.stateChanged.connect(lambda state: self.toggle_timer(state))
        self.checkbox_test.stateChanged.connect(lambda state: self.switch_test_demo(state))
        
        # 重新加载按钮
        self.reload_buttons[0].clicked.connect(lambda: self.reload_config())
        self.reload_buttons[1].clicked.connect(lambda: self.reload_camera())
        self.reload_buttons[2].clicked.connect(lambda: self.reload_serial())

    def start_paraller_init(self):
        self.config_worker = InitWorker(self.config_path, self)
        self.config_worker.signals.finished.connect(self.on_config_loaded)
        self.config_worker.signals.error.connect(self.show_error)
        self.thread_pool.start(self.config_worker)

    def on_config_loaded(self, result):

        self.config = result['config']
        self.sess = result['sess']
        self.input_name = result['input_name']
        self.output_names = result['output_names']

        self.segm_sess = result['segm_sess']
        self.segm_input_name = result['segm_input_name']
        self.segm_output_names = result['segm_output_names']

        self.base_point = self.config['base_point'] 
        self.origin_x = self.base_point[0]
        self.origin_y = self.base_point[1]   
        # self.coordinate = [self.origin_x, self.origin_y]
    
        self.factor = self.config['factor']

        self.head_x_offset = self.config['head_roi']['x_offset']
        self.head_w = self.config['head_roi']['w']
        self.head_h = self.config['head_roi']['h']
        self.head_y_offset = self.config['head_roi']['y_offset']

        self.end_x = self.origin_x + self.head_x_offset
        self.start_x = self.end_x - self.head_w
        self.start_y = self.origin_y + self.head_y_offset
        self.end_y = self.start_y + self.head_h

        self.pad = (800-self.head_h) // 2
        self.x_border = self.pad + self.head_h
        self.roi_rotate_base_p_x = self.origin_y - self.start_y + self.pad
        self.roi_rotate_base_p_y = self.head_w - (self.origin_x - self.start_x)
        self.factor_w_offset = self.head_w / 1300
        self.head_start_x_border = self.pad + 10
        self.head_end_x_border = self.x_border - 10

        # 增加夹爪边缘位置， 用于尾部识别错误发送该偏移
        # self.head_end_x_border_ = self.end_y - self.roi_rotate_base_p_x
        self.head_end_x_border_ = self.pad - self.roi_rotate_base_p_x

        # tail
        self.tail_x_offset = self.config['tail_roi']['x_offset']
        self.tail_w = self.config['tail_roi']['w']
        self.tail_h = self.config['tail_roi']['h']
        self.tail_x_border = self.tail_h - 50
        self.tail_y_offset = self.config['tail_roi']['y_offset']

        self.tail_end_x = self.origin_x + self.tail_x_offset
        # start_x = end_x - 1800
        self.tail_start_x = self.tail_end_x - self.tail_w
        self.tail_end_y = self.origin_y + self.tail_y_offset
        self.tail_start_y = self.tail_end_y - self.tail_h
        self.roi_tail_rotate_base_p_x = self.origin_y - self.tail_start_y
        self.roi_tail_rotate_base_p_y = self.tail_w - (self.origin_x - self.tail_start_x)   

        self.tail_factor_h_offset = self.tail_h / 800 
        self.tail_factor_w_offset = self.tail_w / 1300
        
        self.x_diff = self.config['x_diff_th']
        # self.tail_offset = self.config['tail_offset']
        self.y_offset = self.config['y_offset']
        # 串口信号信息
        # 前部
        head_cmd_id_x_offset = 0x010C  # 发送前部坐标
        self.sof = b'\xA5'
        self.head_cmd_id_bytes = struct.pack('<H', head_cmd_id_x_offset)
        # self.cmd_id_bytes = struct.pack('<H', cmd_id_x_offset)

        # 发送移动指令
        cmd_id_head_align = 0x0101
        head_cmd_id_bytes_align = struct.pack('<H', cmd_id_head_align)
        # self.cmd_id_bytes_f = struct.pack('<H', cmd_id_head_align)
        # self.data_f = b'\x00'
        head_move_front_byte_data = self.sof + head_cmd_id_bytes_align + bytes([0])
        data_frame_f_crc = crc8_rohc(head_move_front_byte_data)
        self.head_move_data_frame = head_move_front_byte_data + data_frame_f_crc
        # head_move_data_frame_crc = crc8_rohc(head_move_data_frame)
        # self.head_move_data_frame = head_move_data_frame + head_move_data_frame_crc

        # 收到布料宽度响应
        cmd_id_w_cloth = 0x0103  # 发送前部坐标
        self.cmd_id_bytes_w_cloth = struct.pack('<H', cmd_id_w_cloth)
        # self.cloth_w_data = b'\x00'
        w_front_part = self.sof + self.cmd_id_bytes_w_cloth + bytes([0]) 
        w_front_part_crc = crc8_rohc(w_front_part)
        self.w_full_frame_data = w_front_part + w_front_part_crc
        print('w_full_frame_data', self.w_full_frame_data)
        # w_full_frame_data_crc = crc8_rohc(w_full_frame_data)
        # print('w_full_frame_data_crc', w_full_frame_data_crc)
        # self.w_full_frame_data = w_full_frame_data + w_full_frame_data_crc

        # 主动访问宽度数据
        cmd_id_w_cloth_take = 0x0102  # 发送前部坐标
        cmd_id_bytes_w_cloth_take = struct.pack('<H', cmd_id_w_cloth_take)
        w_front_part_take = self.sof + cmd_id_bytes_w_cloth_take + bytes([0]) 
        w_front_part_take_crc = crc8_rohc(w_front_part_take)
        self.w_full_frame_data_take = w_front_part_take + w_front_part_take_crc
        # w_full_frame_data_take_crc = crc8_rohc(w_full_frame_data_take)
        # self.w_full_frame_data_take = w_full_frame_data_take + w_full_frame_data_take_crc


        # 发送错误指令
        data_lenght = b'\x04'
        cmd_id_error = 0x0001
        self.cmd_id_error_bytes = struct.pack('<H', cmd_id_error)
        front_part_error_cmd = self.sof + self.cmd_id_error_bytes + data_lenght
        front_part_error_crc_bytes = crc8_rohc(front_part_error_cmd)

        # 前部错误命令
        self.head_error = b'\x00\x01\x00\x00'
        head_error_cmd = front_part_error_cmd + front_part_error_crc_bytes + self.head_error
        head_full_frame_error_crc_bytes = crc8_rohc(head_error_cmd)
        self.head_error_cmd = head_error_cmd + head_full_frame_error_crc_bytes

        # 尾部错误命令
        self.tail_error = b'\x04\x01\x00\x00'
        tail_error_cmd = front_part_error_cmd + front_part_error_crc_bytes + self.tail_error
        tail_full_frame_error_crc_bytes = crc8_rohc(tail_error_cmd)
        self.tail_error_cmd = tail_error_cmd + tail_full_frame_error_crc_bytes

        # 宽度错误命令
        w_data_lenght = b'\x04'
        self.w_error = b'\x02\x01\x00\x00'
        # j_w_error = b'\x01\x02\x00\x00'
        front_part_w_error = self.sof + self.cmd_id_error_bytes + w_data_lenght
        front_part_w_crc_bytes = crc8_rohc(front_part_w_error)
        # front_part_w_crc_bytes = struct.pack('B', front_part_w_crc)
        full_frame_w_error = front_part_w_error + front_part_w_crc_bytes + self.w_error
        full_frame_sod_crc_bytes = crc8_rohc(full_frame_w_error)
        # full_frame_sod_crc_bytes = struct.pack('B', full_frame_crc)
        self.w_error_cmd = full_frame_w_error + full_frame_sod_crc_bytes


        # 尾部
        cmd_id_tail_offset = 0x010D # 发送尾部坐标
        self.cmd_id_bytes_ = struct.pack('<H', cmd_id_tail_offset)

        # 尾部发送移动指令
        hex_data_f = f'A5 05 01'
        # self.byte_data_f = bytes.fromhex(hex_data_f.replace(' ', ''))
        tail_move_front_byte_data = bytes.fromhex(hex_data_f.replace(' ', ''))
        # tail_move_front_byte_crc = crc8_rohc(tail_move_front_byte_data)
        self.tail_move_byte_data = tail_move_front_byte_data
        # tail_move_byte_data_crc = crc8_rohc(tail_full_frame_move_data)
        # self.tail_move_byte_data = tail_full_frame_move_data + tail_move_byte_data_crc

        # 硬件初始化
        if not self.reload_buttons[0].isChecked():
            self.start_hardware_init()

        # 创建文件夹
        self.create_folders()
    
    
    def start_hardware_init(self):
        
        self.reload_camera()
        self.reload_serial()

    def reload_config(self):
        self.config_worker = InitWorker(self.config_path, self)
        self.config_worker.signals.finished.connect(self.on_config_loaded)
        self.config_worker.signals.error.connect(self.show_error)
        self.thread_pool.start(self.config_worker)

    def reload_camera(self):
        if self.camera is None:
            self.camera_worker = CameraWorker(self.config, self)  
            self.camera_worker.signals.finished.connect(self.on_camera_ready)
            self.camera_worker.signals.error.connect(self.show_error)
            self.thread_pool.start(self.camera_worker)

    def reload_serial(self):
        if self.ser is None:
            self.serial_worker = SerialWorker(self.config, self)
            self.serial_worker.signals.finished.connect(self.on_serial_ready)
            self.serial_worker.signals.error.connect(self.show_error)
            self.thread_pool.start(self.serial_worker)

    def on_camera_ready(self, camera):
        self.camera = camera
        # self.enable_all_buttons()
        self.camera_ready = True
        self.check_if_all_devices_ready()

    def on_serial_ready(self, ser):
        self.ser = ser
        self.serial_ready = True
        self.serial_listener = SerialListener_crc(self.ser, self.loger)
        self.serial_listener.serial_signal.connect(self.ser_infer)
        self.serial_listener.serial_signal_w_cloth.connect(self.ser_infer)
        self.serial_listener.start()
        if int(self.cloth_d) == 0 and int(self.cloth_w) == 0:
            self.ser.write(self.w_full_frame_data_take)
            time.sleep(0.1)
        self.check_if_all_devices_ready()
    
    def check_if_all_devices_ready(self):
        if self.camera_ready and self.serial_ready:
            self.start_preview()
            self.enable_all_buttons()
            QMessageBox.information(self, "提示", "初始化成功！")
            
    ### 待定， 初始化失败
    def on_init_error(self, error_msg):
        self.show_error(error_msg)
        self.close()
    ###################

    def show_error(self, error_msg):
        QMessageBox.critical(self, "错误", error_msg)

    def toggle_timer(self, state):
        if state == 2:
            self.preview_timer.start(33)
        else:
            self.preview_timer.stop()
    def switch_test_demo(self, state):
        if state == 2:
            self.inference_button.setVisible(True)
            self.front_radio.setVisible(True)
            self.rear_radio.setVisible(True)
        else:
            self.inference_button.setVisible(False)
            self.front_radio.setVisible(False)
            self.rear_radio.setVisible(False)

    def start_preview(self):
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)
        # self.preview_timer.start(33)
        
        self.infer_timer = QTimer(self)
        self.infer_timer.timeout.connect(self.run_inference)
        self.infer_running = False

    def toggle_inference(self):
        if self.infer_running:
            self.infer_timer.stop()
            self.infer_running = False
            self.inference_button.setText("执行推理")
            
        else:
            if self.reload_buttons[0].isChecked():
                self.reload_config()
                self.reload_buttons[0].setChecked(False)
            self.inference_count = 0
            self.infer_timer.start(30)  
            self.infer_running = True
            self.inference_button.setText("停止推理")

    def update_preview(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # rgb_image = cv2.resize(rgb_image, (960, 720))
                rgb_image = cv2.resize(rgb_image, (720, 960))
                # h, w, ch = rgb_image.shape
                # bytes_per_line = ch * w
                rgb_image = QImage(rgb_image, 720, 960, QImage.Format_RGB888)
                q_img = rgb_image.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(QPixmap.fromImage(q_img))

    ##################################################
    def disable_all_buttons(self):
        """禁用所有按钮"""
        self.checkbox_infer.setEnabled(False)
        self.save_button.setEnabled(False)
        self.p_detect_button.setEnabled(False)
        self.factor_detect_button.setEnabled(False)
        self.inference_button.setEnabled(False)
        # self.ser_inference_button.setEnabled(False)
        self.checkbox.setEnabled(False)
        self.checkbox_test.setEnabled(False)
        # self.checkbox_segm.setEnabled(False)
        self.front_radio.setEnabled(False)
        self.rear_radio.setEnabled(False)
        self.reload_buttons[0].setEnabled(False)
        self.reload_buttons[1].setEnabled(False)
        self.reload_buttons[2].setEnabled(False)

    def enable_all_buttons(self):
        """恢复所有按钮"""
        self.checkbox_infer.setEnabled(True)
        self.save_button.setEnabled(True)
        self.p_detect_button.setEnabled(True)
        self.factor_detect_button.setEnabled(True)
        self.inference_button.setEnabled(True)
        # self.ser_inference_button.setEnabled(True)
        self.checkbox.setEnabled(True)
        self.checkbox_test.setEnabled(True)
        # self.checkbox_segm.setEnabled(True)
        self.front_radio.setEnabled(True)
        self.rear_radio.setEnabled(True)
        self.reload_buttons[0].setEnabled(True)
        self.reload_buttons[1].setEnabled(True)
        self.reload_buttons[2].setEnabled(True)
    ##################################################

    def run_inference(self):
        '''
        仅仅是查看模型推理效果
        '''
        if self.camera is not None:
            ret, frame = self.camera.read()
            # 12mm镜头 head
            if self.front_radio.isChecked():

                region = (self.start_y, self.end_y, self.start_x, self.end_x)
                frame, head_frame_re, detection, segm_pred = self.process_frame(frame, region, self.pad)

                pred = np.where(segm_pred[0].squeeze() > 0, 255, 0).astype(np.uint8)
                pred = cv2.resize(pred, (frame.shape[1],frame.shape[0]))
                pts = detection[0][0][0].astype(np.float64)
                pts[:, 1] *= self.factor_w_offset

            else:
                # 12mm镜头  tail
                # rotate
                region = (self.tail_start_y, self.tail_end_y, self.tail_start_x, self.tail_end_x)
                print('region', region)
                print('frame shape ', frame.shape)
                frame, tail_frame_re, detection, segm_pred = self.process_frame(frame, region, 0)

                pred = np.where(segm_pred[0].squeeze() > 0, 255, 0).astype(np.uint8)
                pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]))
                pts = detection[0][0][0].astype(np.float64)
                pts[:, 0] *= self.tail_factor_h_offset 
                pts[:, 1] *= self.tail_factor_w_offset 

            pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            color_list = [[255, 0, 0], [0,255,0]]
            pts = pts[pts[:, 1].argsort()]
            # if pts[0][1] < 0:
            #     pts[0][1] = pts[1][1]
            if VAIL_MODEL:
                '''
                记录关键点检测和分割检测的结果
                主要是在计算布料边缘的轮廓点集合的平均值
                '''
                if self.inference_count >= 500:
                    print('已经完成检测500次')
                    return 
                color = self.nametext.text()
                max_contour = get_max_contour(pred)
                x_coords, _ = get_kde_x(pred, max_contour, pts,False)
                optimizer = AlignmentOptimizer(x_coords, x_coords, self.roi_rotate_base_p_x, self.roi_tail_rotate_base_p_x, True)
                optimal_head, optimal_tail, circle_head_x, circle_x,circle_head_x_, circle_x_  = optimizer.find_optimal_split()
                self.write_to_excel(str(color),self.inference_count, optimal_head, None)
                self.inference_count += 1


            pts = pts.astype(np.int32)
            for i, pt in enumerate(pts):
                cv2.circle(frame, tuple(pt.astype(np.int32)), 5, color_list[i], 5)
     
            frame = cv2.addWeighted(frame, 0.8, pred_bgr, 0.5, 0)

            roi_img_cp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi_img_cp = cv2.resize(roi_img_cp, (960, 720))
            # roi_img_cp = cv2.resize(roi_img_cp, (1075, 400))
            # print(roi_img_cp.shape[1], roi_img_cp.shape[0])
            image = QImage(roi_img_cp, 960, 720, QImage.Format_RGB888)
            # 调整图像大小以适应标签尺寸
            if self.front_radio.isChecked():
                scaled_image = image.scaled(self.label_head.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_head.setPixmap(QPixmap.fromImage(scaled_image))
            else:
                scaled_image = image.scaled(self.label_tail.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_tail.setPixmap(QPixmap.fromImage(scaled_image))

            
    def write_to_excel(self, name, count, result1, result2):
        '''
        将推理结果和次数写入Excel
        '''

        # 每次写入都保存Excel文件
        if '全流程' in name:
            self.excel_data.append([count, result1,result2])
            df = pd.DataFrame(self.excel_data, columns=['次数', '前部推理结果', '尾部推理结果'])
        else:
            self.excel_data.append([count, result1])
            df = pd.DataFrame(self.excel_data, columns=['次数', '推理结果'])
        df.to_excel(f'{name}_inference_results.xlsx', index=False)

    def switch_mode(self, state):
        if state == 2:
            self.label_head.setVisible(True)
            if not self.label_head in self.hbox_labels.children():
                self.hbox_labels.addWidget(self.label_head)
            self.label_tail.setVisible(True)
            if not self.label_tail in self.hbox_labels.children():
                self.hbox_labels.addWidget(self.label_tail)
            if VAIL_MODEL:
                self.label_align.setVisible(True)
                if not self.label_align in self.hbox_labels.children():
                    self.hbox_labels.addWidget(self.label_align)
        else:
            self.label_head.setVisible(False)
            if self.label_head in self.hbox_labels.children():
                self.hbox_labels.removeWidget(self.label_head)
            self.label_tail.setVisible(False)
            if self.label_tail in self.hbox_labels.children():
                self.hbox_labels.removeWidget(self.label_tail)
            if VAIL_MODEL:
                self.label_align.setVisible(False)
                if not self.label_align in self.hbox_labels.children():
                    self.hbox_labels.removeWidget(self.label_align)
        self.hbox_labels.update()

    def is_segm(self, state):
        if state == 2:
            self.segm_infer = False
        else:
            self.segm_infer = True

    def create_folders(self):
        time_str = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.Data_save_path = self.config['origin_im_save_path'] + '//' + time_str
        if not os.path.exists(self.Data_save_path):
            os.makedirs(self.Data_save_path, exist_ok=True)

        self.label_save_path = self.config['origin_label_path'] + '//' + time_str
        if not os.path.exists(self.label_save_path):
            os.makedirs(self.label_save_path, exist_ok=True)

        self.result_save_path_head = self.config['infer_head_im_save_path'] + '//' + time_str
        if not os.path.exists(self.result_save_path_head):
            os.makedirs(self.result_save_path_head, exist_ok=True)

        self.result_save_path_tail = self.config['infer_tail_im_save_path'] + '//' + time_str
        if not os.path.exists(self.result_save_path_tail):
            os.makedirs(self.result_save_path_tail, exist_ok=True)


        # self.detect_error_save_path = 'Data/thor_kepoint/error_detect'
        self.detect_error_save_path = self.config['detect_error_save_path'] + '//' + time_str
        self.detect_error_save_path = self.detect_error_save_path
        if not os.path.exists(self.detect_error_save_path):
            os.makedirs(self.detect_error_save_path, exist_ok=True)

        # self.save_path = 'Data/thor_keypoint_data'
        self.save_path = self.config['save_path'] + '//' + time_str
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def process_frame(self, frame, region, pad):
        """通用图像处理流程"""
        y1, y2, x1, x2 = region
        roi = frame[y1:y2, x1:x2]
        
        # 图像预处理
        if pad > 0:
            roi = cv2.copyMakeBorder(roi, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
        roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print('roi shape is: ', roi.shape)
        resized = cv2.resize(roi, (800, 1300))
        # frame_as = np.ascontiguousarray(resized[:, :, ::-1]) 
        frame_as = resized[:,:,::-1]
        frame_as = frame_as.transpose([2, 0, 1])
        detection, segm_pred = self.parallel_inference(frame_as, roi) 
        return roi, resized, detection, segm_pred
    
    def pre_frame(self, frame, region, pad):
        y1, y2, x1, x2 = region
        roi = frame[y1:y2, x1:x2]
        # 图像预处理
        if pad > 0:
            roi = cv2.copyMakeBorder(roi, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
        roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print('roi shape is: ', roi.shape)
        resized = cv2.resize(roi, (800, 1300))
        frame_as = resized[:,:,::-1]
        frame_as = frame_as.transpose([2, 0, 1])
        return frame_as
    
    def parallel_inference(self, frame_as, roi):
        """并行执行检测与分割推理"""
        detection, segm_pred = [], []
        t_detect = Thread(target=lambda: detection.append(infer_frame(frame_as, self.sess, self.input_name, self.output_names)))
        t_segm = Thread(target=lambda: segm_pred.append(infer_segm(roi, self.segm_sess, self.segm_input_name, self.segm_output_names, self.is_640))) if self.segm_infer else None
        t_detect.start()
        if t_segm: t_segm.start()
        t_detect.join()
        if t_segm: t_segm.join()
        return detection, segm_pred 
    
    def calculate_w(self, pt, max_contour, w, loc):
        # pts_x = pts[pts[:, 0].argsort()]
        num = w // 10
        # if num >= 5:
        #     num = 5

        if loc == 'head':
            # max_x = pts_x[-1][0]
            x_samples = [pt + 10 * i for i in range(num)]
        elif loc == 'tail':
            # max_x = pts_x[0][0]
            x_samples = [pt - 10 * i for i in range(num)]
        top_y = []
        bottom_y = []
        last_max_y = None
        last_min_y = None
        for x in x_samples:
            max_y, min_y = find_max_min_y_for_x(max_contour, x)
            if last_max_y is not None and last_min_y is not None:
                if -40 <= (max_y - last_max_y) <= 40 and -40 <= (min_y - last_min_y) <= 40:
                    top_y.append(max_y)
                    bottom_y.append(min_y)
            else:
                top_y.append(max_y)
                bottom_y.append(min_y)
            last_max_y = max_y
            last_min_y = min_y

        t_p_offset = [t - b for t, b in zip(top_y, bottom_y)]
        print('t_p_offset', t_p_offset)
        t_p_offset_ = [d * 0.0414 for d in t_p_offset]
        print('布料宽度列表', t_p_offset_)
        # mask = filter_outliers(t_p_offset)
        # t_p_offset = np.array(t_p_offset)
        # t_p_offset = t_p_offset[mask]
        # print('t_p_offset', t_p_offset)
        average_value = calculate_average_excluding_extremes(t_p_offset_)
        # average_value = average_value - 0.8
        return average_value, t_p_offset

    def ser_infer(self, data, alignment_count, cloth_d):

        '''
        优化y
        前部只需要获取到x坐标点集
        尾部优化x坐标之后， 在优化y
        '''

        if data == 'signal_error':
            self.ser.write(self.head_error_cmd)  # 直接给下位机发送错误信号
        if data == 'cloth_w':
            self.cloth_w = alignment_count
            self.cloth_d = cloth_d
            print(f'收到下位机发送的布料宽度：{self.cloth_w}')
            print(f'收到下位机发送的布料止口深度：{self.cloth_d}')

            if 40 <= self.cloth_w < 81 and 0 < self.cloth_d < 10:
                self.loger.info(f'收到下位机发送的布料宽度：{self.cloth_w}')
                self.loger.info(f'发送接收布料宽度响应指令: {self.w_full_frame_data}')
                time.sleep(0.01)
                self.ser.write(self.w_full_frame_data)
            else:
                time.sleep(0.01)
                self.error_loger.error(f'下位机发送的布料宽度或者止口深度异常：{self.cloth_w}')
                print(f'下位机发送的布料宽度或者止口深度异常：{self.cloth_w}')
                self.ser.write(self.w_error_cmd)

        if int(self.cloth_d) == 0 and int(self.cloth_w) == 0:
            self.ser.write(self.w_full_frame_data_take)
            self.loger.info(f'发送访问布料宽度指令: {self.w_full_frame_data_take}')

        if data == 'head':
            try:
                self.tail_count = 0
                self.loger.info(f'前部收到串口数据：{data}')
                time_s = time.time()
            
                ret, frame = self.camera.read()
                if not ret:
                    self.show_error('相机打开失败, 请重启相机')
                    return
                # data_frame_f = self.sof + self.cmd_id_bytes_f + bytes([len(self.data_f)]) + self.data_f
                self.loger.info(f'前部发送前部对齐指令: {self.head_move_data_frame}')
                self.ser.write(self.head_move_data_frame) 

                region = (self.start_y, self.end_y, self.start_x, self.end_x)
                self.head_frame, self.head_frame_re, detection, segm_pred = self.process_frame(frame, region, self.pad)

                pred = np.where(segm_pred[0].squeeze() > 0, 255, 0).astype(np.uint8)
                self.head_pred = cv2.resize(pred, (self.head_frame.shape[1],self.head_frame.shape[0]))  # 800*2050的尺寸
                self.head_pred_re = cv2.resize(pred, (self.head_frame_re.shape[1],self.head_frame_re.shape[0]))
                self.head_max_contour = get_max_contour(self.head_pred)

                # 前部使用正接矩形判断布料是否出界
                head_rect_x, head_rect_y, head_rect_w, head_rect_h = cv2.boundingRect(self.head_max_contour)
                # print('矩形',head_rect_x, head_rect_y, head_rect_w, head_rect_h)
                # print('self.head_start_x_border', self.head_start_x_border)
                # print('self.head_end_x_border', self.head_end_x_border)
                # cv2.rectangle(self.head_frame, (head_rect_x, head_rect_y), (head_rect_x + head_rect_w, head_rect_y + head_rect_h), (0, 255, 0), 2)
                # cv2.namedWindow('head_frame', cv2.WINDOW_NORMAL)
                # cv2.imshow('head_frame', self.head_frame)
                # cv2.waitKey(0)
                head_de_is_ok = True
                if head_rect_x < self.head_start_x_border:
                    head_de_is_ok = False
                    print('矩形x坐标小于夹爪边界')
                    self.error_loger.error('前部轮廓x坐标小于夹爪边界')
                if head_rect_x > self.head_end_x_border:
                    head_de_is_ok = False
                    print('矩形x坐标大于夹爪边界')
                    self.error_loger.error('前部轮廓x坐标大于夹爪边界')
                if head_rect_y < 10:
                    head_de_is_ok = False
                    self.error_loger.error('前部轮廓y坐标小于夹爪边界')
                    print('矩形y坐标小于夹爪边界')
                if head_rect_h < 700:
                    head_de_is_ok = False
                    self.error_loger.error('前部轮廓高度小于700')
                    print('矩形高度小于700')
                if not head_de_is_ok:
                    self.error_loger.error('前部检测失败。 图像边界超出')
                    self.ser.write(self.head_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)
                    # cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)
                    return
                
                self.head_max_contour_re = get_max_contour(self.head_pred_re)
                if self.head_max_contour is None:
                    self.error_loger.error('前部分割检测失败')
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re) 
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)  
                    self.ser.write(self.head_error_cmd)

                self.head_pred_bgr = cv2.cvtColor(self.head_pred, cv2.COLOR_GRAY2BGR)
                color_list = [[255, 0, 0], [0,255,0]]
                pts = detection[0][0][0].astype(np.float64)
                pts[:, 1] *= self.factor_w_offset
                print('前部pts: ', pts)
                self.loger.info(f'前部pts: {pts}')


                ########## 增加宽度检测  目前误差在 ±1 之间 ########
                # average_value = self.calculate_w(pts, self.head_max_contour, 'head')
                # # print(f'前部y偏移量：{t_p_offset}')
                # self.error_loger.error(f'前部计算的布料宽度：{average_value}')
                ########## 增加宽度检测  目前误差在 ±1 之间 ########
               
                if np.any(pts[:, 0]<self.head_start_x_border):
                    print('所有x坐标小于夹爪边界')
                elif np.any(pts[:, 0]>self.head_end_x_border):
                    print('所有x坐标大于夹爪边界')
                elif abs(abs(pts[0][0]) - abs(pts[1][0])) > self.x_diff:
                    print('x轴偏差过大')
                elif np.all(pts < 0):
                    print('所有坐标小于0')

                condition = check_condition(pts, self.head_start_x_border, self.head_end_x_border, self.x_diff, self.loger)
                ######### 增加检测不对重复检测的功能 #############
                if condition:
                    self.head_de_is_ok = True
                    pts = pts
                    self.loger.info('前部关键点检测正常')
                    self.loger.info(f'前部关键点检测正常结果pts: {pts}')
                elif self.head_max_contour is not None:
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re) 
                    self.loger.info('前部关键点检测异常, 重新检测轮廓')
                    ret, frame = self.camera.read()
                    frame_as = self.pre_frame(frame, region, self.pad)
                    # background = np.ones_like(self.head_pred) * 114
                    # cv2.drawContours(background, [self.head_max_contour], -1, (255), thickness=cv2.FILLED)
                    # self.background_im = background
                    # head_pred_bgr = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
                    # head_pred_bgr = cv2.resize(head_pred_bgr, (self.head_frame_re.shape[1],self.head_frame_re.shape[0]))
                    # head_pred_bgr = head_pred_bgr[:, :, ::-1] 
                    # head_pred_bgr = head_pred_bgr.transpose([2, 0, 1])
                    detection_contour = infer_frame(frame_as, self.sess, self.input_name, self.output_names)
                    pts = detection_contour[0][0].astype(np.float64)
                    pts[:, 1] *= self.factor_w_offset
                    print('前部pts: ', pts)
                    self.loger.info(f'前部关键点检测异常, 重新获取图像推理结果pts: {pts}')
                    condition = check_condition(pts, self.head_start_x_border, self.head_end_x_border, self.x_diff, self.loger)

                    if condition:
                        self.head_de_is_ok = True
                        pts = pts
                        self.loger.info('重新检测轮廓正常')
                        self.loger.info(f'前部关键点检测异常结果pts: {pts}')
                    else:
                        self.loger.info('重新检测轮廓异常， 直接使用分割找角点')
                        # print('self.background_im shape is: ', self.head_pred_re.shape)
                        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
                        # cv2.imshow('mask', self.head_pred_re)
                        # self.background_im = cv2.resize(self.background_im, (self.head_frame_re.shape[1],self.head_frame_re.shape[0]))
                        # print('self.background_im shape is: ', self.background_im.shape)
                        # cv2.waitKey(0)
                        # max_cnt =get_max_contour(self.head_max_contour_re)
                        self.head_pred_re_bgr = cv2.cvtColor(self.head_pred_re, cv2.COLOR_GRAY2BGR)
                        pts = get_contour_corner(self.head_max_contour_re, self.head_pred_re_bgr, 'start', False)  
                        if pts is not None:
                            pts = np.array(pts).astype(np.float64)
                            pts[:, 1] *= self.factor_w_offset 
                            print('从mask中检测到的角点 前部pts: ', pts)
                            self.loger.info(f'从mask中检测到的角点 前部pts: {pts}')
                            condition = check_condition(pts, self.head_start_x_border, self.head_end_x_border, self.x_diff, self.loger)
                            if condition: 
                                self.head_de_is_ok = True
                                pts = pts
                                self.loger.info(f'前部直接从轮廓获取角点结果pts: {pts}')
                            else:
                                self.head_de_is_ok = False
                                self.error_loger.error('前部检测有误， 三种方法检测均异常， 请检查图像')
                                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)
                                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)   
                                return
                        else:
                            self.head_de_is_ok = False
                            self.error_loger.error('前部 pts is None 检测有误， 三种方法检测均异常， 请检查图像')
                            time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                            cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)
                            cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)   
                            return

                else:
                    self.head_de_is_ok = False
                    self.error_loger.error('前部分割检测有误， 请检查图像')
                    # 取消图像保存
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re) 
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)  
                    return
                ######### 增加检测不对重复检测的功能 #############
                

                ####################### 单一检测方法 #######################
                pts_y = pts[pts[:, 1].argsort()]  # 按y轴排序
                self.head_x_coords, self.points_between_head = get_kde_x(self.head_pred, self.head_max_contour, pts,'head',False) # 返回边缘x坐标列表， 用于与尾部计算偏移
                if len(self.head_x_coords) < 50:
                    self.error_loger.error('前部检测失败。 无法从轮廓边缘获取到正确数量的x坐标')
                    self.ser.write(self.head_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re) 
                    return
                
                self.loger.info(f'前部花费总时间： {time.time() - time_s}s')
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.Data_save_path}/{time_str}_start.png',self.head_frame_re)
                with open (f'{self.label_save_path}/{time_str}_start.txt', 'w', encoding='utf-8') as f:
                    for pt in pts:
                        f.write(f"{int(pt[0])},{int(pt[1])}\n")
                # else:
                #     self.loger.info('前部检测有误， 请检查图像')
                #     self.head_de_is_ok = False
                #     # 取消图像保存
                #     time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                #     cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)  
                ####################### 单一检测方法 #######################
            except Exception as e:
                self.error_loger.error(f'前部检测异常：{e}')
                self.ser.write(self.head_error_cmd)
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re) 
                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re) 
                # return
        elif data == 'tail':
            try:
                self.tail_count += 1
                self.loger.info(f'尾部收到串口数据：{data}')
                ret, frame = self.camera.read()
                if not ret:
                    self.show_error('相机打开失败, 请重启相机')
                    return
                region = (self.tail_start_y, self.tail_end_y, self.tail_start_x, self.tail_end_x)
                roi_tail, self.tail_frame_re, detection, segm_pred = self.process_frame(frame, region, 0)

                pred = np.where(segm_pred[0].squeeze() > 0, 255, 0).astype(np.uint8)
                pred = cv2.resize(pred, (roi_tail.shape[1], roi_tail.shape[0]))
                pred_re = cv2.resize(pred, (self.tail_frame_re.shape[1], self.tail_frame_re.shape[0]))
                
                pixel_d = int(self.cloth_d / 0.042) + 10
                end_p = self.tail_h - 20
                # print(11111111111111)
                print('pixel_d, end_p',pixel_d, end_p)

                #  print('pred shape is:', pred.shape)
                max_contour = get_max_contour(pred)

                # 使用正接矩形进行判断是否超出边界
                rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(max_contour)
                print('矩形',rect_x, rect_y, rect_w, rect_h)
                tail_de_is_ok = True
                if rect_x + rect_w < pixel_d:
                    tail_de_is_ok = False
                    print('尾部x坐标在边界， 准备超出视野')
                    self.error_loger.error('尾部x坐标在边界， 准备超出视野')
                if rect_y + rect_h > 2490:
                    tail_de_is_ok = False
                    print('尾部y坐标在下边界， 准备超出视野')   
                    self.error_loger.error('尾部y坐标在下边界， 准备超出视野')
                if rect_y < 10:
                    tail_de_is_ok = False
                    print('尾部y坐标在上边界， 准备超出视野')
                    self.error_loger.error('尾部y坐标在上边界， 准备超出视野')
                if not tail_de_is_ok:
                    self.error_loger.error('尾部检测失败。 图像边界超出')
                    self.ser.write(self.tail_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                    return
                if rect_x + rect_w > end_p:
                    print('尾部x坐标在边界， 角点未进入视野')
                    self.error_loger.error('尾部x坐标在边界， 角点未进入视野')
                    tail_ = rect_y + rect_h 
                    tail_ -= self.roi_tail_rotate_base_p_x
                    print('tail_', tail_)
                    tail_x = round(tail_ * self.factor, 2)  # 对尾部x坐标进行优化
                    tail_y = round(0 * self.factor, 2)
                    self.loger.info(f'尾部与基点的偏移量：{tail_x}, {tail_y}')
                    tail_x_bytes = struct.pack('<f', tail_x)
                    tail_y_bytes = struct.pack('<f', tail_y)
                    tail_data_length = len(tail_x_bytes) + len(tail_y_bytes)
                    if tail_data_length != 8:
                        print(f'尾部字节数据长度不匹配， 实际长度：{tail_data_length}')
                        self.error_loger.error(f'尾部字节数据长度不匹配， 实际长度：{tail_data_length}')
                        self.ser.write(self.tail_error_cmd)
                        return
                    head_x = round(self.head_end_x_border_ * self.factor, 2) # 对x坐标进行优化
                    # head_y = round(self.head_p_y * self.factor, 2)
                    head_y = round(0 * self.factor, 2)
                    self.loger.info(f'前部与基点的偏移量：{head_x}, {head_y}')
                    x_bytes = struct.pack('<f', head_x)
                    y_bytes = struct.pack('<f', head_y)
                    head_data_length = len(x_bytes) + len(y_bytes)

                    if head_data_length != 8:
                        self.error_loger.error(f'前部字节数据长度不匹配， 实际长度：{data_length}')
                        self.ser.write(self.tail_error_cmd)
                        return
                    alignment_count_bytes = struct.pack('<i', alignment_count)
                    
                    data_length = head_data_length + tail_data_length + len(alignment_count_bytes)

                    tail_front_part_data_frame = self.tail_move_byte_data + bytes([data_length])
                    tail_front_part_data_frame_crc = crc8_rohc(tail_front_part_data_frame)
                    full_data_frame = tail_front_part_data_frame + tail_front_part_data_frame_crc + alignment_count_bytes + x_bytes + y_bytes + tail_x_bytes + tail_y_bytes
                    full_data_frame_crc = crc8_rohc(full_data_frame)
                    data_frame = full_data_frame + full_data_frame_crc

                    self.ser.write(data_frame) 
                    # time.sleep(0.05)
                    # self.ser.write(self.tail_move_byte_data)
                    self.loger.info(f'尾部发送动作指令：{data_frame}')
                    return

                # cv2.rectangle(roi_tail, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)
                # cv2.namedWindow('roi_tail', cv2.WINDOW_NORMAL)
                # cv2.imshow('roi_tail', roi_tail)
                # cv2.waitKey(0)        
                  
                max_contour_re = get_max_contour(pred_re)
                '''
                尾部可能拉的太远 只显示两个角点， 会有两个轮廓， 虽然取了最大轮廓， 但是这时布料分割出来的结果是两个部分
                所以还要进行y值的判断， 轮廓的最大与最小y值小于self.y_offset-50, 则反馈错误
                '''
                if max_contour is None:
                    self.error_loger.error('尾部分割检测失败')
                    self.ser.write(self.tail_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)

                tail_pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            
                color_list = [[0, 0, 255], [0,255,0],[255,0,0], [255,0,0], [0,255,255],[255,0,255]]
                pts = detection[0][0][0].astype(np.float64)
            
                pts[:, 0] *= self.tail_factor_h_offset 
                pts[:, 1] *= self.tail_factor_w_offset 
                # pts_y = pts[pts[:, 1].argsort()]
                print('尾部pts: ', pts)
                self.loger.info(f'尾部pts: {pts}')

                if abs(pts[0][0]) - abs(pts[1][0]) > self.x_diff:
                    print('x轴偏差过大')
                # elif np.any(pts[0, :] > 1150):
                #     print('x坐标在尾部边界')
                elif np.all(pts < 0):
                    print('有坐标小于0')
                elif np.any(pixel_d > pts[:, 0]) or np.any(pts[:, 0] > end_p):
                    print('x坐标在尾部边界')
                
                '''
                新增尾部布料在图像边界的判断
                '''
                condition_tail = check_tail_condition(pts, pixel_d, end_p, self.x_diff, self.loger)
                if condition_tail:
                    '''出现尾部没有拉到位的情况， 但是关键点识别的很近， 所以还需要加一个y的判断（暂时取一个12mm(300mm)为阈值）以及增加x方向不能太靠近图像边缘'''
                    pts = pts
                    self.loger.info('尾部关键点检测正常')
                    self.loger.info(f'尾部关键点检测正常结果pts: {pts}')
                elif max_contour is not None:
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)              
                    self.loger.info('尾部关键点检测异常, 重新检测轮廓')
                    # background = np.ones_like(pred) * 114
                    # cv2.drawContours(background, [max_contour], -1, (255), thickness=cv2.FILLED)
                    # tail_pred_bgr = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
                    # tail_pred_bgr = cv2.resize(tail_pred_bgr, (self.tail_frame_re.shape[1],self.tail_frame_re.shape[0]))
                    # tail_pred_bgr = tail_pred_bgr[:, :, ::-1]
                    # tail_pred_bgr = tail_pred_bgr.transpose([2, 0, 1])
                    ret, frame = self.camera.read()
                    tail_frame = self.pre_frame(frame, region, 0)
                    detection_contour = infer_frame(tail_frame, self.sess, self.input_name, self.output_names)
                    pts = detection_contour[0][0].astype(np.float64)
                    pts[:, 0] *= self.tail_factor_h_offset 
                    pts[:, 1] *= self.tail_factor_w_offset 
                    # pts_y = pts[pts[:, 1].argsort()]
                    print(222222222222222222222222222222)
                    print('尾部pts: ', pts)
                    condition_tail = check_tail_condition(pts, pixel_d, end_p, self.x_diff, self.loger)

                    if condition_tail:
                        pts = pts
                        self.loger.info(f'尾部关键点检测异常常重新推理结果pts: {pts}')
                    else:
                        self.loger.info('重新检测轮廓异常， 直接使用分割找角点')
                        pred_re_bgr = cv2.cvtColor(pred_re, cv2.COLOR_GRAY2BGR)
                        pts = get_contour_corner(max_contour_re, pred_re_bgr, 'end', False)
                        if pts is not None:
                            pts = np.array(pts).astype(np.float64)
                            pts[:, 0] *= self.tail_factor_h_offset 
                            pts[:, 1] *= self.tail_factor_w_offset
                            self.loger.info(f'尾部直接从轮廓获取角点结果pts: {pts}')
                            condition_tail = check_tail_condition(pts, pixel_d, end_p, self.x_diff, self.loger)
                        
                            if condition_tail:
                                pts = pts
                                self.loger.info(f'尾部直接从轮廓获取角点结果pts: {pts}')
                            else:
                                self.error_loger.error('尾部检测有误， 请检查图像')
                                self.ser.write(self.tail_error_cmd)
                                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)
                                return
                        else:
                            self.ser.write(self.tail_error_cmd)
                            self.error_loger.error('尾部检测有误， 三种方法检测均异常， 请检查图像')
                            time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                            cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)  
                            cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)
                            return
                else:
                    self.error_loger.error('尾部检测有误， 请检查图像')
                    self.ser.write(self.tail_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)
                    return
                
                tail_x_coords, points_between_tail = get_kde_x(pred, max_contour, pts, 'tail',False)
                print('tail_x_coords', tail_x_coords)
                if len(tail_x_coords) < 50:
                    self.error_loger.error('尾部检测失败， 无法从轮廓边缘获取到正确数量的x坐标')
                    self.ser.write(self.tail_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)
                    return
                
                if self.cloth_d == 0:
                    is_V = False
                elif 0 < self.cloth_d < 10:
                    is_V = True
                if hasattr(self, 'head_x_coords') and self.head_x_coords is not None:
                    optimizer = AlignmentOptimizer(self.head_x_coords, tail_x_coords, self.roi_rotate_base_p_x, self.roi_tail_rotate_base_p_x, is_V)
                    optimal_head, optimal_tail, circle_head_x, circle_x,circle_head_x_, circle_x_, total_flag  = optimizer.find_optimal_split()
                    if total_flag:
                        self.total_ += 1
                        self.loger.info(f'进入6666 共计： {self.total_}')
                    else:
                        self.t_ += 1
                        self.loger.info(f'未进入 共计： {self.t_}')
                    # optimizer.visualize()
                else:
                    '''这里是不是要发送检测错误的信号'''
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)  
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)
                    print('self.head_x_coords', self.head_x_coords)
                    self.error_loger.error('前部检测检测有误， 没有head_x_coords， 请检查图像')
                    self.ser.write(self.tail_error_cmd)
                    print('视觉检测有误')
                    return 
                
                # 优化y
                '''
                1、首先获取前部和尾部上下边缘的y值
                2、计算同侧前部和尾部得偏移  不取绝对值
                3、计算上边缘与下边缘总的偏移  不去绝对值
                4、前部 找到与平均值最近的p1_y 加上总的偏移 / 总的点数
                5、尾部 找到与平均值最近的p3_y
                '''
                w = self.x_border - circle_head_x_ - 10
                self.loger.info(f'轮廓取的宽度： {w}')
                self.loger.info(f'对y值优化时，取的x开始点以及结束点： circle_head_x_: {circle_head_x_}, self.x_border:{self.x_border}, circle_x_: {circle_x_}')

                head_avg, head_t_p_offset = self.calculate_w(circle_head_x_, self.head_max_contour, w, 'head')
                self.loger.info(f'前部布料平均宽度：{head_avg}, 原始值：{head_t_p_offset}')
                if self.cloth_w - self.y_offset > head_avg or head_avg > self.cloth_w + self.y_offset:
                    self.error_loger.error(f'前部布料宽度异常，下位机发送的布料宽度：{self.cloth_w}, 检测宽度： {head_avg}')
                    self.ser.write(self.head_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)
                    return
                
                tail_avg, tail_t_p_offset =self.calculate_w(circle_x_, max_contour, w, 'tail')
                self.loger.info(f'尾部布料平均宽度：{tail_avg} 原始值： {tail_t_p_offset}')
                if self.cloth_w - self.y_offset > tail_avg or tail_avg > self.cloth_w + self.y_offset:
                    self.error_loger.error(f'尾部布料宽度异常，下位机发送的布料宽度：{self.cloth_w}, 检测宽度： {tail_avg}')
                    self.ser.write(self.tail_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)
                    return
                
                if w < 50:
                    step = 1
                else:
                    step = 10
                # print('step', step)
                self.loger.info(f'y值优化步长为： {step}')
                tail_border = circle_x_ - (w)
                self.loger.info(f'尾部获取y值的终点： {tail_border}')

                # TODO：
                # 这里需要进行优化， 增加宽度的判断， 只有宽度在一定范围内， 才会参与y值的优化
                '''
                因为看对齐错误时， 优化后的取值， -357 和 -220 明显的就属于异常值， 但是没有过滤掉
                top_y_offset [-357 -220 -175 -162 -155 -155 -155 -155 -155 -155 -155 -155 -155 -155
                -155 -161 -161 -161 -161 -161]
                bottom_offset [167 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156 156
                156 161]
                这就是问题所在， 需要将这异常值去掉
                '''

                # 将其改为使用缝纫点进行取y值 20250521
                self.loger.info(f'对y值优化时，取的x开始点以及结束点： circle_head_x_: {circle_head_x_}, self.x_border:{self.x_border}, circle_x_: {circle_x_}, tail_border: {tail_border}')
                p1,p2, p3, p4 = get_y(self.head_pred, self.head_max_contour, circle_head_x_, self.x_border, pred, max_contour, circle_x_, tail_border,step, False)
                if p1 is None:
                    self.error_loger.error(f'优化y值， 无法获取到有效点，检查get_y函数')
                    self.ser.write(self.tail_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)
                    return
                
                p1_y = p1[:, 1] 
                p2_y = p2[:, 1] # 上边缘
                p3_y = p3[:, 1] 
                p4_y = p4[:, 1] # 上边缘
                self.loger.info(f'y值优化后的结果： y1: {p1_y}, y2: {p2_y}, y3: {p3_y}, y4: {p4_y}')

                if len(p2_y) != len(p4_y):
                    self.error_loger.error(f'获取到的y值数量不等')
                    print('获取到的y值数量不等')
                    self.ser.write(self.tail_error_cmd)
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)
                    cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)
                    return
                
                ###################### 异常值再次过滤 ########################
                # 计算前部和尾部的y轴偏移

                top_y_offset = p2_y - p4_y  # 如果为负 说明上边缘 前部盖过尾部
                print('top_y_offset', top_y_offset)
                self.loger.info(f'前部与尾部上边缘y轴偏移：{top_y_offset}')
                # print('np.sum(top_y_offset', np.sum(top_y_offset))
                bottom_offset = p3_y - p1_y # 如果为负 说明前部盖过尾部
                print('bottom_offset', bottom_offset)
                self.loger.info(f'前部与尾部下边缘y轴偏移：{top_y_offset}')

                # print('np.sum(bottom_offset', np.sum(bottom_offset))

                # 分别计算两个数组的掩码
                mask_top = filter_outliers(top_y_offset)
                mask_bottom = filter_outliers(bottom_offset)

                # 联合掩码：仅保留两个数组均正常的位置
                combined_mask = mask_top & mask_bottom

                # 应用联合掩码过滤数据
                filtered_top = top_y_offset[combined_mask]
                filtered_bottom = bottom_offset[combined_mask]

                self.loger.info(f'过滤后的top_y_offset： {filtered_top}')
                self.loger.info(f'过滤后的bottom_offset： {filtered_bottom}')

                print("过滤后的 top_y_offset:", filtered_top)
                print("过滤后的 bottom_offset:", filtered_bottom)

                p1_y = p1_y[combined_mask]
                p2_y = p2_y[combined_mask]
                p3_y = p3_y[combined_mask]
                p4_y = p4_y[combined_mask]
                self.loger.info(f'对优化后的y值进行过滤后的结果： y1: {p1_y}, y2: {p2_y}, y3: {p3_y}, y4: {p4_y}')

                print('p1_y', p1_y)
                print('p2_y', p2_y)
                print('p3_y', p3_y)
                print('p4_y', p4_y)
                ###################### 异常值再次过滤 ########################

                # total_y_mean_offset = (np.sum(top_y_offset) - np.sum(bottom_offset)) / (2 * len(top_y_offset))
                total_y_mean_offset = (np.sum(filtered_top) - np.sum(filtered_bottom)) / (2 * len(filtered_top))
                self.loger.info(f'前部与尾部总y轴偏移：{total_y_mean_offset}')
                print('total_y_mean_offset', total_y_mean_offset)
                # if abs(total_y_mean_offset) > 500:
                #     self.error_loger.error(f'前部与尾部总y轴偏移异常， 偏移量：{total_y_mean_offset}')
                #     self.ser.write(self.tail_error_cmd)
                #     time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                #     cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                #     cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)
                #     cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)
                #     cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start_pred.png', self.head_pred_re)
                #     return
                
                # 找到与平均值最近的p2_y
                p2_y_mean = np.mean(p2_y)
                p2_y_diff = p2_y - p2_y_mean
                p2_y_diff_abs = np.abs(p2_y_diff) 
        
                p2_y_diff_min_idx = np.argmin(p2_y_diff_abs)
                p2_y_min = p2_y[p2_y_diff_min_idx]
                p1_y_min = p1_y[p2_y_diff_min_idx]
                p3_y_min = p3_y[p2_y_diff_min_idx]
                p4_y_min = p4_y[p2_y_diff_min_idx]
                offs = p2_y_min - p4_y_min
                # print('p4_y_min', p4_y_min)
                p2_final_y = p2_y_min + total_y_mean_offset - offs
                diff_2_2 = abs(p2_final_y - p2_y_min)
                head_p_y = p2_final_y - self.roi_rotate_base_p_y
                # print('head_p_y', head_p_y)
         
                #################### 计算实际输出的x对应的上下边缘的y值 ##########
                draw_head_y, draw_head_y1 = find_max_min_y_for_x(self.head_max_contour, circle_head_x_)
                draw_tail_y, draw_tail_y1 = find_max_min_y_for_x(max_contour, circle_x_)
                head_offst = abs(draw_head_y - draw_head_y1)
                tail_offst = abs(draw_tail_y - draw_tail_y1)
                # if abs(head_offst) * 0.042 < self.cloth_w - self.y_offset  or abs(head_offst) * 0.042 > self.cloth_w + self.y_offset:
                #     self.error_loger.error(f'前部布料宽度检测有误， 推理宽度：{abs(head_offst) * 0.042}, 预设宽度：{self.cloth_w}')
                #     # self.ser.write(self.byte_data)
                #     self.head_de_is_ok = False

                # if abs(tail_offst) * 0.042 < self.cloth_w - self.y_offset  or abs(tail_offst) * 0.042 > self.cloth_w + self.y_offset:
                #     self.error_loger.error(f'尾部布料宽度检测有误， 推理宽度：{abs(tail_offst) * 0.042}, 预设宽度：{self.cloth_w}')
                #     # self.ser.write(self.byte_data) 
                #     self.head_de_is_ok = False

                #################### 计算实际输出的x对应的上下边缘的y值 ##########

                #################### 尾部信息 ###############################
                tail_p_y = p4_y_min - self.roi_tail_rotate_base_p_y

                tail_x = round(optimal_tail * self.factor, 2)  # 对尾部x坐标进行优化

                tail_y = round(tail_p_y * self.factor, 2)
                self.loger.info(f'尾部与基点的偏移量：{tail_x}, {tail_y}')
                tail_x_bytes = struct.pack('<f', tail_x)
                tail_y_bytes = struct.pack('<f', tail_y)
                tail_data_length = len(tail_x_bytes) + len(tail_y_bytes)

                if tail_data_length != 8:
                    self.error_loger.error(f'尾部 字节数据长度不匹配， 实际长度：{tail_data_length}')
                    print(f'尾部字节数据长度不匹配， 实际长度：{tail_data_length}')
                    self.ser.write(self.tail_error_cmd)
                    return
                
                #################### 尾部信息 ###############################


                #################### 前部信息 ###############################
                if self.head_de_is_ok:
                    self.ser.reset_input_buffer() # 清理缓存
                    try:
                        head_x = round(optimal_head * self.factor, 2) # 对x坐标进行优化
                        # head_y = round(self.head_p_y * self.factor, 2)
                        head_y = round(head_p_y * self.factor, 2)
                        self.loger.info(f'前部与基点的偏移量：{head_x}, {head_y}')

                        x_bytes = struct.pack('<f', head_x)
                        y_bytes = struct.pack('<f', head_y)
                        alignment_count_bytes = struct.pack('<i', alignment_count)
                        head_data_length = len(x_bytes) + len(y_bytes)

                        if head_data_length != 8:
                            self.error_loger.error(f'前部 字节数据长度不匹配， 实际长度：{data_length}')
                            print(f'前部 字节数据长度不匹配， 实际长度：{data_length}')
                            self.ser.write(self.head_error_cmd)
                            return
                        data_length = head_data_length + tail_data_length + len(alignment_count_bytes)
                        # data_frame = self.sof + self.head_cmd_id_bytes + bytes([data_length]) + alignment_count_bytes + x_bytes + y_bytes + tail_x_bytes + tail_y_bytes
                        tail_front_part_data_frame = self.tail_move_byte_data + bytes([data_length])
                        tail_front_part_data_frame_crc = crc8_rohc(tail_front_part_data_frame)
                        full_data_frame = tail_front_part_data_frame + tail_front_part_data_frame_crc + alignment_count_bytes + x_bytes + y_bytes + tail_x_bytes + tail_y_bytes
                        full_data_frame_crc = crc8_rohc(full_data_frame)
                        data_frame = full_data_frame + full_data_frame_crc
                        self.ser.write(data_frame) 
                        print(f'发送前部+尾部数据：{data_frame}')
                        self.loger.info(f'前部+尾部发送对齐指令：{data_frame}')
                        # time.sleep(0.05)
                        # self.ser.write(self.tail_move_byte_data)
                        self.loger.info(f'尾部发送动作指令：{data_frame}')

                    except Exception as e:
                        self.error_loger.error(f'前部发送数据错误：: {e}')
                        print(f'前部发送数据错误：: {e}')
                else:
                    head_x = 0
                    head_y = 0
                    self.ser.write(self.head_error_cmd)

                #################### 前部信息 ###############################

                # tail_p_y = p4_y_min - self.roi_tail_rotate_base_p_y
                # tail_x = round(optimal_tail * self.factor, 2)  # 对尾部x坐标进行优化

                # if self.tail_offset is not None:
                #     '''
                #     这里需要判断是否需要进行尾部偏移
                #     '''
                #     pass

                # tail_y = round(tail_p_y * self.factor, 2)
                # self.loger.info(f'尾部与基点的偏移量：{tail_x}, {tail_y}')

                ########### 将数据写入excel表格 ###########
                # if VAIL_MODEL:
                #     '''
                #     同时将前部和尾部的x偏移写入表格
                #     '''
                #     # result = (str(head_x), str(tail_x))
                #     name_text = self.nametext.text()
                #     self.write_to_excel(name_text, self.ser_infer_count, head_x, tail_x)
                #     self.ser_infer_count += 1
                #     if self.ser_infer_count == 100:
                #         self.excel_data = []
                #         print('已经完成检测100次')
                #         return 
                ########### 将数据写入excel表格 ###########

                # tail_x_bytes = struct.pack('<f', tail_x)
                # tail_y_bytes = struct.pack('<f', tail_y)
                # tail_data_length = len(tail_x_bytes) + len(tail_y_bytes)

                # if tail_data_length != 8:
                #     print(f'字节数据长度不匹配， 实际长度：{tail_data_length}')
                    
                # tail_data_frame = self.sof + self.cmd_id_bytes_ + bytes([tail_data_length]) + tail_x_bytes + tail_y_bytes
                # self.ser.write(tail_data_frame) 
                # time.sleep(0.01)
                # self.ser.write(self.byte_data_f)
                # self.loger.info(f'尾部发送尾部对齐指令：{tail_data_frame}')
                # self.loger.info(f'尾部花费总时间：{time.time() - time_read_frame}s')
                # print(f'尾部花费总时间 {time.time() - time_read_frame}s')

                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.Data_save_path}/{time_str}_end.png', self.tail_frame_re)
                with open (f'{self.label_save_path}/{time_str}_end.txt', 'w', encoding='utf-8') as f:
                    for pt in pts:
                        f.write(f"{int(pt[0])},{int(pt[1])}\n")

                # head_frame_cp = self.head_frame.copy()
                # head_frame_cp1 = self.head_frame.copy()
                # if self.tail_count == 1:
                cv2.circle(self.head_frame, (int(circle_head_x_), int(p2_final_y)), 5, color_list[4], 5)
                cv2.circle(self.head_frame, (int(circle_head_x_), int(p2_final_y)+int(tail_offst)), 5, color_list[2], 5)
                cv2.line(self.head_frame, (int(circle_head_x_), int(p2_final_y)), (int(circle_head_x_), int(p1_y_min)), color_list[1], 5)
                head_frame_save = cv2.resize(self.head_frame, (800, 1300)) 
                cv2.imwrite(f'{self.result_save_path_head}/{time_str}_{head_x}_{head_y}_start.png', head_frame_save)
                # elif self.tail_count == 2:
                #     cv2.circle(head_frame_cp, (int(circle_head_x_), int(p2_final_y)), 5, color_list[4], 5)
                #     cv2.circle(head_frame_cp, (int(circle_head_x_), int(p2_final_y)+int(tail_offst)), 5, color_list[2], 5)
                #     cv2.line(head_frame_cp, (int(circle_head_x_), int(p2_final_y)), (int(circle_head_x_), int(p1_y_min)), color_list[1], 5)
                #     head_frame_save = cv2.resize(head_frame_cp, (800, 1300)) 
                #     cv2.imwrite(f'{self.result_save_path_head}/{time_str}_{head_x}_{head_y}_start.png', head_frame_save)                
                # elif self.tail_count == 3:
                #     cv2.circle(head_frame_cp1, (int(circle_head_x_), int(p2_final_y)), 5, color_list[4], 5)
                #     cv2.circle(head_frame_cp1, (int(circle_head_x_), int(p2_final_y)+int(tail_offst)), 5, color_list[2], 5)
                #     cv2.line(head_frame_cp1, (int(circle_head_x_), int(p2_final_y)), (int(circle_head_x_), int(p1_y_min)), color_list[1], 5)                
                #     head_frame_save = cv2.resize(head_frame_cp1, (800, 1300)) 
                #     cv2.imwrite(f'{self.result_save_path_head}/{time_str}_{head_x}_{head_y}_start.png', head_frame_save)

                cv2.circle(roi_tail, (int(circle_x_), int(p4_y_min)), 5, color_list[4], 5)
                cv2.circle(roi_tail, (int(circle_x_), int(p4_y_min)+int(diff_2_2)), 5, color_list[3], 5)
                cv2.circle(roi_tail, (int(circle_x_), int(p4_y_min)+int(head_offst)), 5, color_list[2], 5)
                cv2.line(roi_tail, (int(circle_x_), int(p4_y_min)), (int(circle_x_), int(p3_y_min)), color_list[1], 5)
            
                frame_save = cv2.resize(roi_tail, (800, 1300))
                cv2.imwrite(f'{self.result_save_path_tail}/{time_str}_{tail_x}_{tail_y}_end.png', frame_save)

                self.show_windows_im_head(self.head_frame, self.head_pred_bgr, self.label_head)
                self.show_windows_im_tail(roi_tail, tail_pred_bgr, self.label_tail)

                ########### 将数据写入excel表格 ###########
                # if VAIL_MODEL:
                #     mask = np.zeros_like(pred, dtype=np.uint8)
                #     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                #     print('mask shape is:', mask.shape)
                #     for coord in points_between_tail:
                #         cv2.circle(mask, (int(coord[0]), int(coord[1])), 3, (0, 0, 255), -1)
                #     abs_diff = int(abs(optimal_tail) - abs(optimal_head))
                #     for coord in self.points_between_head:
                #         cv2.circle(mask, (int(coord[0] + abs_diff), int(coord[1])), 3, (0, 255, 0), -1)
                #     rgb_mask = cv2.resize(mask, (960, 720))
                #     image_ = QImage(rgb_mask, 960, 720, QImage.Format_RGB888)
                #     scaled_image = image_.scaled(self.label_align.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                #     self.label_align.setPixmap(QPixmap.fromImage(scaled_image))
                ########### 将数据写入excel表格 ###########


                '''
                增加判断尾部是否在图像边界
                '''
                # elif np.any(pts[:, 0] > self.tail_x_border):
                #     print('pts', pts)
                #     print('尾部判断有误， 但是检测到一个有效点')
                #     self.error_loger.error('尾部判断有误， 但是检测到一个有效点')
                #     tail_ = np.max(pts[:, 0])
                #     tail_ -= self.roi_tail_rotate_base_p_x
                #     print('tail_', tail_)

                #     tail_x = round(tail_ * self.factor, 2)  # 对尾部x坐标进行优化
                #     tail_y = round(0 * self.factor, 2)
                #     self.loger.info(f'尾部与基点的偏移量：{tail_x}, {tail_y}')

                #     tail_x_bytes = struct.pack('<f', tail_x)
                #     tail_y_bytes = struct.pack('<f', tail_y)
                #     tail_data_length = len(tail_x_bytes) + len(tail_y_bytes)

                #     if tail_data_length != 8:
                #         print(f'尾部字节数据长度不匹配， 实际长度：{tail_data_length}')

                #     head_x = round(self.head_end_x_border_ * self.factor, 2) # 对x坐标进行优化
                #     # head_y = round(self.head_p_y * self.factor, 2)
                #     head_y = round(0 * self.factor, 2)
                #     self.loger.info(f'前部与基点的偏移量：{head_x}, {head_y}')

                #     x_bytes = struct.pack('<f', head_x)
                #     y_bytes = struct.pack('<f', head_y)
                #     head_data_length = len(x_bytes) + len(y_bytes)

                #     if head_data_length != 8:
                #         self.error_loger.error(f'前部 字节数据长度不匹配， 实际长度：{data_length}')

                #     alignment_count_bytes = struct.pack('<i', alignment_count)
                    
                #     data_length = head_data_length + tail_data_length + len(alignment_count_bytes)

                #     data_frame = self.sof + self.cmd_id_bytes + bytes([data_length]) + alignment_count_bytes + x_bytes + y_bytes + tail_x_bytes + tail_y_bytes
                #     self.ser.write(data_frame) 
                #     time.sleep(0.05)

                #     #################### 前部信息 ##############################
                        
                #     # time.sleep(0.01)
                #     self.ser.write(self.byte_data_f)
                #     self.loger.info(f'尾部发送动作指令：{self.byte_data_f}')
            
                    
                # else:
                #     self.error_loger.error('尾部检测有误， 请检查图像')
                #     self.ser.write(self.byte_data)
                #     # 取消图像保存
                #     time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                #     cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)

            except Exception as e:
                self.error_loger.error(f'尾部检测异常：{e}')
                self.ser.write(self.tail_error_cmd)            
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_tail_pred.png', pred_re)


    def show_windows_im_head(self, im, pred_bgr, im_label):
        frame_cp = cv2.addWeighted(im, 0.8, pred_bgr, 0.5, 0)
        frame_cp = cv2.flip(frame_cp, 1)
        # img_rgb = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2RGB)
        # img_rgb = cv2.resize(frame_cp, (960, 720))
        img_rgb = cv2.resize(frame_cp, (288, 768))
        image = QImage(img_rgb, 288, 768, QImage.Format_RGB888)
        scaled_image = image.scaled(288, 768, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        im_label.setPixmap(QPixmap.fromImage(scaled_image))

    def show_windows_im_tail(self, im, pred_bgr, im_label):
        frame_cp = cv2.addWeighted(im, 0.8, pred_bgr, 0.5, 0)
        # img_rgb = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2RGB)
        # img_rgb = cv2.resize(frame_cp, (960, 720))
        img_rgb = cv2.resize(frame_cp, (480, 960))
        image = QImage(img_rgb, 480, 960, QImage.Format_RGB888)
        scaled_image = image.scaled(480, 960, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        im_label.setPixmap(QPixmap.fromImage(scaled_image))

    # def show_windows_im(self, im, pred_bgr, im_label):
    #     # 合并图像
    #     frame_cp = cv2.addWeighted(im, 0.8, pred_bgr, 0.5, 0)
        
    #     # 将图像转换为 RGB 格式
    #     img_rgb = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2RGB)
        
    #     # 获取图像的原始尺寸
    #     height, width, channel = img_rgb.shape
    #     height, width = height * 0.4, width * 0.4
    #     # 创建 QImage 对象
    #     image = QImage(img_rgb, width, height, channel * width, QImage.Format_RGB888)
        
    #     # 设置 QLabel 的大小为图像的原始尺寸
    #     im_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    #     im_label.setFixedSize(width, height)
        
    #     # 将图像设置到 QLabel 中
    #     im_label.setPixmap(QPixmap.fromImage(image))

    def save_image(self):

        str_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.png'
        img_name = os.path.join(self.save_path, str_time)
        ret, image = self.camera.read()

        if not ret:
            print('camera init error')

        # 前部
        if self.config['save_im_way']['head']:
            # end_x = self.origin_x + self.head_x_offset
            # start_x = end_x - self.head_w
            # start_y = self.origin_y + self.head_y_offset
            # end_y = start_y + self.head_h
            # image = image[start_y:end_y, start_x:end_x]
            # image = cv2.copyMakeBorder(image, 250, 250, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
            # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            region = (self.start_y, self.end_y, self.start_x, self.end_x)
            
            y1, y2, x1, x2 = region
            roi = image[y1:y2, x1:x2]
            
            # 图像预处理
            if self.pad > 0:
                roi = cv2.copyMakeBorder(roi, self.pad, self.pad, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
            roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.resize(roi, (800, 1300))

        elif self.config['save_im_way']['full_im']:
            image = image
        else:
            # 尾部
            # end_x = self.origin_x + self.tail_x_offset
            # # start_x = end_x - 1800
            # start_x = end_x - self.tail_w
            # end_y = self.origin_y + self.tail_y_offset
            # start_y = end_y - self.tail_h
            # image = image[start_y:end_y, start_x:end_x]
            # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            region = (self.tail_start_y, self.tail_end_y, self.tail_start_x, self.tail_end_x)
            y1, y2, x1, x2 = region
            roi = image[y1:y2, x1:x2]
            
            # 图像预处理
            roi = cv2.copyMakeBorder(roi, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
            roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.resize(roi, (800, 1300))

        cv2.imwrite(img_name, image)
        print("图片保存成功！")

    def start_corner_p_detection(self):
        self.detection_count = 0 
        self.corner_p_detect()

    def corner_p_detect(self):
        self.DC_save_path = 'C:/Data/thor_kepoint/infer_result/DC_result'
        self.DC_save_path = self.DC_save_path
        if not os.path.exists(self.DC_save_path):
            os.makedirs(self.DC_save_path, exist_ok=True)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        if self.camera is not None and self.detection_count < 25:
                ret, frame = self.camera.read()
                c,r = detect_circle(frame)
                if c is not None:
                    cv2.circle(frame, (c[0], c[1]), 5, (0, 0, 255), 5)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
                    cv2.destroyAllWindows()
                    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                    cv2.imwrite(f'{self.DC_save_path}/{time_str}.png', frame)
                    with open('corner_point.txt', 'a') as f:
                        f.write(f'{int(c[0])},{int(c[1])}\n')
                    
                    self.detection_count += 1
                    # time.sleep(0.1)
                    self.corner_p_detect() 
                else:
                    self.detection_count += 1
                    self.corner_p_detect()                      
        else:
            self.calculate_average_coordinates() # 将基点坐标进行更新
   
    def calculate_average_coordinates(self):

        # 文件复制
        backup_path = self.config_path.replace('.yaml', '_backup.yaml')
        shutil.copy(self.config_path, backup_path)

        coordinates = []
        with open('corner_point.txt', 'r') as f:
            for line in f:
                x, y = map(int, line.strip().split(','))
                coordinates.append((x, y))

        if coordinates:
            avg_x = sum(x for x, _ in coordinates) / len(coordinates)
            avg_y = sum(y for _, y in coordinates) / len(coordinates)
            # points = {'point': (avg_x, avg_y)}
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)

            # config_data['base_point'] = [avg_x, avg_y]
            config_data['base_point'] = [avg_x, avg_y]
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            QMessageBox.information(self, '提示', '基点检测完成') 
        else:
            QMessageBox.warning(self, '警告', '未检测到有效基点',QMessageBox.Yes) 
            print('未检测到任何坐标')

    def calib_pixel_to_real(self):
        
        ret, frame = self.camera.read()
        factor = calculate_distance(frame)

        backup_path = self.config_path.replace('.yaml', '_backup.yaml')
        shutil.copy(self.config_path, backup_path)        

        with open(self.config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
            
        if factor:
            config_data['factor'] = factor
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)   
            QMessageBox.information(self, '提示', '转换系数检测完成') 
        else:
            QMessageBox.warning(self, '警告', '转换系数检测有误',QMessageBox.Yes) 
            print('未检测到任何坐标')

    def closeEvent(self, event):
        # self.info_listener.stop()
        # self.error_listener.stop()
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        if self.checkbox.isChecked():
            self.preview_timer.stop()
        if self.inference_button.isChecked():
            self.infer_timer.stop()
        if self.camera:
            self.camera.release()
        if self.serial_listener:
            self.serial_listener.stop()
            self.ser.close()
        event.accept()


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
