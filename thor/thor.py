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
                            QMessageBox,QApplication, QSizePolicy, QLineEdit, QDialog)
import onnxruntime as rt
import yaml
import cv2
from utils.detect_circle import detect_circle
from utils.irregular_im_de_simple import get_kde_x, get_kde_y
from utils.get_max_contour import get_max_contour
from utils.logger import setup_logging
from utils.calculate_distance import calculate_distance
# from utils.x_front_rear_min_diff import OffsetBalancer
# from utils.area_min_offset import FabricAligner
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
from threading import Thread, Lock

VAIL_MODEL = True
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

class CameraCaptureThread(Thread):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.latest_frame = None
        self.running = True
        self.lock = Lock()

    def run(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False

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

class SerialListener(QThread):
    '''
    持续监听串口数据, 只发送满足条件的帧数据
    '''
    serial_signal = pyqtSignal(str)
    def __init__(self,ser, loger):
        super().__init__()
        self.ser = ser
        self.running = True
        self.loger = loger
    def run(self):
        try:
            self.ser.reset_input_buffer() #  增加清理输入缓存的操作
            while self.running:
                if self.ser.in_waiting > 0:
                    self.data = self.ser.read(20)
                    # self.data = self.ser.read_all()
                    cmd_id = self.unpack_frame()
                    if cmd_id == 0x0100:
                        data_ = 'head'
                        self.serial_signal.emit(data_) # 发射信号

                    elif cmd_id == 0x0104:
                        data_ = 'tail'
                        self.serial_signal.emit(data_) # 发射信号

        except serial.SerialException as e:
            print(f'串口错误：{e}')

    def unpack_frame(self):
        # 检查 SOF
        if self.data[0] != 0xA5:
            self.loger.error(f'SOF 错误：{self.data[0]}')
            # raise ValueError("SOF 不正确")
            return -1

        # 解包 cmd_id 和 data_length
        cmd_id_bytes = self.data[1:3]  # 提取 2 字节的 cmd_id（小端序）
        cmd_id = int.from_bytes(cmd_id_bytes, 'little')  # 转换为整数（小端序到整数）
        # cmd_id_hex = format(cmd_id, '04x')  # 格式化为四位十六进制字符串
        return cmd_id
       
        '''暂时不检查数据长度， 出现a50001040d的情况'''
        data_length = self.data[3]  # 提取 1 字节的 data_length

        # 检查 data_length 是否正确
        if len(self.data) - 4 != data_length:
            self.loger.error(f'数据长度错误：{self.data.hex()}')
            # raise ValueError("data_length 不正确")

        # 解包 data
        data_payload = self.data[4:4 + data_length]
        alignment_count = struct.unpack('<I', data_payload)[0]
        # self.loger.error(f'对齐次数： {alignment_count}')
        print(f'对齐次数： {alignment_count}')

        return cmd_id
    
        if cmd_id in [0x0100, 0x0104]:
            # 对于命令 0x0100 和 0x0104，data 部分是一个 uint32 类型的数字
            if data_length != 4:
                raise ValueError("命令 0x0100 和 0x0104 的 data_length 应该为 4")
            alignment_count = struct.unpack('<I', data_payload)[0]
            return cmd_id, alignment_count
        
    def stop(self):
        self.running = False
        self.quit()
        self.wait()

def infer_frame(image, sess, input_name, output_names):
    img = np.asarray([image]).astype(np.float32)
    detection = sess.run(output_names, {input_name: img})
    return detection

def infer_segm(image, sess, input_name, output_names):
    frame_se = cv2.resize(image, (384, 384))
    frame_se = np.ascontiguousarray(frame_se[:, :, ::-1].transpose(2, 0, 1))
    img = np.asarray([frame_se]).astype(np.float32)
    detection = sess.run(output_names, {input_name: img})[0]
    return detection


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

        self.log_file = 'thor_log.log'
        self.loger = setup_logging(self.log_file)
        self.config_path = 'profiles//config.yaml'
        self.backup_dir = Path('profiles') / 'backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_config()
        self.delete_old_backups()

        self.camera = None
        self.ser = None
        self.segm_infer = True

        # 初始化线程池
        self.thread_pool = QThreadPool.globalInstance()

        # 并行启动初始化任务
        self.start_paraller_init()

        self.camera_capture_thread = CameraCaptureThread(self.camera)
        self.camera_capture_thread.start()

        # 初始化ui
        # self.button_clicked = False
        if VAIL_MODEL:
            self.inference_count = 0
            self.ser_infer_count = 0
            self.excel_data = []

        self.init_ui()

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

    # def toggle_button_color(self):
    #     # 切换按钮状态
    #     self.button_clicked = not self.button_clicked

    #     if self.button_clicked:
    #         # 设置按钮为点击后的样式表
    #         self.ser_inference_button.setStyleSheet("""
    #             QPushButton {
    #                 background-color: #4CAF50; /* 点击后的背景颜色 */
    #                 color: white; /* 文字颜色 */
    #                 border: none;
    #                 border-radius: 4px;
    #                 padding: 10px 20px;
    #                 text-align: center;
    #             }
    #         """)
    #     else:
    #         # 设置按钮为默认样式表
    #         self.ser_inference_button.setStyleSheet("""
    #             QPushButton {
    #                 background-color: white; /* 默认背景颜色 */
    #                 color: black; /* 文字颜色 */
    #                 border: none;
    #                 border-radius: 4px;
    #                 padding: 10px 20px;
    #                 text-align: center;
    #             }
    #         """)
    def setup_layout(self):
        """布局设置"""
        # 图像布局
        self.hbox_labels = QHBoxLayout()
        self.hbox_labels.addWidget(self.image_label)
        self.hbox_labels.addWidget(self.label_head)
        self.hbox_labels.addWidget(self.label_tail)
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

        # 设置中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

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
        
        # tail
        self.tail_x_offset = self.config['tail_roi']['x_offset']
        self.tail_w = self.config['tail_roi']['w']
        self.tail_h = self.config['tail_roi']['h']
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
        # 串口信号信息
        # 前部
        cmd_id_x_offset = 0x010C  # 发送前部坐标
        self.sof = b'\xA5'
        self.cmd_id_bytes = struct.pack('<H', cmd_id_x_offset)
        # 发送移动指令
        cmd_id_head_align = 0x0101 
        self.cmd_id_bytes_f = struct.pack('<H', cmd_id_head_align)
        self.data_f = b'\x01'
        # 发送错误指令
        cmd_id_error = 0x0001
        self.cmd_id_error_bytes = struct.pack('<H', cmd_id_error)
        self.data_f_error = b'\x00'
        self.error_cmd = self.sof + self.cmd_id_error_bytes + self.data_f_error

        # 尾部
        cmd_id_tail_offset = 0x010D # 发送尾部坐标
        self.cmd_id_bytes_ = struct.pack('<H', cmd_id_tail_offset)
        # 发送移动指令
        hex_data_f = f'A5 05 01 04 00 00 00 00'
        self.byte_data_f = bytes.fromhex(hex_data_f.replace(' ', ''))
        # 发送错误指令
        hex_data1 = f'A5 01 00 00'
        self.byte_data = bytes.fromhex(hex_data1.replace(' ', ''))    

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
        self.enable_all_buttons()
        self.start_preview()

    def on_serial_ready(self, ser):
        self.ser = ser
        # print(self.ser)
        self.serial_listener = SerialListener(self.ser, self.loger)
        self.serial_listener.serial_signal.connect(self.ser_infer)
        self.serial_listener.start()
    
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
            self.excel_data = []
            self.infer_timer.start(30)  
            self.infer_running = True
            self.inference_button.setText("停止推理")

    def update_preview(self):
        if self.camera and self.camera.isOpened():
            # ret, frame = self.camera.read()
            frame = self.camera_capture_thread.get_latest_frame()
            # if ret:
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_image = cv2.resize(rgb_image, (960, 720))
                # h, w, ch = rgb_image.shape
                # bytes_per_line = ch * w
                rgb_image = QImage(rgb_image, 960, 720, QImage.Format_RGB888)
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
                optimizer = AlignmentOptimizer(x_coords, x_coords, self.roi_rotate_base_p_x, self.roi_tail_rotate_base_p_x)
                optimal_head, optimal_tail, circle_head_x, circle_x,circle_head_x_, circle_x_  = optimizer.find_optimal_split()
                self.write_to_excel(str(color),self.inference_count, optimal_head)
                self.inference_count += 1


            pts = pts.astype(np.int32)
            for i, pt in enumerate(pts):
                cv2.circle(frame, tuple(pt.astype(np.int32)), 5, color_list[i], 5)
     
            frame = cv2.addWeighted(frame, 0.8, pred_bgr, 0.5, 0)

            roi_img_cp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi_img_cp = cv2.resize(roi_img_cp, (960, 720))
            # print(roi_img_cp.shape[1], roi_img_cp.shape[0])
            image = QImage(roi_img_cp, 960, 720, QImage.Format_RGB888)
            # 调整图像大小以适应标签尺寸
            if self.front_radio.isChecked():
                scaled_image = image.scaled(self.label_head.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_head.setPixmap(QPixmap.fromImage(scaled_image))
            else:
                scaled_image = image.scaled(self.label_tail.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_tail.setPixmap(QPixmap.fromImage(scaled_image))

            
    def write_to_excel(self, name, count, result):
        '''
        将推理结果和次数写入Excel
        '''
        self.excel_data.append([count, result])

        # 每次写入都保存Excel文件
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
        self.Data_save_path = self.config['origin_im_save_path']
        if not os.path.exists(self.Data_save_path):
            os.makedirs(self.Data_save_path, exist_ok=True)

        self.label_save_path = self.config['origin_label_path']
        if not os.path.exists(self.label_save_path):
            os.makedirs(self.label_save_path, exist_ok=True)

        self.result_save_path_head = self.config['infer_head_im_save_path']
        if not os.path.exists(self.result_save_path_head):
            os.makedirs(self.result_save_path_head, exist_ok=True)

        self.result_save_path_tail = self.config['infer_tail_im_save_path']
        if not os.path.exists(self.result_save_path_tail):
            os.makedirs(self.result_save_path_tail, exist_ok=True)


        self.detect_error_save_path = 'Data/thor_kepoint/error_detect'
        self.detect_error_save_path = self.detect_error_save_path
        if not os.path.exists(self.detect_error_save_path):
            os.makedirs(self.detect_error_save_path, exist_ok=True)

        self.save_path = 'Data/thor_keypoint_data'
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
        # print('roi shape is: ', roi.shape)
        resized = cv2.resize(roi, (800, 1300))
        frame_as = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)) 
        detection, segm_pred = self.parallel_inference(frame_as, roi) 
        return roi, resized, detection, segm_pred
    
    def parallel_inference(self, frame_as, roi):
        """并行执行检测与分割推理"""
        detection, segm_pred = [], []
        t_detect = Thread(target=lambda: detection.append(infer_frame(frame_as, self.sess, self.input_name, self.output_names)))
        t_segm = Thread(target=lambda: segm_pred.append(infer_segm(roi, self.segm_sess, self.segm_input_name, self.segm_output_names))) if self.segm_infer else None
        t_detect.start()
        if t_segm: t_segm.start()
        t_detect.join()
        if t_segm: t_segm.join()
        return detection, segm_pred 
    
    def ser_infer(self, data):
        '''
        优化y
        前部只需要获取到x坐标点集
        尾部优化x坐标之后， 在优化y
        '''
        if data == 'head':
            self.loger.info(f'前部 收到串口数据：{data}')
            time_s = time.time()
        
            ret, frame = self.camera.read()
            data_frame_f = self.sof + self.cmd_id_bytes_f + bytes([len(self.data_f)]) + self.data_f
            self.loger.info(f'前部 发送前部对齐指令: {data_frame_f}')
            self.ser.write(data_frame_f) 

            region = (self.start_y, self.end_y, self.start_x, self.end_x)
            self.head_frame, self.head_frame_re, detection, segm_pred = self.process_frame(frame, region, self.pad)

            pred = np.where(segm_pred[0].squeeze() > 0, 255, 0).astype(np.uint8)
            self.head_pred = cv2.resize(pred, (self.head_frame.shape[1],self.head_frame.shape[0]))
            self.head_max_contour = get_max_contour(self.head_pred)

            self.head_pred_bgr = cv2.cvtColor(self.head_pred, cv2.COLOR_GRAY2BGR)

            color_list = [[255, 0, 0], [0,255,0]]
            pts = detection[0][0][0].astype(np.float64)
            pts[:, 1] *= self.factor_w_offset
            # 判断检测是否有问题
            # if pts.ndim == 2 and np.all(pts > 0) and 950 < np.linalg.norm(pts[0] - pts[1]) < 1900: 
            if pts.ndim == 2 and np.all(pts > 0) and np.all(pts[:, 0]>190) and np.all(pts[:, 0]<613) and abs(pts[0][0]) - abs(pts[1][0]) < self.x_diff: 

                self.loger.info('前部检测正常')
                pts_y = pts[pts[:, 1].argsort()]  # 按y轴排序
                circle_x = pts_y[0][0]
                # self.circle_head_y = pts_y[0][1]

                # pts = pts.astype(np.int32)
                self.head_x_coords, self.points_between_head = get_kde_x(self.head_pred, self.head_max_contour, pts,False) # 返回边缘x坐标列表， 用于与尾部计算偏移

                '''需要验证在尾部时该值是否会发生变化'''
                # self.head_x_coords_ = [x - self.roi_rotate_base_p_x for x in self.head_x_coords] # 将所有x坐标与基点计算偏移
                # self.head_p_y =  self.circle_head_y - self.roi_rotate_base_p_y

                self.loger.info(f'前部花费总时间： {time.time() - time_s}s')

                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.Data_save_path}/{time_str}_start.png',self.head_frame_re)
                with open (f'{self.label_save_path}/{time_str}_start.txt', 'w', encoding='utf-8') as f:
                    for pt in pts:
                        f.write(f"{int(pt[0])},{int(pt[1])}\n")
            else:
                self.loger.info('前部检测有误， 请检查图像')
                self.ser.write(self.error_cmd)
                head_x = 0
                head_y = 0
                circle_x = 0
                circle_y = 0
                # 取消图像保存
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', self.head_frame_re)  

        elif data == 'tail':
            self.loger.info(f'尾部收到串口数据：{data}')
            time_read_frame = time.time()
            ret, frame = self.camera.read()
            region = (self.tail_start_y, self.tail_end_y, self.tail_start_x, self.tail_end_x)
            roi_tail, self.tail_frame_re, detection, segm_pred = self.process_frame(frame, region, 0)

            pred = np.where(segm_pred[0].squeeze() > 0, 255, 0).astype(np.uint8)
            pred = cv2.resize(pred, (roi_tail.shape[1], roi_tail.shape[0]))
            # print('pred shape is:', pred.shape)
            max_contour = get_max_contour(pred)

            tail_pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        
            color_list = [[0, 0, 255], [0,255,0]]
            pts = detection[0][0][0].astype(np.float64)
        
            pts[:, 0] *= self.tail_factor_h_offset 
            pts[:, 1] *= self.tail_factor_w_offset 
            pts_y = pts[pts[:, 1].argsort()]
            '''
            将计算长度改为是否有角点靠近灰色边缘
            '''
            if pts.ndim == 2 and np.all(pts > 0) and abs(pts[0][0]) - abs(pts[1][0]) < self.x_diff:
                circle_x = pts_y[0][0]
                circle_y = pts_y[0][1]

                # pts = pts.astype(np.int32)
                tail_x_coords, points_between_tail = get_kde_x(pred, max_contour, pts, False)
                    
                # 进行旋转
                # tail_x_coords_ = [x - self.roi_tail_rotate_base_p_x for x in tail_x_coords] # 将所有x坐标与基点计算偏移
                # print('self.head_x_coords', self.head_x_coords)
                if self.head_x_coords is not None:
                    optimizer = AlignmentOptimizer(self.head_x_coords, tail_x_coords, self.roi_rotate_base_p_x, self.roi_tail_rotate_base_p_x)
                    optimal_head, optimal_tail, circle_head_x, circle_x,circle_head_x_, circle_x_  = optimizer.find_optimal_split()
                    # optimizer.visualize()
                else:
                    '''这里是不是要发送检测错误的信号'''
                    self.loger.error('视觉检测有误')
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
                p1, p2, head_num = get_kde_y(self.head_pred, circle_head_x, self.head_max_contour, self.x_border, 'head', None, True)
                p3, p4, tail_num = get_kde_y(pred, circle_x, max_contour, self.x_border, 'tail', head_num, True)
                p1_y = p1[:, 1] 
                p2_y = p2[:, 1] # 上边缘
                p3_y = p3[:, 1] 
                p4_y = p4[:, 1] # 上边缘

                p2_y_num = len(p2_y)
                p4_y_num = len(p4_y)
                min_num = min(p2_y_num, p4_y_num)
                p2_y = p2_y[:min_num]
                p4_y = p4_y[:min_num]
                p1_y = p1_y[:min_num]
                p3_y = p3_y[:min_num]
                if len(p2_y) != len(p4_y):
                    print('获取到的y值数量不等')
                    return
                # 计算前部和尾部的y轴偏移
                top_y_offset = p1_y - p3_y  # 如果为负 说明上边缘 前部盖过尾部
                print('top_y_offset', top_y_offset)
                print('np.sum(top_y_offset', np.sum(top_y_offset))
                bottom_offset = p4_y - p2_y # 如果为负 说明前部盖过尾部
                print('bottom_offset', bottom_offset)
                print('np.sum(bottom_offset', np.sum(bottom_offset))

                total_y_mean_offset = (np.sum(top_y_offset) - np.sum(bottom_offset)) / (2 * len(top_y_offset))
                print('total_y_mean_offset', total_y_mean_offset)
                
                # 找到与平均值最近的p1_y
                p2_y_mean = np.mean(p2_y)
                p2_y_diff = p2_y - p2_y_mean
                p2_y_diff_abs = np.abs(p2_y_diff)
                p2_y_diff_min_idx = np.argmin(p2_y_diff_abs)
                p2_y_min = p2_y[p2_y_diff_min_idx]
                p4_y_min = p4_y[p2_y_diff_min_idx]
                offs = p2_y_min - p4_y_min
                print('p4_y_min', p4_y_min)
                p2_final_y = p2_y_min + total_y_mean_offset - offs
                head_p_y = p2_final_y - self.roi_rotate_base_p_y
                print('head_p_y', head_p_y)

                #################### 前部信息 ###############################
                self.ser.reset_input_buffer() # 清理缓存
                try:
                    head_x = round(optimal_head * self.factor, 2) # 对x坐标进行优化
                    # head_y = round(self.head_p_y * self.factor, 2)
                    head_y = round(head_p_y * self.factor, 2)
                    self.loger.info(f'前部与基点的偏移量：{head_x}, {head_y}')

                    x_bytes = struct.pack('<f', head_x)
                    y_bytes = struct.pack('<f', head_y)
                    data_length = len(x_bytes) + len(y_bytes)

                    if data_length != 8:
                        self.loger.error(f'前部 字节数据长度不匹配， 实际长度：{data_length}')

                    data_frame = self.sof + self.cmd_id_bytes + bytes([data_length]) + x_bytes + y_bytes
                    self.ser.write(data_frame) 
                    time.sleep(0.05)
                except Exception as e:
                    self.loger.error(f'前部 发送数据错误：: {e}')
                #################### 前部信息 ###############################

                # tail_p_x = circle_x - self.roi_tail_rotate_base_p_x
                # tail_p_y = circle_y - self.roi_tail_rotate_base_p_y
                tail_p_y = p4_y_min - self.roi_tail_rotate_base_p_y

                tail_x = round(optimal_tail * self.factor, 2)  # 对尾部x坐标进行优化
                tail_y = round(tail_p_y * self.factor, 2)
                self.loger.info(f'尾部与基点的偏移量：{tail_x}, {tail_y}')

                if VAIL_MODEL:
                    '''
                    同时将前部和尾部的x偏移写入表格
                    '''
                    result = (str(head_x), str(tail_x))
                    name_text = self.nametext.text()
                    self.write_to_excel(name_text, self.ser_infer_count, result)
                    self.ser_infer_count += 1
                    if self.ser_infer_count == 100:
                        self.excel_data = []
                        print('已经完成检测100次')
                        return 

                tail_x_bytes = struct.pack('<f', tail_x)
                tail_y_bytes = struct.pack('<f', tail_y)
                tail_data_length = len(tail_x_bytes) + len(tail_y_bytes)

                if tail_data_length != 8:
                    print(f'字节数据长度不匹配， 实际长度：{tail_data_length}')
                    
                tail_data_frame = self.sof + self.cmd_id_bytes_ + bytes([tail_data_length]) + tail_x_bytes + tail_y_bytes
                self.ser.write(tail_data_frame) 
                time.sleep(0.01)
                self.ser.write(self.byte_data_f)
                self.loger.info(f'尾部发送尾部对齐指令：{tail_data_frame}')
                self.loger.info(f'尾部花费总时间：{time.time() - time_read_frame}s')
                # print(f'尾部花费总时间 {time.time() - time_read_frame}s')

                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.Data_save_path}/{time_str}_end.png', self.tail_frame_re)
                with open (f'{self.label_save_path}/{time_str}_end.txt', 'w', encoding='utf-8') as f:
                    for pt in pts:
                        f.write(f"{int(pt[0])},{int(pt[1])}\n")

                cv2.circle(self.head_frame, (int(circle_head_x), int(p2_final_y)), 2, color_list[0], 5)
                cv2.line(self.head_frame, (int(circle_head_x), int(p2_final_y)), (int(circle_head_x), int(p2_final_y)+1000), color_list[0], 2)
                cv2.line(self.head_frame, (int(circle_head_x_), int(p2_final_y)), (int(circle_head_x_), int(p2_final_y)+1000), color_list[1], 2)
                # cv2.line(self.head_frame, (int(circle_head_x), int(head_p_y)), (int(circle_head_x)+200, int(head_p_y)), color_list[0], 2)
                head_frame_save = cv2.resize(self.head_frame, (800, 1300)) 
                cv2.imwrite(f'{self.result_save_path_head}/{time_str}_{head_x}_{head_y}_start.png', head_frame_save)

                cv2.circle(roi_tail, (int(circle_x), int(p4_y_min)), 2, color_list[0], 5)
                cv2.line(roi_tail, (int(circle_x), int(p4_y_min)), (int(circle_x), int(p4_y_min)+2000), color_list[0], 2)
                cv2.line(roi_tail, (int(circle_x_), int(p4_y_min)), (int(circle_x_), int(p4_y_min)+2000), color_list[1], 2)
                # cv2.line(roi_tail, (int(circle_x)-200, int(p4_y_min)), (int(circle_x), int(p4_y_min)+2000), color_list[0], 2)
                frame_save = cv2.resize(roi_tail, (800, 1300))
                cv2.imwrite(f'{self.result_save_path_tail}/{time_str}_{tail_x}_{tail_y}_end.png', frame_save)

                self.show_windows_im(self.head_frame, self.head_pred_bgr, self.label_head)
                self.show_windows_im(roi_tail, tail_pred_bgr, self.label_tail)

                if VAIL_MODEL:
                    mask = np.zeros_like(pred, dtype=np.uint8)
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    print('mask shape is:', mask.shape)
                    for coord in points_between_tail:
                        cv2.circle(mask, (int(coord[0]), int(coord[1])), 3, (0, 0, 255), -1)
                    abs_diff = int(abs(optimal_tail) - abs(optimal_head))
                    for coord in self.points_between_head:
                        cv2.circle(mask, (int(coord[0] + abs_diff), int(coord[1])), 3, (0, 255, 0), -1)
                    rgb_mask = cv2.resize(mask, (960, 720))
                    image_ = QImage(rgb_mask, 960, 720, QImage.Format_RGB888)
                    scaled_image = image_.scaled(self.label_align.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.label_align.setPixmap(QPixmap.fromImage(scaled_image))

                self.head_frame = None
                self.circle_head_y = None
                # self.head_x_coords_ = None
                self.head_x_coords = None

            else:
                self.loger.error('尾部检测有误， 请检查图像')
                tail_x = -1
                tail_y = -1
                self.ser.write(self.byte_data)
                # 取消图像保存
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', self.tail_frame_re)
    
    def show_windows_im(self, im, pred_bgr, im_label):
        frame_cp = cv2.addWeighted(im, 0.8, pred_bgr, 0.5, 0)
        # img_rgb = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(frame_cp, (960, 720))
        image = QImage(img_rgb, 960, 720, QImage.Format_RGB888)
        scaled_image = image.scaled(im_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        im_label.setPixmap(QPixmap.fromImage(scaled_image))

    def save_image(self):

        str_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.png'
        img_name = os.path.join(self.save_path, str_time)
        ret, image = self.camera.read()

        if not ret:
            print('camera init error')

        # 前部
        if self.config['save_im_way']['head']:
            end_x = self.origin_x + self.head_x_offset
            start_x = end_x - self.head_w
            start_y = self.origin_y + self.head_y_offset
            end_y = start_y + self.head_h
            image = image[start_y:end_y, start_x:end_x]
            image = cv2.copyMakeBorder(image, 250, 250, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.resize(image, (800, 1300))

        elif self.config['save_im_way']['full_im']:
            image = image
        else:
            # 尾部
            end_x = self.origin_x + self.tail_x_offset
            # start_x = end_x - 1800
            start_x = end_x - self.tail_w
            end_y = self.origin_y + self.tail_y_offset
            start_y = end_y - self.tail_h

            image = image[start_y:end_y, start_x:end_x]
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            image = cv2.resize(image, (800, 1300))

        cv2.imwrite(img_name, image)
        print("图片保存成功！")

    def start_corner_p_detection(self):
        self.detection_count = 0 
        self.corner_p_detect()

    def corner_p_detect(self):
        self.DC_save_path = 'Data/thor_kepoint/infer_result/DC_result'
        self.DC_save_path = self.DC_save_path
        if not os.path.exists(self.DC_save_path):
            os.makedirs(self.DC_save_path, exist_ok=True)

        if self.camera is not None and self.detection_count < 25:
                ret, frame = self.camera.read()
                c,r = detect_circle(frame)
                if c is not None:
                    cv2.circle(frame, (c[0], c[1]), 5, (0, 0, 255), 5)
                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
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
        self.camera_capture_thread.stop()
        self.camera_capture_thread.join()
        event.accept()


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
