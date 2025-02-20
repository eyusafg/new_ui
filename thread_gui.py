import os
import time
import numpy as np
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
import serial
import serial.tools.list_ports
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QCheckBox, QRadioButton, 
                            QMessageBox)
import onnxruntime as rt
import yaml
import cv2
from utils.detect_circle import detect_circle
from utils.irregular_im_de import get_kde
from utils.calculate_distance import calculate_distance
import struct
import shutil
from PyCameraList.camera_device import list_video_devices

'''
pyinstaller --onedir thor.py -c  --hidden-import encodings  --hidden-import codecs  --hidden-import io  --hidden-import _io  --hidden-import zipimport  --paths "thor_pk\Lib\site-packages"  --add-data "thor_pk\Lib\encodings;encodings"  --add-data "thor_pk\Lib\codecs.py;."  --add-data "thor_pk\Lib\io.py;."  --hidden-import start --hidden-import pyimod02_importers --hidden-import jinja2 --hidden-import sip
'''
        
def Scan_serial():
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) == 0:
        print('无可用串口')
        ser = -1
    else:
        if len(port_list) == 1:
            port = port_list[0].device
            ser = serial.Serial(port, 115200, bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE, 
                                parity=serial.PARITY_NONE, timeout=0.005)
        else:
            ser = -2
    return ser

class SerialListener(QThread):
    serial_signal = pyqtSignal(str)
    def __init__(self,ser):
        super().__init__()
        self.ser = ser
        self.running = True
    def run(self):
        try:
            self.ser.reset_input_buffer() #  增加清理输入缓存的操作
            while self.running:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(20)

                    if data.hex()[2:4] in ['00', '04', '0e', '0f']:
                        self.serial_signal.emit(data.hex()[2:]) # 发射信号
        except serial.SerialException as e:
            print(f'串口错误：{e}')
    def stop(self):
        self.running = False
        self.quit()
        self.wait()


def infer_frame(image, sess, input_name, output_names):
    img = np.asarray([image]).astype(np.float32)
    detection = sess.run(output_names, {input_name: img})
    return detection

def infer_segm(image, sess, input_name, output_names):
    img = np.asarray([image]).astype(np.float32)
    detection = sess.run(output_names, {input_name: img})[0]
    return detection


class InitWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

    def run(self):
        try:
            result = {}

            # 加载配置
            self.progress.emit("正在加载配置文件...")
            config = self.load_config()
            result['config'] = config

            # self.progress.emit("正在初始化相机...")
            # camera = self.init_camera()
            # result['camera'] = camera
            
            # 加载模型
            self.progress.emit("正在加载模型...")
            sess, segm_sess = self.load_models(config)
            
            # 返回结果
            result.update({
                'sess': sess,
                'segm_sess': segm_sess,
                'input_name': sess.get_inputs()[0].name,
                'output_names': [out.name for out in sess.get_outputs()],
                'segm_input': segm_sess.get_inputs()[0].name,
                'segm_outputs': [out.name for out in segm_sess.get_outputs()]
            })
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"初始化失败: {str(e)}")

    def load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
        
    def init_camera(self):
        """延迟初始化相机"""
        try:
            # 初始化相机为 None
            camera_list = list_video_devices()
            device_dict = {camera[1]: camera[0] for camera in camera_list if not any(keyworld.lower() in camera[1].lower() for keyworld in ['obs', 'webcam'])}
            if len(device_dict) == 1: 
                camera_index = list(device_dict.values())[0]
                camera = cv2.VideoCapture(camera_index)  
                if not camera.isOpened():
                    raise RuntimeError("无法打开相机设备")

                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                camera.set(cv2.CAP_PROP_FPS, 20)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2880)
                return camera

        except Exception as e:
            raise RuntimeError(f"相机初始化失败: {str(e)}")

    def load_models(self, config):
        # 加载关键点模型
        if not os.path.exists(config['model_path']):
            raise FileNotFoundError(f"关键点模型不存在: {config['model_path']}")
        
        # 加载分割模型
        if not os.path.exists(config['segm_model_path']):
            raise FileNotFoundError(f"分割模型不存在: {config['segm_model_path']}")
        
        so = rt.SessionOptions()
        sess = rt.InferenceSession(
            config['model_path'], 
            so, 
            providers=['CPUExecutionProvider']
        )
        
        so_segm = rt.SessionOptions()
        segm_sess = rt.InferenceSession(
            config['segm_model_path'],
            so_segm,
            providers=['CPUExecutionProvider']
        )
       
        return sess, segm_sess

class CameraInitWorker(QObject):
    finished = pyqtSignal(cv2.VideoCapture)
    error = pyqtSignal(str)
    def __init__(self):
        super().__init__()

    def run(self):
        try:
            camera = self.init_camera()
            self.finished.emit(camera)
        except Exception as e:
            self.error.emit(f"相机初始化失败: {str(e)}")

    def init_camera(self):
        """延迟初始化相机"""
        try:
            # 初始化相机为 None
            print(11111)
            # camera_list = list_video_devices()
            # device_dict = {camera[1]: camera[0] for camera in camera_list if not any(keyworld.lower() in camera[1].lower() for keyworld in ['obs', 'webcam'])}
            # if len(device_dict) == 1: 
                # camera_index = list(device_dict.values())[0]
            camera = cv2.VideoCapture(1)  
            print(2222)
            if not camera.isOpened():
                raise RuntimeError("无法打开相机设备")

            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            camera.set(cv2.CAP_PROP_FPS, 20)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2880)
            print(33333)
            return camera

        except Exception as e:
            raise RuntimeError(f"相机初始化失败: {str(e)}")
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera = None
        self.ser = None
        self.serial_listener = None
        self.config_path = 'profiles//config.yaml'
        self.setup_loading_ui()
        self.start_async_init()
        self.start_async_camera_init()

    def setup_loading_ui(self):
        """初始加载界面"""
        self.setWindowTitle("Camera Viewer - 加载中...")
        self.resize(400, 200)
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        self.loading_label = QLabel("正在初始化，请稍候...", self)
        self.progress_label = QLabel("", self)  # 新增进度标签
        
        for lbl in [self.loading_label, self.progress_label]:
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
        
        central_widget.setLayout(layout)

    def start_async_init(self):
        """启动异步初始化"""
        self.init_thread = QThread()
        self.init_worker = InitWorker(self.config_path)
        self.init_worker.moveToThread(self.init_thread)
        
        # 连接信号
        self.init_worker.finished.connect(self.on_init_success)
        self.init_worker.error.connect(self.on_init_error)
        self.init_worker.progress.connect(self.update_progress)  # 连接进度信号
        self.init_thread.started.connect(self.init_worker.run)
        
        self.init_thread.start()

    def start_async_camera_init(self):
        """启动相机异步初始化"""
        self.camera_init_thread = QThread()
        self.camera_init_worker = CameraInitWorker()
        self.camera_init_worker.moveToThread(self.camera_init_thread)
        
        # 连接信号
        self.camera_init_worker.finished.connect(self.on_camera_init_success)
        self.camera_init_worker.error.connect(self.on_camera_init_error)
        self.camera_init_thread.started.connect(self.camera_init_worker.run)
        
        self.camera_init_thread.start()

    def update_progress(self, message):
        """更新加载进度"""
        self.progress_label.setText(message)
        QApplication.processEvents()  # 强制刷新UI

    def on_init_success(self, result):
        """初始化成功处理"""
        try:
            self.segm_infer = False
            # 保存初始化结果
            self.config = result['config']
            # self.camera = result['camera']

            self.base_point = self.config['base_point'] 
            self.origin_x = self.base_point[0]
            self.origin_y = self.base_point[1]   
            self.coordinate = [self.origin_x, self.origin_y]
        
            self.factor = self.config['factor']

            self.head_x_offset = self.config['head_roi']['x_offset']
            self.head_w = self.config['head_roi']['w']
            self.head_h = self.config['head_roi']['h']
            self.head_y_offset = self.config['head_roi']['y_offset']
            # tail
            self.tail_x_offset = self.config['tail_roi']['x_offset']
            self.tail_w = self.config['tail_roi']['w']
            self.tail_h = self.config['tail_roi']['h']
            self.tail_y_offset = self.config['tail_roi']['y_offset']


            self.sess = result['sess']
            self.segm_sess = result['segm_sess']
            self.input_name = result['input_name']
            self.output_names = result['output_names']
            self.segm_input = result['segm_input']
            self.segm_outputs = result['segm_outputs']
            
            # 关闭初始化线程
            self.init_thread.quit()
            self.init_thread.wait()
            
            # 设置主界面
            self.setup_main_ui()
            self.post_init_setup()
            self.init_serial()
            self.disable_all_buttons()

            self.setWindowTitle("Camera Viewer")
        except Exception as e:
            self.show_error_message(f"界面初始化失败: {str(e)}")

    def on_camera_init_success(self, camera):
        # self.progress_label.setText("相机初始化成功")
        # 这里可以将 camera 对象保存起来以供后续使用
        # mesg = "相机初始化成功"
        # self.show_success_message(mesg)
        self.camera = camera
        self.enable_all_buttons() 
        self.camera_init_thread.quit()

    def on_camera_init_error(self, error):
        # self.progress_label.setText(error)
        self.camera_init_thread.quit()
        self.show_error_message(error)
        self.close()


    def on_init_error(self, error_msg):
        """初始化失败处理"""
        self.init_thread.quit()
        self.show_error_message(error_msg)
        self.close()

    def setup_main_ui(self):
        """主界面设置"""
        # 清空加载界面
        self.centralWidget().layout().removeWidget(self.loading_label)
        self.loading_label.deleteLater()
        
        # 创建主界面控件
        self.create_widgets()
        self.setup_layout()
        self.connect_signals()

    def create_widgets(self):
        """创建界面控件"""
        # 图像显示
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        
        self.infer_label = QLabel(self)
        self.infer_label.setAlignment(Qt.AlignCenter)
        self.infer_label.setMinimumSize(320, 240)
        self.infer_label.setVisible(False)
        
        # 按钮
        self.save_button = QPushButton("保存图片")
        self.p_detect_button = QPushButton("基点检测")
        self.factor_detect_button = QPushButton("转换系数检测")
        self.inference_button = QPushButton("单独推理")
        self.ser_inference_button = QPushButton("通信推理")
        
        # 单选/复选框
        self.front_radio = QRadioButton('前部推理')
        self.rear_radio = QRadioButton('尾部推理')
        self.front_radio.setChecked(True)
        
        self.checkbox_infer = QCheckBox("开启推理界面")
        self.checkbox_segm = QCheckBox("使用分割")
        
        self.checkbox = QCheckBox("开启更新画面")
        self.reload_buttons = [
            QRadioButton(text) for text in [
                "重新加载配置文件",
                "重新加载相机",
                "重新加载串口"
            ]
        ]

    def setup_layout(self):
        """布局设置"""
        # 图像布局
        self.hbox_labels = QHBoxLayout()
        self.hbox_labels.addWidget(self.image_label)
        self.hbox_labels.addWidget(self.infer_label)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(self.hbox_labels)
        main_layout.addWidget(self.checkbox_segm)
        main_layout.addWidget(self.checkbox_infer)
        main_layout.addWidget(self.checkbox)
        
        # 添加重新加载按钮
        for btn in self.reload_buttons:
            main_layout.addWidget(btn)
            
        # 添加单选按钮组
        main_layout.addWidget(self.front_radio)
        main_layout.addWidget(self.rear_radio)
        
        # 添加功能按钮
        button_group = [
            self.save_button,
            self.inference_button,
            self.p_detect_button,
            self.factor_detect_button,
            self.ser_inference_button
        ]
        for btn in button_group:
            main_layout.addWidget(btn)
        
        # 设置中心部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def connect_signals(self):
        """连接信号槽"""
        # 按钮点击
        self.save_button.clicked.connect(self.save_image)
        self.p_detect_button.clicked.connect(self.start_corner_p_detection)
        self.factor_detect_button.clicked.connect(self.calib_pixel_to_real)
        self.inference_button.clicked.connect(self.toggle_inference)
        self.ser_inference_button.clicked.connect(self.ser_infer)
        
        # 复选框
        self.checkbox_infer.stateChanged.connect(self.switch_mode)
        self.checkbox_segm.stateChanged.connect(self.is_segm)
        self.checkbox.stateChanged.connect(self.toggle_timer)
        
        # 重新加载按钮
        self.reload_buttons[0].clicked.connect(self.reload_config)
        self.reload_buttons[1].clicked.connect(self.init_camera)
        self.reload_buttons[2].clicked.connect(self.init_serial)

    def post_init_setup(self):
        """后续初始化（非关键）"""
 
        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        self.infer_timer = QTimer(self)
        self.infer_timer.timeout.connect(self.run_inference)
        self.infer_running = False
        

    def init_camera(self):
        """延迟初始化相机"""
        if self.camera is None:
            try:
                # 初始化相机为 None
                self.camera_list = list_video_devices()
                device_dict = {camera[1]: camera[0] for camera in self.camera_list if not any(keyworld.lower() in camera[1].lower() for keyworld in ['obs', 'webcam'])}
                if len(device_dict) == 1: 
                    camera_index = list(device_dict.values())[0]
                    self.camera = cv2.VideoCapture(camera_index)  
                    self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    self.camera.set(cv2.CAP_PROP_FPS, 20)
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2880)
                    # ret, frame = self.camera.read()
                elif len(device_dict) > 1:
                    QMessageBox.warning(self, '错误', '插入多个相机', QMessageBox.Yes)
                else:
                    QMessageBox.warning(self, '错误', '未找到可用的相机!', QMessageBox.Yes)
            except Exception as e:
                self.show_error_message(f"相机初始化失败: {str(e)}")

    def init_serial(self):
        """延迟初始化串口"""
        if self.ser is None:
            try:
                self.ser = Scan_serial()
                if self.ser == -1:
                    QMessageBox.warning(self, '错误', '未找到可用的串口！', QMessageBox.Yes)
                elif self.ser == -2:
                    QMessageBox.warning(self, '错误', '插入多个串口', QMessageBox.Yes)
                else:
                    self.serial_listener = SerialListener(self.ser)
                    self.serial_listener.serial_signal.connect(self.ser_infer)
                    self.serial_listener.start()
            except Exception as e:
                self.show_error_message(f"串口初始化失败: {str(e)}")

    def show_error_message(self, msg):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", msg)
        self.close()
    
    def show_success_message(self, ms):
        QMessageBox.information(self, "提示", ms)

    def ser_infer(self, data):

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
        
        self.result_save_path_compose_frame = 'Data/thor_kepoint/infer_result/compose_frame'
        if not os.path.exists(self.result_save_path_compose_frame):
            os.makedirs(self.result_save_path_compose_frame, exist_ok=True)

        self.detect_error_save_path = 'Data/thor_kepoint/error_detect'
        if not os.path.exists(self.detect_error_save_path):
            os.makedirs(self.detect_error_save_path, exist_ok=True)

        restore_coordinate = None
        if data != False and str(data[0:2]) == '00':
 
            time_s = time.time()
            ret, frame = self.camera.read()

            # ritate
            end_x = self.coordinate[0] + self.head_x_offset
            # end_x = self.coordinate[0] + 646
            # start_x = end_x - 2200
            start_x = end_x - self.head_w
            start_y = self.coordinate[1] + self.head_y_offset
            end_y = start_y + self.head_h
            frame = frame[start_y:end_y, start_x:end_x]
            print(frame.shape)
            
            pad = (800-self.head_h) // 2
            frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_cp = frame.copy()

            roi_rotate_base_p_x = self.origin_y - start_y + pad
            roi_rotate_base_p_y = self.head_w - (self.origin_x - start_x)
            print(f'roi_rotate_base_p_x:{roi_rotate_base_p_x}, roi_rotate_base_p_y:{roi_rotate_base_p_y}')
            
            frame_re = cv2.resize(frame, (800, 1300))
            frame_as = np.ascontiguousarray(frame_re[:, :, ::-1].transpose(2, 0, 1))
            detection = infer_frame(frame_as, self.sess, self.input_name, self.output_names)

            if self.segm_infer:
                print('开始推理分割模型')
                segm_t = time.time()
                frame_size = cv2.resize(frame, (640, 640))
                frame_se = np.ascontiguousarray(frame_size[:, :, ::-1].transpose(2, 0, 1))
                pred = infer_segm(frame_se, self.sess_, self.input_name_, self.output_names_)
                pred = np.where(pred.squeeze() > 0, 255, 0).astype(np.uint8)
                pred = cv2.resize(pred, (frame.shape[1],frame.shape[0]))
                print(f"分割推理时间：{time.time() - segm_t}s")
                pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

            color_list = [[255, 0, 0], [0,255,0]]
            pts = detection[0][0] 
            # pts[:, 1] = pts[:, 1] * (2200 / 1300)
            pts[:, 1] = pts[:, 1] * (self.head_w / 1300)
                        
            circle_x = 0
            circle_y = 0
            is_error = False
            # 判断检测是否有问题
            if pts.ndim == 2 and np.any(pts > 0) and 950 < np.linalg.norm(pts[0] - pts[1]) < 1900: 
            # if pts.ndim == 2 and np.any(pts > 0) and  abs(pts[0][0] - pts[1][0]) < 100 and abs(pts[0][1] - pts[1][1]) > 600: 
                print('检测正常')
                # 检测两个角点正常
                pts_y = pts[pts[:, 1].argsort()]  # 按y轴排序
                circle_x = pts_y[0][0]
                circle_y = pts_y[0][1]
                if self.segm_infer:
                    pts = pts.astype(np.int32)
                    circle_x = int(get_kde(pred, pts))
                    # circle_x = int(circle_x)
                    # cv2.line(frame, (circle_x, pts[0][1]), (circle_x, pts[1][1]), (0, 0, 255), 2)

                # 可能存在异形情况， 暂时不使用
                # circle_x = pts_y[1][0] if pts_y[1][0] - circle_x > 100 else pts_y[0][0]

                # 进行旋转
                head_p_x = circle_x - roi_rotate_base_p_x
                head_p_y = circle_y - roi_rotate_base_p_y
                restore_coordinate = [circle_x, circle_y]
                
            else:
                # 角点检测有问题
                is_error = True
                head_p_x = 0
                head_p_y = 0
                print('检测有误， 请检查图像')

            if not is_error:
                # left_x = round(head_p_x * 0.0604, 2) * -2
                # left_y = round(head_p_y * 0.0604, 2) * 1
                self.ser.reset_input_buffer() # 清理缓存
                try:
                    head_x = round(head_p_x * self.factor, 2) * -2
                    head_y = round(head_p_y * self.factor, 2) * 1
                    print(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
                    print('head_x', head_x, 'head_y', head_y)

                    x_bytes = struct.pack('<f', head_x)
                    y_bytes = struct.pack('<f', head_y)
                    data_length = len(x_bytes) + len(y_bytes)
                    # print(f'x_bytes: {x_bytes}, y_bytes: {y_bytes}')

                    if data_length != 8:
                        print(f'字节数据长度不匹配， 实际长度：{data_length}')

                    cmd_id_x_offset = 0x010C
                    sof = b'\xA5'
                    cmd_id_bytes = struct.pack('<H', cmd_id_x_offset)
                    data = x_bytes + y_bytes
                    data_frame = sof + cmd_id_bytes + bytes([data_length]) + data
                    self.ser.write(data_frame) 
                    time.sleep(0.01)

                    cmd_id_head_align = 0x0101
                    cmd_id_bytes_f = struct.pack('<H', cmd_id_head_align)
                    data_f = b'\x01'
                    data_frame_f = sof + cmd_id_bytes_f + bytes([len(data_f)]) + data_f
                    self.ser.write(data_frame_f) 
                except Exception as e:
                    print(f'发送数据错误：{e}')
   
                print(f'总时间： {time.time() - time_s}s')
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.Data_save_path}/{time_str}_start.png', frame_re)
                with open (f'{self.label_save_path}/{time_str}_start.txt', 'w', encoding='utf-8') as f:
                    for pt in pts:
                        f.write(f"{int(pt[0])},{int(pt[1])}\n")
                
            else:
                sof = b'\xA5'
                cmd_id_error = 0x0001
                cmd_id_error_bytes = struct.pack('<H', cmd_id_error)
                data_f = b'\x00'
                data_error = sof + cmd_id_error_bytes + data_f
                self.ser.write(data_error)
                head_x = 0
                head_y = 0
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_start.png', frame_re)                
 
            cv2.circle(frame, (int(circle_x), int(circle_y)), 2, color_list[0], 5)
            cv2.line(frame, (int(circle_x), int(circle_y)), (int(circle_x), int(circle_y)+2000), color_list[0], 2)
            frame = cv2.resize(frame, (800, 1300))
            cv2.imwrite(f'{self.result_save_path_head}/{time_str}_{head_x}_{head_y}_start.png', frame)


        elif data != False and str(data[0:2]) == '04':

            time_read_frame = time.time()
            ret, frame = self.camera.read()
            # ori_frame_ = frame.copy()

            end_x = self.coordinate[0] + self.tail_x_offset
            # start_x = end_x - 1800
            start_x = end_x - self.tail_w
            end_y = self.coordinate[1] + self.tail_y_offset
            start_y = end_y - self.tail_h

            frame = frame[start_y:end_y, start_x:end_x]
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_cp = frame.copy()

            roi_tail_rotate_base_p_x = self.origin_y - start_y
            roi_tail_rotate_base_p_y = self.tail_w - (self.origin_x - start_x)
            print(f'roi_tail_rotate_base_p_x:{roi_tail_rotate_base_p_x}, roi_tail_rotate_base_p_y:{roi_tail_rotate_base_p_y}')

            frame_re = cv2.resize(frame, (800, 1300))
            frame_as = np.ascontiguousarray(frame_re[:, :, ::-1].transpose(2, 0, 1))

            infer_time = time.time()
            detection = infer_frame(frame_as, self.sess, self.input_name, self.output_names)
            print(f'推理时间：{time.time() - infer_time}s')

            if self.segm_infer:
                print('开始推理分割模型')
                segm_t = time.time()
                frame_size = cv2.resize(frame, (640, 640))
                frame_se = np.ascontiguousarray(frame_size[:, :, ::-1].transpose(2, 0, 1))
                pred = infer_segm(frame_se, self.sess_, self.input_name_, self.output_names_)
                pred = np.where(pred.squeeze() > 0, 255, 0).astype(np.uint8)
                pred = cv2.resize(pred, (frame.shape[1],frame.shape[0]))
                print(f"分割推理时间：{time.time() - segm_t}s")
                pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)


            color_list = [[0, 0, 255], [0,255,0]]
            pts = detection[0][0]  
            pts[:, 0] = pts[:, 0] * (self.tail_h / 800) 
            pts[:, 1] = pts[:, 1] * (self.tail_w / 1300) 
            # pts[:, 1] = pts[:, 1] * (1800 / 1300) 
            pts_y = pts[pts[:, 1].argsort()]

            circle_x = 0
            circle_y = 0
            is_error = False
            # 判断检测是否有问题
            if pts.ndim == 2 and np.any(pts > 0) and 950 < np.linalg.norm(pts[0] - pts[1]) < 1900:
            # if pts.ndim == 2 and np.any(pts > 0)  and abs(pts[0][0] - pts[1][0]) < 100 and abs(pts[0][1] - pts[1][1]) > 620:
                # 检测两个角点正常
                print('检测正常')
                circle_x = pts_y[0][0]
                circle_y = pts_y[0][1]
                if self.segm_infer:
                    pts = pts.astype(np.int32)
                    circle_x = get_kde(pred, pts)
                    # circle_x = int(circle_x)
                # 可能存在异形
                # circle_x = pts_y[1][0] if circle_x - pts_y[1][0] > 100 else pts_y[0][0]

                tail_p_x = circle_x - roi_tail_rotate_base_p_x
                tail_p_y = circle_y - roi_tail_rotate_base_p_y
                restore_coordinate = [circle_x, circle_y]

            else:
                # 角点检测有问题
                is_error = True
                tail_p_x = 0
                tail_p_y = 0
                print('检测有误， 请检查图像')

            if not is_error:
                # right_x = round(tail_p_x * 0.0604, 2) * 1
                # right_y = round(tail_p_y * 0.0604, 2) * 1
                tail_x = round(tail_p_x * self.factor, 2) * 1
                tail_y = round(tail_p_y * self.factor, 2) * 1
                print('tail_x', tail_x, 'tail_y', tail_y)

                tail_x_bytes = struct.pack('<f', tail_x)
                tail_y_bytes = struct.pack('<f', tail_y)
                tail_data_length = len(tail_x_bytes) + len(tail_y_bytes)
                # print(f'x_bytes: {x_bytes}, y_bytes: {y_bytes}')

                if tail_data_length != 8:
                    print(f'字节数据长度不匹配， 实际长度：{data_length}')

                cmd_id_tail_offset = 0x010D
                sof = b'\xA5'
                cmd_id_bytes_ = struct.pack('<H', cmd_id_tail_offset)
                tail_data = tail_x_bytes + tail_y_bytes
                tail_data_frame = sof + cmd_id_bytes_ + bytes([tail_data_length]) + tail_data
                self.ser.write(tail_data_frame) 

                time.sleep(0.01)
                hex_data_f = f'A5 05 01 04 00 00 00 00'
                byte_data_f = bytes.fromhex(hex_data_f.replace(' ', ''))
                self.ser.write(byte_data_f)
                print(f'尾部花费总时间 {time.time() - time_read_frame}s')
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.Data_save_path}/{time_str}_end.png', frame_re)
                with open (f'{self.label_save_path}/{time_str}_end.txt', 'w', encoding='utf-8') as f:
                    for pt in pts:
                        f.write(f"{int(pt[0])},{int(pt[1])}\n")
            else:
                tail_x = -1
                tail_y = -1
                hex_data1 = f'A5 01 00 00'
                byte_data = bytes.fromhex(hex_data1.replace(' ', ''))
                self.ser.write(byte_data)
                time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
                cv2.imwrite(f'{self.detect_error_save_path}/{time_str}_end.png', frame_re)
       
            cv2.circle(frame, (int(circle_x), int(circle_y)), 2, color_list[0], 5)
            cv2.line(frame, (int(circle_x), int(circle_y)), (int(circle_x), int(circle_y)+2000), color_list[0], 2)
            frame = cv2.resize(frame, (800, 1300))
            cv2.imwrite(f'{self.result_save_path_tail}/{time_str}_{tail_x}_{tail_y}_end.png', frame)

        elif data != False and str(data[0:2]) == '0f':
            time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            ret, frame = self.camera.read()
            cv2.imwrite(f'{self.result_save_path_compose_frame}/{time_str}.png', frame)

        if not restore_coordinate is None:
            # for i, pt in enumerate(pts):
            #     cv2.circle(frame, tuple(pt.astype(np.int32)), 20, color_list[i], 10)
            
            cv2.circle(frame_cp, (int(restore_coordinate[0]), int(restore_coordinate[1])), 5, color_list[0], 5)
            cv2.line(frame_cp, (int(restore_coordinate[0]), int(restore_coordinate[1])), (int(restore_coordinate[0]), int(restore_coordinate[1]+2000)), (0, 0, 255), 2)

            if self.segm_infer:
                frame_cp = cv2.addWeighted(frame_cp, 0.8, pred_bgr, 0.5, 0)
            img_rgb = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(frame_cp, (960, 720))
            image = QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
            scaled_image = image.scaled(self.infer_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.infer_label.setPixmap(QPixmap.fromImage(scaled_image))

    def run_inference(self):
        if self.camera is not None:
            ret, frame = self.camera.read()
            # 12mm镜头 head
            if self.front_radio.isChecked():
                # rotate
                end_x = self.coordinate[0] + self.head_x_offset
                start_x = end_x - self.head_w
                start_y = self.coordinate[1] + self.head_y_offset
                end_y = start_y + self.head_h

                frame = frame[start_y:end_y, start_x:end_x]
                print(frame.shape)
                pad = (800-self.head_h) // 2
                frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                roi_rotate_base_p_x = self.origin_y - start_y + pad
                roi_rotate_base_p_y = self.head_w - (self.origin_x - start_x)
                print(f'roi_rotate_base_p_x:{roi_rotate_base_p_x}, roi_rotate_base_p_y:{roi_rotate_base_p_y}')
            else:
                # 12mm镜头  tail
                # rotate
                end_x = self.coordinate[0] + self.tail_x_offset
                start_x = end_x - self.tail_w
                end_y = self.coordinate[1] + self.tail_y_offset
                start_y = end_y - self.tail_h
                frame = frame[start_y:end_y, start_x:end_x]
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                roi_tail_rotate_base_p_x = self.origin_y - start_y
                roi_tail_rotate_base_p_y = self.tail_w - (self.origin_x - start_x)
                print(f'roi_tail_rotate_base_p_x:{roi_tail_rotate_base_p_x}, roi_tail_rotate_base_p_y:{roi_tail_rotate_base_p_y}')

            frame_re = cv2.resize(frame, (800, 1300))
            frame_as = np.ascontiguousarray(frame_re[:, :, ::-1].transpose(2, 0, 1))

            time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            infer_t = time.time()
            detection = infer_frame(frame_as, self.sess, self.input_name, self.output_names)
            print(f'kp推理时间：{time.time() - infer_t}s')

            if self.segm_infer:
                print('开始推理分割模型')
                frame_size = cv2.resize(frame, (640, 640))
                frame_se = np.ascontiguousarray(frame_size[:, :, ::-1].transpose(2, 0, 1))
                pred = infer_segm(frame_se, self.sess_, self.input_name_, self.output_names_)
                pred = np.where(pred.squeeze() > 0, 255, 0).astype(np.uint8)
                pred = cv2.resize(pred, (frame.shape[1],frame.shape[0]))
                pred_bgr = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
               
                # contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # max_cn = max(contours, key=cv2.contourArea)

            color_list = [[255, 0, 0], [0,255,0]]
            pts = detection[0][0]
            if self.front_radio.isChecked():
                pts[:, 1] = pts[:, 1] * (self.head_w / 1300) # head
            else:
                # tail
                pts[:, 0] = pts[:, 0] * (self.tail_h / 800) 
                pts[:, 1] = pts[:, 1] * (self.tail_w / 1300) 

            pts = pts[pts[:, 1].argsort()]
            if pts[0][1] < 0:
                pts[0][1] = pts[1][1]
            pts = pts.astype(np.int32)
            if self.segm_infer:
                x = get_kde(pred, pts)
                x = int(x)
                cv2.line(frame, (x, pts[0][1]), (x, pts[1][1]), (0, 0, 255), 2)
          
            for i, pt in enumerate(pts):
                cv2.circle(frame, tuple(pt.astype(np.int32)), 5, color_list[i], 5)
     
            if self.segm_infer:
                frame = cv2.addWeighted(frame, 0.8, pred_bgr, 0.5, 0)
            roi_img_cp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi_img_cp = cv2.resize(roi_img_cp, (960, 720))
            # print(roi_img_cp.shape[1], roi_img_cp.shape[0])
            image = QImage(roi_img_cp, roi_img_cp.shape[1], roi_img_cp.shape[0], QImage.Format_RGB888)
            # 调整图像大小以适应标签尺寸
            scaled_image = image.scaled(self.infer_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.infer_label.setPixmap(QPixmap.fromImage(scaled_image))

    def toggle_inference(self):
        if self.infer_running:
            self.infer_timer.stop()
            self.infer_running = False
            self.inference_button.setText("执行推理")
            
        else:
            if self.reload_buttons[0].isChecked():
                self.reload_config()
                self.reload_buttons[0].setChecked(False)

            self.infer_timer.start(30)  
            self.infer_running = True
            self.inference_button.setText("停止推理")

    def reload_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            
            self.base_point = self.config['base_point'] 
            self.origin_x = self.base_point[0]
            self.origin_y = self.base_point[1]   
            self.coordinate = [self.origin_x, self.origin_y]
        
            self.factor = self.config['factor']

            self.head_x_offset = self.config['head_roi']['x_offset']
            self.head_w = self.config['head_roi']['w']
            self.head_h = self.config['head_roi']['h']
            self.head_y_offset = self.config['head_roi']['y_offset']
            # tail
            self.tail_x_offset = self.config['tail_roi']['x_offset']
            self.tail_w = self.config['tail_roi']['w']
            self.tail_h = self.config['tail_roi']['h']
            self.tail_y_offset = self.config['tail_roi']['y_offset']
        except Exception as e:
            QMessageBox.critical(self, "错误", f'加载配置文件错误： {e}')
            self.close()
            return

    def disable_all_buttons(self):
        """禁用所有按钮"""
        self.checkbox_infer.setEnabled(False)
        self.save_button.setEnabled(False)
        self.p_detect_button.setEnabled(False)
        self.factor_detect_button.setEnabled(False)
        self.inference_button.setEnabled(False)
        self.ser_inference_button.setEnabled(False)
        self.checkbox.setEnabled(False)
        self.checkbox_segm.setEnabled(False)
        self.reload_buttons[0].setEnabled(False)
        self.reload_buttons[1].setEnabled(False)
        self.reload_buttons[2].setEnabled(False)

    def enable_all_buttons(self):
        """禁用所有按钮"""
        self.checkbox_infer.setEnabled(True)
        self.save_button.setEnabled(True)
        self.p_detect_button.setEnabled(True)
        self.factor_detect_button.setEnabled(True)
        self.inference_button.setEnabled(True)
        self.ser_inference_button.setEnabled(True)
        self.checkbox.setEnabled(True)
        self.checkbox_segm.setEnabled(True)
        self.reload_buttons[0].setEnabled(True)
        self.reload_buttons[1].setEnabled(True)
        self.reload_buttons[2].setEnabled(True)

    def update_frame(self):
        ret, frame = self.camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (960, 720))
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  
        # 调整图像大小以适应标签尺寸
        scaled_image1 = image.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(scaled_image1))

    def toggle_timer(self, state):
        if state == 2:
            self.timer.start(33)
        else:
            self.timer.stop()
    def switch_mode(self, state):
        if state == 2:
            self.infer_label.setVisible(True)
            if not self.infer_label in self.hbox_labels.children():
                self.hbox_labels.addWidget(self.infer_label)
        else:
            self.infer_label.setVisible(False)
            if self.infer_label in self.hbox_labels.children():
                self.hbox_labels.removeWidget(self.infer_label)

        self.hbox_labels.update()

    def is_segm(self, state):
        if state == 2:
            self.segm_infer = True
        else:
            self.segm_infer = False

  
    def save_image(self):

        save_path = 'Data/thor_keypoint_data'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        str_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.png'
        img_name = os.path.join(save_path, str_time)
        ret, image = self.camera.read()

        if not ret:
            print('camera init error')

        # 前部
        if self.config['save_im_way']['head']:
            end_x = self.coordinate[0] + self.head_x_offset
            start_x = end_x - self.head_w
            start_y = self.coordinate[1] + self.head_y_offset
            end_y = start_y + self.head_h
            image = image[start_y:end_y, start_x:end_x]
            image = cv2.copyMakeBorder(image, 250, 250, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.resize(image, (800, 1300))

        elif self.config['save_im_way']['full_im']:
            image = image
        else:
            # 尾部
            end_x = self.coordinate[0] + self.tail_x_offset
            # start_x = end_x - 1800
            start_x = end_x - self.tail_w
            end_y = self.coordinate[1] + self.tail_y_offset
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
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        if self.serial_listener is not None:
            self.serial_listener.stop()
            self.ser.close()
        event.accept()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
