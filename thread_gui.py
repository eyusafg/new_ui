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
                            QHBoxLayout, QWidget, QCheckBox, QRadioButton, 
                            QMessageBox,QApplication, QSizePolicy)
import onnxruntime as rt
import yaml
import cv2
from utils.detect_circle import detect_circle
from utils.irregular_im_de import get_kde
from utils.calculate_distance import calculate_distance
import struct
import shutil
from PyCameraList.camera_device import list_video_devices

    
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

            # 并行加载模型
            sess = self.load_model(config['model_path'])
            segm_sess = self.load_model(config['segm_model_path'], delay=False)

            # 返回结果
            result = {
                'config': config,
                'sess': sess,
                'input_name': sess.get_inputs()[0].name,  # 可能会错
                'output_names': [out.name for out in sess.get_outputs()],
                'segm_sess': segm_sess,
                'segm_input_name': segm_sess.get_inputs()[0].name,  
                'segm_output_names': [out.name for out in sess.get_outputs()]               
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
            print('camera index:', index)
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.config_path = 'profiles//config.yaml'
        self.camera = None
        self.ser = None
        self.segm_infer = False

        # 初始化线程池
        self.thread_pool = QThreadPool.globalInstance()

        # 并行启动初始化任务
        self.start_paraller_init()

        # 初始化ui
        self.init_ui()
        
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
        
        self.infer_label = QLabel(self)
        self.infer_label.setAlignment(Qt.AlignCenter)
        self.infer_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
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
        self.reload_buttons[1].clicked.connect(self.reload_camera)
        self.reload_buttons[2].clicked.connect(self.reload_serial)

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

        # 硬件初始化
        if not self.reload_buttons[0].isChecked():
            self.start_hardware_init()
    
    def start_hardware_init(self):
        
        self.reload_camera()
        self.reload_serial()

        # self.camera_worker = CameraWorker(self.config)
        # self.camera_worker.signals.finished.connect(self.on_camera_ready)
        # self.camera_worker.signals.error.connect(lambda e: self.show_error(e))

        # self.serial_worker = SerialWorker(self.config)
        # self.serial_worker.signals.finished.connect(self.on_serial_ready)
        # self.serial_worker.signals.error.connect(lambda e: self.show_error(e))

        # self.thread_pool.start(self.camera_worker)
        # self.thread_pool.start(self.serial_worker)

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
        print(self.ser)
        self.serial_listener = SerialListener(self.ser)
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

            self.infer_timer.start(30)  
            self.infer_running = True
            self.inference_button.setText("停止推理")

    def update_preview(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
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
        self.ser_inference_button.setEnabled(False)
        self.checkbox.setEnabled(False)
        self.checkbox_segm.setEnabled(False)
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
        self.ser_inference_button.setEnabled(True)
        self.checkbox.setEnabled(True)
        self.checkbox_segm.setEnabled(True)
        self.reload_buttons[0].setEnabled(True)
        self.reload_buttons[1].setEnabled(True)
        self.reload_buttons[2].setEnabled(True)
    ##################################################

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
        self.result_save_path_compose_frame = self.result_save_path_compose_frame
        if not os.path.exists(self.result_save_path_compose_frame):
            os.makedirs(self.result_save_path_compose_frame, exist_ok=True)

        self.detect_error_save_path = 'Data/thor_kepoint/error_detect'
        self.detect_error_save_path = self.detect_error_save_path
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
                pred = infer_segm(frame_se, self.segm_sess, self.segm_input_name, self.segm_output_names)
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
                pred = infer_segm(frame_se, self.segm_sess, self.segm_input_name, self.segm_output_names)
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

    def save_image(self):
        save_path = 'Data/thor_keypoint_data'
        save_path = save_path
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
        if self.preview_timer:
            self.preview_timer.stop()
        if self.infer_timer:
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
