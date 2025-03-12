import sys
import time
import os
from datetime import datetime
import threading
from PyQt6.QtWidgets import QApplication, QMainWindow, QListWidgetItem,QMainWindow,QTreeWidgetItem
from PyQt6.QtCore import QTimer,pyqtSignal,QThread,pyqtSignal
import serial
import serial.tools.list_ports 
from iap import Ui_IAP
import Bin_Unpacking
import Bootloader_func
from Bin_Unpacking import packetstxt
from Bin_Unpacking import indextxt
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox,QInputDialog
from test_sql import DeviceManagementDB
import string
Debug = 0
rec_data = 0x00



class ListConfig:
    def __init__(self, unit, Set_burning_prefix, Succeed_prefix,\
                 Ser_prefix,\
                 Binfile, UIDConfine, Version_prefix, issucceed, Version_prefixed):
        self.unit = unit
        self.Set_burning_prefix = Set_burning_prefix
        self.Succeed_prefix = Succeed_prefix
        self.Ser_prefix = Ser_prefix
        self.Binfile = Binfile
        self.UIDConfine = UIDConfine
        self.Version_prefix = Version_prefix
        self.idSucceed = issucceed
        self.Version_prefixed = Version_prefixed

# 全局变量
Listdata = ListConfig("byte", 
                    "烧录设定次数为:", 
                    "烧录成功次数为:",\
                    '串口状态：串口关闭',\
                    "Bin文件未导入",\
                    "UID文件未导入，请导入文件", \
                    "升级前软件版本为: ",\
                    "烧录状态： ",
                    "升级后软件版本为: ")

class SerialConfig:
    def __init__(self, baud_rate, serial_port, bytesize, parity, stopbits,Ser_Enable,Ser_Disbale,SerConfig,Senddata,Recdata,\
                 UIDConfine,SendCmd, version_number, comparative_verison, version_number_):
        self.baud_rate = baud_rate
        self.serial_port = serial_port
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.enable = Ser_Enable
        self.disable = Ser_Disbale
        self.SerConfig = SerConfig
        self.Senddata = Senddata
        self.Recdata = Recdata
        self.UIDConfine = UIDConfine
        self.SendCmd = SendCmd
        self.version_number = version_number
        self.comparative_verison = comparative_verison
        self.version_number_ = version_number_
# 全局变量
Serconfig = SerialConfig(0, 0, 0, 0, 0, 0, 0 , 0, [], [], 0, 0, 0, '2.4.0', 0)

class user_init:
    def __init__(self, SendID,RecID, Burning_time,Enable_Updata,Count_finish,progress_count,Sendcomplete,dataCount,UsedTime, hardware_version_addr, unique_ID_addr, uid):
        self.SendID = SendID
        self.RecID = RecID
        self.Burning_time = Burning_time
        self.Enable_Updata = Enable_Updata
        self.Count_finish = Count_finish
        self.progress_count = progress_count
        self.Sendcomplete = Sendcomplete
        self.dataCount = dataCount
        self.UsedTime = UsedTime
        self.hardware_version_addr = hardware_version_addr
        self.unique_ID_addr = unique_ID_addr
        self.UID = uid
# 全局变量
Userdata = user_init(0, 0, 0, 0 ,0 , 0 , 0, 0, '0', 0, 0, 0)


# IAP本身的状态机
class Worker(QThread):
    Updata_text_signal = pyqtSignal()
    Show_List_signal = pyqtSignal()
    Show_data2tree_signal = pyqtSignal()
    Detect_ser_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.detected_serial = False  # 添加一个标志来控制 Detect_ser_signal 的处理
    def run(self):
        self.timer = QTimer(self)
        self.timer.start(500)  # 每 500 毫秒发射一次信号
        self.timer.timeout.connect(self.emit_updates)
        self.exec()
    def emit_updates(self):
        self.Updata_text_signal.emit()
        self.Show_List_signal.emit()
        self.Show_data2tree_signal.emit()
        self.Detect_ser_signal.emit()

    
class IAP(QMainWindow,Ui_IAP):
    def __init__(self,parent = None)-> None:
        super(IAP,self).__init__(parent=parent)
        self.setupUi(self)
        self.Send_ID.setText(str(1))
        self.Rec_ID.setText(str(1))
        self.burning_time.setText(str(1))
        # 状态机映射
        self.worker = Worker()
        self.worker.start()
        self.worker.Updata_text_signal.connect(self.Updata_text)
        self.worker.Show_List_signal.connect(self.Read_information_list)
        # self.worker.Show_data2tree_signal.connect(self.Show_data2tree)
        self.worker.Detect_ser_signal.connect(self.Func_Detect_Serial_Auto)
        # 串口变量初始化
        self.added_ports = set()  # 用来存储已添加的端口，以去除重复项
        self.BaudrateBox.setCurrentIndex(7)
        self.DataBox.setCurrentIndex(3)
        global Serconfig
        global Userdata
        global Listdata
  
        # 创建 FuncIAP 实例并连接信号
        self.Func = FuncIAP()
        self.Func.error_signal.connect(self.show_error_message)
        self.Func.request_company_name_signal.connect(self.get_company_name)  # 连接信号
        self.Func.company_name_received_signal.connect(self.Func.save_company_uid)  # 连接信号
        self.Func.reply_signal.connect(self.show_confirmation_dialog)
        self.Func.update_result_signal.connect(self.handle_update_result)
        # 启动线程
        t1 = threading.Thread(target=self.Func.Function)
        t1.start()

    def closeEvent(self, event):
        os._exit(0)
        
    def get_company_name(self):
        # 弹出输入框，获取用户输入的公司名
        company_name, ok = QInputDialog.getText(self, "输入公司名", "请输入公司名：")
        if ok and company_name:  # 如果用户点击了确定并且输入了内容
            print('company_name', company_name)
            self.Func.company_name_received_signal.emit(company_name)  # 将公司名传回

    # 检索串口
    def Func_Detect_Serial_Auto(self):
        if not self.worker.detected_serial:  # 检查标志
            ports = serial.tools.list_ports.comports()
            current_ports = set()
            for port in ports:
                current_ports.add(port.device)
            for port in current_ports:
                if port not in self.added_ports:
                    # 如果端口尚未添加，则添加到PortBox和added_ports集合中
                    self.PortBox.addItem(port)
                    self.added_ports.add(port)
            self.worker.detected_serial = True  # 设置标志为已检测

    def Func_Detect_Serial(self):
        ports = serial.tools.list_ports.comports()
        current_ports = set()
        for port in ports:
            current_ports.add(port.device)
        for port in current_ports:
            if port not in self.added_ports:
                # 如果端口尚未添加，则添加到PortBox和added_ports集合中
                self.PortBox.addItem(port)
                self.added_ports.add(port)

    # 开启串口
    def Func_Open_Serial(self):# 传信号
        self.Func_Detect_Serial()
        Serconfig.baud_rate = self.BaudrateBox.currentText()
        Serconfig.serial_port = self.PortBox.currentText()
        Serconfig.bytesize = self.DataBox.currentText()
        Serconfig.stopbits = self.StopBox.currentText()
        if self.VerifyBox.currentText() == 'None':
            Serconfig.parity = serial.PARITY_NONE
        elif self.VerifyBox.currentText() == 'Odd':
            Serconfig.parity = serial.PARITY_ODD
        elif self.VerifyBox.currentText() == 'Even':
            Serconfig.parity = serial.PARITY_EVEN
        elif self.VerifyBox.currentText() == 'Mark':
            Serconfig.parity = serial.PARITY_MARK
        elif self.VerifyBox.currentText() == 'Space':
            Serconfig.parity = serial.PARITY_SPACE
        Serconfig.enable = 1

    # 关闭串口
    def Func_Close_Serial(self):
        Serconfig.disable = 1

    # 输入文件
    def FuncDataInput(self): # 导入烧录文件和检查UID合并
        Serconfig.SendCmd = 1
    def FuncEncryptDataInput(self):  # 导入需要加密的文件
        Serconfig.SendCmd = 10
    def FuncSaveParameter(self):  # 保存各种参数到本地
        Serconfig.SendCmd = 9
    # def Del_Sql_Data(self):  # 清除数据库数据 
    #     Serconfig.SendCmd = 8
    def Del_sql_data(self):
        Serconfig.SendCmd = 7

    # 系统 GUI 界面数据更新与获取
    def Updata_text(self):
        Userdata.RecID = self.Rec_ID.text()
        Userdata.SendID = self.Send_ID.text()
        Userdata.Burning_time = self.burning_time.text()
        Bootloader_func.Sendid = Userdata.RecID
        Bootloader_func.Recid = Userdata.RecID

        Bootloader_func.size_package()
        self.progressBar.setProperty("value", Userdata.progress_count)

    def Read_information_list(self):
        Userdata.UID = self.Rec_ID_3.text()  # 改为uid
        self.listWidget.clear()  # 清空当前列表
        if Serconfig.SerConfig == 1:
            Listdata.Ser_prefix = '串口状态：串口已打开'
        elif Serconfig.SerConfig == 0:
            Listdata.Ser_prefix = '串口状态：串口关闭'
        if Bin_Unpacking.Bin_variable.inputfinish == 1:
            Listdata.Binfile = "Bin文件导入成功"
        elif Bin_Unpacking.Bin_variable.inputfinish == 0:
            Listdata.Binfile = "Bin文件未导入"
        if Serconfig.UIDConfine == 1:
            Listdata.UIDConfine = "UID正确，允许烧录"
        elif Serconfig.UIDConfine == 2:
            Listdata.UIDConfine = "UID错误，不允许烧录"
        elif Serconfig.UIDConfine == 0:
            Listdata.UIDConfine = "UID文件未导入，请导入文件"
        new_data = [Listdata.UIDConfine,\
                    Listdata.Ser_prefix,\
                    Listdata.Set_burning_prefix + str(Userdata.Burning_time),\
                    Listdata.Succeed_prefix + str(Userdata.Count_finish),\
                    Listdata.Binfile,\
                    "烧录时间为：" + str(Userdata.UsedTime),\
                    "烧录文件名：" + str(Bin_Unpacking.Bin_variable.file_name), \
                    Listdata.Version_prefix + str(Serconfig.version_number),\
                    Listdata.idSucceed,\
                    Listdata.Version_prefixed + str(Serconfig.version_number_)]
        
        for item_text in new_data:
            item = QListWidgetItem(item_text)
            self.listWidget.addItem(item)

    def Show_data2tree(self):
        count = 0
        LEN = 0
        PID = 0
        if Userdata.Sendcomplete == 1:
            print('Userdata.dataCount', Userdata.dataCount)
            for i in range(Userdata.dataCount):
                current_time = datetime.now().strftime('%H:%M:%S')
                count += 1
                LEN = int(len(Serconfig.Senddata[i]) / 2)
                send_row = QTreeWidgetItem(self.treeWidget,\
                                       [str(count),current_time, str(PID), str(Serconfig.serial_port),\
                                        "暂无", str(LEN), str(Serconfig.Senddata[i]),\
                                        '发送帧'])
                count += 1
                LEN = int(len(Serconfig.Recdata[i]) / 2)
                receive_row = QTreeWidgetItem(self.treeWidget,\
                                       [str(count),current_time, str(PID), str(Serconfig.serial_port), \
                                        "暂无", str(LEN),str(Serconfig.Recdata[i]),\
                                        '接收帧'])
                self.treeWidget.expandAll()
            Userdata.dataCount = 0

    # 新增槽函数显示错误
    def show_error_message(self, message):
        QMessageBox.critical(self, "错误", message)

    def show_confirmation_dialog(self):
        """显示是否更新的弹窗"""
        reply = QMessageBox.question(
            self, 
            "冲突", 
            "该机器码已存在，是否覆盖？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # 通知 FuncIAP 执行更新
            self.Func.perform_update()
        else:
            QMessageBox.information(self, "提示", "已取消更新操作")

    def handle_update_result(self, success, message):
        """处理更新结果"""
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.critical(self, "错误", message)
            

def parse_version(version_str):
    """
    将版本号字符串解析为一个可比较的元组
    :param version_str: 版本号字符串，例如 "2.1.12"
    :return: 版本号元组，例如 (2, 1, 12)
    """
    return tuple(map(int, version_str.split('.')))

Burning_Flags = 0
class FuncIAP(QObject):
    error_signal = pyqtSignal(str)  # 错误信号
    reply_signal = pyqtSignal()  # 错误信号
    show_id_signal = pyqtSignal(str)  # 显示ID信号
    request_company_name_signal = pyqtSignal()  # 请求用户输入公司名
    company_name_received_signal = pyqtSignal(str)  # 将用户输入的公司名传回
    update_result_signal = pyqtSignal(bool, str)  # 新增：传递更新结果 (成功状态, 消息)
    def __init__(self):
        super().__init__()
        self.Func_Ser = serial.Serial()
        global Serconfig
        global Userdata
        global packetstxt
        global indextxt

    def del_db_data(self):
        self.db.delete_table(self.table_name)
        self.db.delete_data(self.db_file_name)

    def SerialEnable(self):
        try:
            if Serconfig.enable == 1:
                if self.Func_Ser.is_open:
                    print("串口已经使能")
                else:
                    self.Func_Ser = serial.Serial(
                        port = Serconfig.serial_port,
                        baudrate = Serconfig.baud_rate,
                        parity = Serconfig.parity,  # 设置奇校验
                        stopbits = int(Serconfig.stopbits),  # 通常使用1个停止位
                        bytesize = int(Serconfig.bytesize),  # 8个数据位
                        timeout = 1  # 设置读取超时为1秒
                    )
                    print("串口使能成功")
                Serconfig.SerConfig = 1
                Serconfig.enable = 0
            if Serconfig.disable == 1:
                self.Func_Ser.close()
                print("串口已经关闭")
                Serconfig.disable = 0
        except Exception as e:
            Serconfig.enable = 0
            Serconfig.disable = 0
            Serconfig.SerConfig = 0
            self.error_signal.emit("串口已在其他地方打开， 请先关闭")
            print("串口使能失败", e)

    def send_data(self, data):
        bytes_hex = bytes.fromhex(data)
        self.Func_Ser.write(bytes_hex)
        time.sleep(0.05)
        raw_data = self.Func_Ser.read_all()
        return raw_data.hex()
    
    def get_version(self):
        global Userdata

        ################## 先发送地址 请求版本号 #############
        version_s = '0104753400022A09'
        ver_data = self.send_data(version_s)
        if ver_data[6:14] == '00000000':
            version_s = '0104753000026BC8'
            ver_data = self.send_data(version_s)
        elif ver_data == '':
            print('初始没有app在内')
            return False     # 如果初始没有app在内则可能返回空字符
        #############################################
        ver_data = ver_data[6:14]
        version_number = f'{int(ver_data[6:8], 16)}.{int(ver_data[0:2],16)}.{int(ver_data[2:4], 16)}'
        Vn = parse_version(version_number)  # 软件版本号
        ############## 得到版本号 ####################

        if Vn >= parse_version(Serconfig.comparative_verison):      # 大于某个版本的用40000
            reboot_addr = f'0106{hex(int(40000))[2:]}00AA2631'      # 重启地址
            unique_ID_addr = f'0103{hex(int(40003))[2:]}00021B8F'   # 获取唯一ID的地址
            hardware_version_addr = f'010475360001CBC8'             # 获取硬件版本信息
        else:
            reboot_addr = f'0106{hex(int(40004))[2:]}00AA67F0'
            unique_ID_addr = f'0103{hex(int(40000))[2:]}0002EB8F'
            hardware_version_addr = f'0104753200018A09'               

        Bootloader_func.Iap_Cmd.Reboot = reboot_addr   # 重启指令只是针对app， 如果板子内无app则无响应
        Userdata.unique_ID_addr = unique_ID_addr
        Userdata.hardware_version_addr = hardware_version_addr

        return version_number
    
    def save_company_uid(self,company_name):  
        # 数据库初始化
        self.db_file_name = 'syt_round_neck_data'
        self.table_name = 'Round_Neck'
        self.db = DeviceManagementDB(host='192.168.6.241', user='syt', password='03106666', filename=self.db_file_name, table_name=self.table_name)
        self.db.create_database()
        self.db.database = self.db_file_name
        self.db.create_table()

        version_number = self.get_version()

        time.sleep(0.1)
        hardware_version = self.send_data(Userdata.hardware_version_addr)[6:10]
        time.sleep(0.2)
        _ = self.send_data(Bootloader_func.Iap_Cmd.Reboot) # 重启
        for i in range(10):
            data = "79"
            raw_data = 0
            bytes_hex = bytes.fromhex(data)
            self.Func_Ser.write(bytes_hex)
            time.sleep(0.05)
            raw_data = self.Func_Ser.read_all()  
            raw_data_ = raw_data.hex()

            if raw_data_:
                uid = raw_data_
                break

        self.pending_company = str(company_name)
        self.pending_uid = str(uid)
        self.pending_version = str(version_number)
        self.pending_hardware = str(hardware_version)

        ret = self.db.insert_device(str(company_name), str(uid), str(version_number), str(hardware_version))
        print('ret', ret)
        if ret == 1:
            self.error_signal.emit("保存成功！")
        elif ret == -1:
            self.reply_signal.emit()
        else:
            self.error_signal.emit(ret)
        self.db.query_devices()
    
    def perform_update(self):
        """执行实际的数据库更新操作"""
        try:
            success = self.db.update_device(
                self.pending_company,
                self.pending_uid,
                self.pending_version,
                self.pending_hardware
            )
            if success:
                self.update_result_signal.emit(True, "记录更新成功！")
            else:
                self.update_result_signal.emit(False, f"数据库错误，{success}")
        except Exception as e:
            self.update_result_signal.emit(False, f"更新失败：{str(e)}")
    def Show_UniqueID(self):  # 获取机器唯一识别码
        global Userdata
        self.get_version()
        unique_ID = self.send_data(Userdata.unique_ID_addr)[6:14]
        self.show_id_signal.emit(unique_ID)

    def get_uid(self):
        version_number = self.get_version()
        if not version_number:
            return False
        Serconfig.version_number = version_number
        data = Bootloader_func.Iap_Cmd.Reboot
        self.send_data(data)
        return True

    def Function(self):
        while True:
            if Serconfig.enable == 1:
                self.SerialEnable()
                Serconfig.enable = 0
            if Serconfig.disable == 1:
                self.SerialEnable()
                Serconfig.SerConfig = 0
                Serconfig.disable = 0   

            cmd = Serconfig.SendCmd
            if cmd == 7:
                self.del_db_data()
                Serconfig.SendCmd = 0
            elif cmd == 9:
                self.request_company_name_signal.emit()
                Serconfig.SendCmd = 0
            elif cmd == 10:  # 选择加密文件 
                self._handle_encrypt_file()
                Serconfig.SendCmd = 0
            elif cmd == 1:
                self._handle_firmware_flashing()
                Serconfig.SendCmd = 0
            
            time.sleep(0.01)

    def _validate_uid(self, uid):
        '''校验uid是否为四字节16进制'''
        return len(uid) == 8 and all(c in string.hexdigits for c in uid)
 
    def _handle_encrypt_file(self):
        '''处理加密文件操作'''
        print('加密文件')
        if not Userdata.UID:
            self.error_signal.emit("请先填写UID！")
            return 
        # 格式校验
        uid = Userdata.UID.strip()
        if not self._validate_uid(uid):
            self.error_signal.emit("UID格式错误！")
            return
        Bin_Unpacking.Bin_variable.uid = uid
        flag = Bin_Unpacking.select_and_encrypt_file()  # 将这里面选择部分去掉
        if not flag:
            self.error_signal.emit("加密失败！")
        else:
            self.error_signal.emit("加密成功！")

    def _handle_firmware_flashing(self):
        '''固件烧录'''
        if not self.Func_Ser.is_open:
            self.error_signal.emit("请先打开串口！")
            return
        if not self.get_uid():
            Serconfig.version_number = 0
        for _ in range(10):
            data = "79"    # 发79指令，如果有响应则在运行bootloader， 如果没有响应则证明没有任何程序在运行
            bytes_hex = bytes.fromhex(data)          
            self.Func_Ser.write(bytes_hex)
            time.sleep(0.05)
            rec_data = self.Func_Ser.read_all().hex()
            if rec_data:
                print(f'收到UID： {rec_data}')
                return self._process_flashing(rec_data)
            
        self.error_signal.emit("无法获取UID， 请检查下位机版本")

    def _process_flashing(self, uid):
        '''根据是否收到UID 处理烧录'''
        Bin_Unpacking.Bin_variable.uid = uid
        ret = Bin_Unpacking.encrypt_file()
        if ret == -1:
            self.error_signal.emit("加密失败！无法烧录！")
        elif ret == -2:
            self.error_signal.emit("UID不匹配")
        elif ret:
            if Serconfig.version_number:
                self.HaveAPPUpdata()
            else:
                self.WithoutAPPUpdata()
            Serconfig.UIDConfine = 1  # 仅是改变提示状态
        else:
            self.error_signal.emit("板子内没有任何程序在运行")

    def HaveAPPUpdata(self):
        global Burning_Flags
        global Serconfig
        global rec_data
        global Userdata
        # 每次开始烧录前清空缓冲区
        self.Func_Ser.reset_input_buffer()
        self.Func_Ser.reset_output_buffer()

        Serconfig.version_number_ = 0
        Listdata.idSucceed = '烧录状态： 烧录中。。。。。'
        Userdata.Sendcomplete = 0
        Start_time = datetime.now()
        for i in range(int(Userdata.Burning_time)):
            Userdata.progress_count = 0
            Burning_Flags = 0
            while Burning_Flags == 0:
                print('0000000000000000000000000000000')
                data = Bootloader_func.Iap_Cmd.Reboot
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount+=1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.01)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 1:
                    break
            Userdata.progress_count += 0
            while Burning_Flags == 1:
                print(1111111111111111111111111111111111111)
                Bootloader_func.Request_version_number()
                data = Bootloader_func.Iap_Cmd.Request_version_numbers
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.006)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 2:
                    break
            Userdata.progress_count += 0
            while Burning_Flags == 2:
                print(222222222222222222222222222222222)
                Bootloader_func.Request_software_reset()
                data = Bootloader_func.Iap_Cmd.Request_software
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.006)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 3:
                    print(3333333333333333333333333333333333333333333333333333333)
                    break
            Userdata.progress_count += 1
            while Burning_Flags == 3:
                Bootloader_func.Request_package_size()
                data = Bootloader_func.Iap_Cmd.Request_package_size
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.006)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 4:
                    print(4444444444444444444444444444444444444444444444444444444444)
                    break
            Userdata.progress_count += 1
            Bootloader_func.size_package()
            data = Bootloader_func.Iap_Cmd.size_package
            bytes_hex = bytes.fromhex(data)
            time.sleep(0.001)
            Userdata.dataCount += 1
            self.Func_Ser.write(bytes_hex)
            time.sleep(0.006)
            self.read_serial_data()
            Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
            Serconfig.Recdata.append(rec_data)
            Userdata.progress_count += 1
            Bootloader_func.Deliver_data()
            data = 0
            for i in range(Bootloader_func.Iap_Cmd.index+1):
                data = Bootloader_func.Iap_Cmd.APP_data[i]
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.2)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                Userdata.progress_count+=1.5
            Bootloader_func.COPY_file()
            data = Bootloader_func.Iap_Cmd.COPY_file
            bytes_hex = bytes.fromhex(data)
            time.sleep(0.01)
            Userdata.dataCount += 1
            self.Func_Ser.write(bytes_hex)
            time.sleep(0.1)
            self.read_serial_data()
            Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
            Serconfig.Recdata.append(rec_data)
            time.sleep(5)
            Userdata.progress_count += 1.5
            if  Userdata.progress_count != 100:
                Userdata.progress_count = 100
            while Burning_Flags == 4:
                data = Bootloader_func.Iap_Cmd.Reboot
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.01)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 5:
                    print(555555555555555555555555555555555555)
                    break
            End_time = datetime.now()
            Userdata.UsedTime = End_time - Start_time
            Userdata.Count_finish+=1
            Userdata.Sendcomplete = 1
            Serconfig.UIDConfine = 3
        Listdata.idSucceed = '烧录状态： 烧录完成'
        print('烧录完成')
        time.sleep(5)
        version_number = self.get_version()
        Serconfig.version_number_ = version_number
        rec_data = 0x00
        Burning_Flags = 0
        Bootloader_func.Iap_Cmd.APP_data = []
        Bootloader_func.Iap_Cmd.index = 0
        print("烧录流程完全重置")

    def WithoutAPPUpdata(self):
        global Burning_Flags
        global Serconfig
        global rec_data
        Userdata.progress_count = 0
        Userdata.Sendcomplete = 0
        for i in range(int(Userdata.Burning_time)):
            Burning_Flags = 1
            while Burning_Flags == 1:
                Bootloader_func.Request_version_number()
                data = Bootloader_func.Iap_Cmd.Request_version_numbers
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.006)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 2:
                    break
            Userdata.progress_count += 0
            while Burning_Flags == 2:
                Bootloader_func.Request_software_reset()
                data = Bootloader_func.Iap_Cmd.Request_software
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.006)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 3:
                    break
            Userdata.progress_count += 1
            while Burning_Flags == 3:
                Bootloader_func.Request_package_size()
                data = Bootloader_func.Iap_Cmd.Request_package_size
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.006)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 4:
                    break
            Userdata.progress_count += 1
            Bootloader_func.size_package()
            data = Bootloader_func.Iap_Cmd.size_package
            bytes_hex = bytes.fromhex(data)
            time.sleep(0.001)
            Userdata.dataCount += 1
            self.Func_Ser.write(bytes_hex)
            time.sleep(0.006)
            self.read_serial_data()
            Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
            Serconfig.Recdata.append(rec_data)
            Userdata.progress_count += 1
            Bootloader_func.Deliver_data()
            data = 0
            for i in range(Bootloader_func.Iap_Cmd.index + 1):
                data = Bootloader_func.Iap_Cmd.APP_data[i]
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.2)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                Userdata.progress_count += 1.8
            Bootloader_func.COPY_file()
            data = Bootloader_func.Iap_Cmd.COPY_file
            bytes_hex = bytes.fromhex(data)
            time.sleep(0.01)
            Userdata.dataCount += 1
            self.Func_Ser.write(bytes_hex)
            time.sleep(0.1)
            self.read_serial_data()
            Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
            Serconfig.Recdata.append(rec_data)
            time.sleep(2)
            Userdata.progress_count += 1
            while Burning_Flags == 4:
                data = Bootloader_func.Iap_Cmd.Reboot
                bytes_hex = bytes.fromhex(data)
                time.sleep(0.001)
                Userdata.dataCount += 1
                self.Func_Ser.write(bytes_hex)
                time.sleep(0.01)
                self.read_serial_data()
                Serconfig.Senddata.append(hex(int.from_bytes(bytes_hex, 'big')))
                Serconfig.Recdata.append(rec_data)
                if Burning_Flags == 5:
                    break
            Userdata.progress_count += 0.6
            Userdata.Count_finish += 1
            Userdata.Sendcomplete = 1

        Listdata.idSucceed = '烧录状态： 烧录完成'
        print('烧录完成')
        time.sleep(5)
        version_number = self.get_version()
        Serconfig.version_number_ = version_number
        rec_data = 0x00
        Burning_Flags = 0
        Bootloader_func.Iap_Cmd.APP_data = []
        Bootloader_func.Iap_Cmd.index = 0
        print("烧录流程完全重置")

    # 读取串口数据
    def read_serial_data(self):
        global Burning_Flags
        global rec_data
        frame_header = b'\x55'  # 定义帧头
        raw_data = self.Func_Ser.read_all()
        rec_data = raw_data.hex()
        if raw_data or Debug:
            if(Burning_Flags == 0):# 重启
                Burning_Flags = 1
            elif(Burning_Flags == 1):# 请求版本
                Burning_Flags = 2
            elif(Burning_Flags == 2):# 请求软件
                data_Length_reset = len(raw_data)
                if raw_data.startswith(frame_header) and data_Length_reset == 12:
                    print('Pass')
                    Burning_Flags = 3
            elif(Burning_Flags == 3):# 请求分包
                data_Length_pack = len(raw_data)
                if raw_data.startswith(frame_header) and data_Length_pack == 14:
                    print('Pass')
                    Burning_Flags = 4
            elif(Burning_Flags == 4):# 再重启
                Burning_Flags = 5
        raw_data = 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    UI = IAP()
    UI.show()
    sys.exit(app.exec())
