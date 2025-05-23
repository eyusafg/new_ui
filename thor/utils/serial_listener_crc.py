import struct
import time
from PyQt5.QtCore import QThread, pyqtSignal
import serial


class SerialListener_crc(QThread):
    '''
    增强版串口监听，支持CRC-8/ROHC校验和双浮点数解析
    '''
    serial_signal = pyqtSignal(str, int, int)          # 对齐信号
    serial_signal_w_cloth = pyqtSignal(str, float, float)       # 新增：同时发送宽度和深度

    CMD_CONFIG = {
        0x0100: {'length': 4, 'format': '<I'},   # 对齐次数
        0x0104: {'length': 4, 'format': '<I'},
        0x0102: {'length': 8, 'format': '<2f'},   # 修改：双浮点数
    }

    def __init__(self, ser, logger):
        super().__init__()
        self.ser = ser
        self.running = True
        self.logger = logger
        self.data_buffer = bytearray()
        self.last_data_time = None
        self.frame_timeout = 20  # 秒
        self.SOF = 0xA5

    def crc8_rohc(self, data):
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
        return crc

    def process_frame(self, cmd_id, values):
        """处理解析成功的帧"""
        if cmd_id == 0x0100:
            self.logger.info(f'对齐次数: {values[0]}')
            self.serial_signal.emit('head', 0, 0)
        elif cmd_id == 0x0102:
            width, depth = values
            print(f'布料参数 - 宽度: {width:.2f}mm, 深度: {depth:.2f}mm')
            self.logger.info(f'布料参数 - 宽度: {width:.2f}mm, 深度: {depth:.2f}mm')
            self.serial_signal_w_cloth.emit('cloth_w', width, depth)
        elif cmd_id == 0x0104:
            self.logger.info(f'尾部对齐次数: {values[0]}')
            self.serial_signal.emit('tail', int(values[0]), 0)

    def unpack_frame(self):
        """增强解帧逻辑"""
        while True:
            # 查找有效帧头
            sof_pos = self.data_buffer.find(self.SOF)
            if sof_pos == -1:
                self.data_buffer.clear()
                return None
            
            # 清理无效前缀
            if sof_pos > 0:
                self.logger.warning(f'清理无效前缀: {sof_pos}字节')
                del self.data_buffer[:sof_pos]
                continue

            # 检查最小帧头长度
            if len(self.data_buffer) < 5:  # SOF(1)+CMD_ID(2)+LEN(1)+H_CRC(1)
                return None

            # 解析基础信息
            try:
                cmd_id = int.from_bytes(self.data_buffer[1:3], 'little')
                data_length = self.data_buffer[3]
                header_crc = self.data_buffer[4]
            except IndexError:
                return None

            # 校验帧头CRC
            header_data = self.data_buffer[0:4]  # SOF+CMD_ID+LEN

            if self.crc8_rohc(header_data) != header_crc:
                self.logger.error(f'帧头校验失败 CMD:0x{cmd_id:04X}')
                del self.data_buffer[:5]
                continue

            # 验证指令有效性
            if cmd_id not in self.CMD_CONFIG:
                self.logger.error(f'未知指令: 0x{cmd_id:04X}')
                del self.data_buffer[:5]
                continue

            # 验证数据长度
            expected_length = self.CMD_CONFIG[cmd_id]['length']
            if data_length != expected_length:
                self.logger.error(f'长度错误 CMD:0x{cmd_id:04X} 期望:{expected_length} 实际:{data_length}')
                del self.data_buffer[:5]
                continue

            # 计算完整帧长度
            full_length = 5 + data_length + 1  # 头5字节 + 数据长度 + 数据CRC
            if len(self.data_buffer) < full_length:
                return None

            # 提取数据段
            data_start = 5
            data_end = data_start + data_length
            data_payload = self.data_buffer[:data_end]  # 将整段数据帧进行crc校验  
            # data_payload = self.data_buffer[data_start:data_end]
            data_crc = self.data_buffer[data_end]

            # 校验数据CRC
            if self.crc8_rohc(data_payload) != data_crc:
                self.logger.error(f'数据校验失败 CMD:0x{cmd_id:04X}')
                del self.data_buffer[:full_length]
                continue
            
            data = self.data_buffer[data_start:data_end]
            # 解析数据
            try:
                values = struct.unpack(self.CMD_CONFIG[cmd_id]['format'], data)
            except struct.error as e:
                self.logger.error(f'数据解析错误: {e}')
                del self.data_buffer[:full_length]
                continue

            # 清理缓冲区
            del self.data_buffer[:full_length]
            return (cmd_id, values)

    def run(self):
        try:
            self.ser.reset_input_buffer()
            while self.running:
                # 读取数据
                if self.ser.in_waiting:
                    self.data_buffer.extend(self.ser.read(self.ser.in_waiting))
                    self.last_data_time = time.time()

                # 处理帧
                while True:
                    result = self.unpack_frame()
                    if not result:
                        break
                    cmd_id, values = result
                    self.process_frame(cmd_id, values)

                # 处理超时
                # if self.last_data_time and (time.time() - self.last_data_time > self.frame_timeout):
                #     self.logger.error('接收超时，清空缓冲区')
                #     self.data_buffer.clear()
                #     self.serial_signal.emit('error', 0)

                # time.sleep(0.01)

        except serial.SerialException as e:
            self.logger.critical(f'串口异常: {e}')
        finally:
            self.ser.close()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def build_frame(self, cmd_id, values):
        header = bytes([0xA5]) + cmd_id.to_bytes(2, 'little') + bytes([len(values)*4])
        header_crc = self.crc8_rohc(header)
        
        data = b''.join(struct.pack('<f', v) for v in values)
        data_crc = self.crc8_rohc(data)
        
        return header + bytes([header_crc]) + data + bytes([data_crc])


if __name__ == '__main__':
 
    # data = b'123456789'
    # crc = SerialListener_crc.crc8_rohc(data)
    # print(f'CRC-8/ROHC校验: {crc:02X}')

    s_crc = SerialListener_crc(None, None, bytearray())
    # 生成测试帧（宽度3.5，深度2.1）
    test_frame = s_crc.build_frame(0x0102, [3.5, 2.1])
    print('test_frame: ', test_frame)

    s_crc = SerialListener_crc(None, None, test_frame)

    cmd_id, values = s_crc.unpack_frame()
    print(f'cmd_id: {cmd_id}, values: {values}')
    s_crc.process_frame(cmd_id, values)
