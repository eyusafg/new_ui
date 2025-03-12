from Verify import data_verify
import Bin_Unpacking

# 全局变量 公用
Protocol_header = 0x55 # 协议头 
version_number = 0x02 # 版本
Type_of_message = 0x02 # 消息类型
Command_group = 0x01 # 命令组
# 请求版本
Equence_Request_version = 0x01 # 请求版本序号
Command_Request_version = 0x01 # 请求版本命令
length_Request_version = (0x00, 0x00) # 请求版本长度
# 请求重置
Equence_reset = 0x03 # 请求版本序号
Command_reset = 0x03 # 请求版本命令
length_reset = (0x00, 0x00) # 请求版本长度
# 请求分包大小
Equence_package_size = 0x04 # 请求版本序号
Command_package_size = 0x04 # 请求版本命令
length_package_size = (0x00, 0x00) # 请求版本长度
# 下发总包大小
Equence_General_size = 0x05 # 请求版本序号
Command_General_size = 0x05 # 请求版本命令
length_General_size = (0x04, 0x00) # 请求版本长度
# 下发文件数据
Command_deliver_data = 0x06 # 请求版本命令
length_deliver_data = (0x04, 0x00) # 请求版本长度

# APP烧录
Equence_APP_burning = 0x01 # 请求版本序号
Command_APP_burning = 0x07 # 请求版本命令
length_APP_burning = (0x04, 0x00) # 请求版本长度

Sendid = 0
Recid = 0
Request_version_numbers_hex_all = 0

class CMD_IAP:
    def __init__(self,Request_version_numbers, Request_software, Request_package_size, size_package, Deliver_data, COPY_file ,\
                 Reboot,APP_data,index):
        self.Request_version_numbers = Request_version_numbers
        self.Request_software = Request_software
        self.Request_package_size = Request_package_size
        self.size_package = size_package
        self.Deliver_data = Deliver_data
        self.COPY_file = COPY_file
        self.Reboot = Reboot  # 重启app地址  40000（十进制）
        self.APP_data = APP_data
        self.index = index
Iap_Cmd = CMD_IAP(0 , 0 , 0 , 0, 0 , 0 , '01069c4000AA2631',[], 0)   
# 01069c4400AA67F0
# 请求版本
def Request_version_number():
    global Iap_Cmd
    global Sendid
    global Recid
    global Request_version_numbers_hex_all
    Protocol_header_edit = Protocol_header
    Version_edit_data = version_number
    Sender_ID_edit_data = Equence_Request_version
    End_ID_edit_data =    Equence_Request_version
    Sender_ID_edit_data = data_verify.Hexadecimal_to_integer(Sendid)
    End_ID_edit_data =    data_verify.Hexadecimal_to_integer(Recid)
    Type_of_message_edit_data = Type_of_message
    Equence_number_edit_data = Equence_Request_version
    Command_group_edit_data = Command_group
    Command_edit_data = Command_Request_version
    length_edit_data = length_Request_version
    # 元素列表
    Request_version_numbers_elements = [
        f"{Protocol_header_edit:02X}",
        f"{Version_edit_data:02X}",
        f"{Sender_ID_edit_data:02X}",
        f"{End_ID_edit_data:02X}",
        f"{Type_of_message_edit_data:02X}",
        f"{Equence_number_edit_data:02X}",
        f"{Command_group_edit_data:02X}",
        f"{Command_edit_data:02X}",
    ]
    # 使用列表推导式来格式化length_edit_data1中的每个元素，并添加到hex_string_parts列表中
    Request_version_numbers_elements.extend(f"{byte:02X}" for byte in length_edit_data)
    Request_version_numbers_string = ''.join(Request_version_numbers_elements)
    # 转 16 进制
    data_bytes = bytes.fromhex(Request_version_numbers_string)
    # 第一次 ADD 校验
    Request_version_numbers_checksum = data_verify.check_sum_uint8(data_bytes)
    Request_version_numbers_hex_checksum = data_verify.hexadecimal2string(Request_version_numbers_checksum)
    Request_version_numbers_hex_string_checksum = Request_version_numbers_string + Request_version_numbers_hex_checksum
    # 第二次 ADD 校验
    Request_version_numbers_hex_string_checksum_second = bytes.fromhex(Request_version_numbers_hex_string_checksum)
    checksum2 = data_verify.check_sum_uint8(Request_version_numbers_hex_string_checksum_second)
    hex_checksum2 = data_verify.hexadecimal2string(checksum2)
    Request_version_numbers_hex_all = Request_version_numbers_hex_string_checksum + hex_checksum2
    Iap_Cmd.Request_version_numbers = Request_version_numbers_hex_all
# 请求软件复位
def Request_software_reset():
    global Iap_Cmd
    Protocol_header_edit = Protocol_header
    Version_edit_data = version_number
    Sender_ID_edit_data = data_verify.Hexadecimal_to_integer(Sendid)
    End_ID_edit_data = data_verify.Hexadecimal_to_integer(Recid)
    Type_of_message_edit_data = Type_of_message
    Equence_number_edit_data = Equence_reset
    Command_group_edit_data = Command_group
    Command_edit_data = Command_reset
    length_edit_data = length_reset
    # 元素列表
    Request_software_reset_elements = [
        f"{Protocol_header_edit:02X}",
        f"{Version_edit_data:02X}",
        f"{Sender_ID_edit_data:02X}",
        f"{End_ID_edit_data:02X}",
        f"{Type_of_message_edit_data:02X}",
        f"{Equence_number_edit_data:02X}",
        f"{Command_group_edit_data:02X}",
        f"{Command_edit_data:02X}",
    ]
    # 使用列表推导式来格式化length_edit_data1中的每个元素，并添加到hex_string_parts列表中
    Request_software_reset_elements.extend(f"{byte:02X}" for byte in length_edit_data)
    Request_software_reset_string = ''.join(Request_software_reset_elements)

    # 转 16 进制
    data_bytes = bytes.fromhex(Request_software_reset_string)
    # 第一次 ADD 校验
    Request_software_reset_checksum = data_verify.check_sum_uint8(data_bytes)
    Request_software_reset_hex_checksum = data_verify.hexadecimal2string(Request_software_reset_checksum)
    Request_software_reset_hex_string_checksum = Request_software_reset_string + Request_software_reset_hex_checksum

    # 第二次 ADD 校验
    Request_software_reset_hex_string_checksum_second = bytes.fromhex(Request_software_reset_hex_string_checksum)
    checksum2 = data_verify.check_sum_uint8(Request_software_reset_hex_string_checksum_second)
    hex_checksum2 = data_verify.hexadecimal2string(checksum2)
    Request_software_reset_hex_all = Request_software_reset_hex_string_checksum + hex_checksum2
    Iap_Cmd.Request_software = Request_software_reset_hex_all
# 请求分包大小
def Request_package_size():
    global Iap_Cmd
    Protocol_header_edit = Protocol_header
    Version_edit_data = version_number
    Sender_ID_edit_data = data_verify.Hexadecimal_to_integer(Sendid)
    End_ID_edit_data = data_verify.Hexadecimal_to_integer(Recid)
    Type_of_message_edit_data = Type_of_message
    Equence_number_edit_data = Equence_package_size
    Command_group_edit_data = Command_group
    Command_edit_data = Command_package_size
    length_edit_data = length_package_size
    # 元素列表
    Request_package_size_elements = [
        f"{Protocol_header_edit:02X}",
        f"{Version_edit_data:02X}",
        f"{Sender_ID_edit_data:02X}",
        f"{End_ID_edit_data:02X}",
        f"{Type_of_message_edit_data:02X}",
        f"{Equence_number_edit_data:02X}",
        f"{Command_group_edit_data:02X}",
        f"{Command_edit_data:02X}",
    ]
    # 使用列表推导式来格式化length_edit_data1中的每个元素，并添加到hex_string_parts列表中
    Request_package_size_elements.extend(f"{byte:02X}" for byte in length_edit_data)
    Request_package_size_string = ''.join(Request_package_size_elements)

    # 转 16 进制
    data_bytes = bytes.fromhex(Request_package_size_string)

    # 第一次 ADD 校验
    Request_package_size_checksum = data_verify.check_sum_uint8(data_bytes)
    Request_package_size_hex_checksum = data_verify.hexadecimal2string(Request_package_size_checksum)
    Request_package_size_hex_string_checksum = Request_package_size_string + Request_package_size_hex_checksum

    # 第二次 ADD 校验
    Request_package_size_hex_string_checksum_second = bytes.fromhex(Request_package_size_hex_string_checksum)
    checksum2 = data_verify.check_sum_uint8(Request_package_size_hex_string_checksum_second)
    hex_checksum2 = data_verify.hexadecimal2string(checksum2)
    Request_package_size_hex_all = Request_package_size_hex_string_checksum + hex_checksum2
    Iap_Cmd.Request_package_size = Request_package_size_hex_all
# 请求总包大小
def size_package():
    global Iap_Cmd
    Protocol_header_edit = Protocol_header
    Version_edit_data = version_number
    Sender_ID_edit_data = data_verify.Hexadecimal_to_integer(Sendid)
    End_ID_edit_data = data_verify.Hexadecimal_to_integer(Recid)
    Type_of_message_edit_data = Type_of_message
    Equence_number_edit_data = Equence_General_size
    Command_group_edit_data = Command_group
    Command_edit_data = Command_General_size
    length_edit_data = length_General_size

    # 元素列表
    size_package_elements = [
        f"{Protocol_header_edit:02X}",
        f"{Version_edit_data:02X}",
        f"{Sender_ID_edit_data:02X}",
        f"{End_ID_edit_data:02X}",
        f"{Type_of_message_edit_data:02X}",
        f"{Equence_number_edit_data:02X}",
        f"{Command_group_edit_data:02X}",
        f"{Command_edit_data:02X}",
    ]

    # 使用列表推导式来格式化length_edit_data1中的每个元素，并添加到hex_string_parts列表中
    size_package_elements.extend(f"{byte:02X}" for byte in length_edit_data)
    size_package_string = ''.join(size_package_elements)

    # 转 16 进制
    data_bytes = bytes.fromhex(size_package_string)

    # 第一次 ADD 校验
    size_package_checksum = data_verify.check_sum_uint8(data_bytes)
    size_package_hex_checksum = data_verify.hexadecimal2string(size_package_checksum)
    size_package_hex_string_checksum = size_package_string+size_package_hex_checksum+Bin_Unpacking.Bin_variable.hex_string_spaced
    # 第二次 ADD 校验
    size_package_hex_string_checksum_second = bytes.fromhex(size_package_hex_string_checksum)
    checksum2 = data_verify.check_sum_uint8(size_package_hex_string_checksum_second)
    hex_checksum2 = data_verify.hexadecimal2string(checksum2)
    size_package_hex_all = size_package_hex_string_checksum + hex_checksum2
    Iap_Cmd.size_package = size_package_hex_all
def Deliver_data():
    global Iap_Cmd
    Equence_deliver_data = 0x06  # 请求版本序号
    File_size = 0
    offset_finall = 0
    Address_offset = 0
    sequence_number = 0  # 请求版本序号，通常从外部传入或作为类属性
    for index, packet in enumerate(Bin_Unpacking.packets):
        file_size = len(Bin_Unpacking.packets[index])
        Protocol_header_edit = Protocol_header
        Version_edit_data = version_number
        print('Sendid', Sendid)
        Sender_ID_edit_data = data_verify.Hexadecimal_to_integer(Sendid)
        End_ID_edit_data = data_verify.Hexadecimal_to_integer(Recid)
        Type_of_message_edit_data = Type_of_message
        Equence_number_edit_data = Equence_deliver_data
        Equence_deliver_data += 1
        Command_group_edit_data = Command_group
        Command_edit_data = Command_deliver_data
        File_size = file_size + 4
        length_edit_data = data_verify.integer2string_little(File_size)
        # 元素列表
        Deliver_data_elements = [
            f"{Protocol_header_edit:02X}",
            f"{Version_edit_data:02X}",
            f"{Sender_ID_edit_data:02X}",
            f"{End_ID_edit_data:02X}",
            f"{Type_of_message_edit_data:02X}",
            f"{Equence_number_edit_data:02X}",
            f"{Command_group_edit_data:02X}",
            f"{Command_edit_data:02X}",
            length_edit_data,
        ]
        # 合并
        Deliver_data_string = ''.join(Deliver_data_elements)

        # 转 16 进制
        data_bytes = bytes.fromhex(Deliver_data_string)
        # 第一次 ADD 校验
        Deliver_data_checksum = data_verify.check_sum_uint8(data_bytes)
        Deliver_data_hex_checksum = data_verify.hexadecimal2string(Deliver_data_checksum)
        Deliver_data_hex_string_checksum = Deliver_data_string + Deliver_data_hex_checksum
        data_hex = offset_finall.to_bytes(4, byteorder='little').hex() + Bin_Unpacking.packets[index].hex()
        Deliver_data_hex_string_checksum = Deliver_data_hex_string_checksum + data_hex
        Deliver_data_hex_string_checksum_second = bytes.fromhex(Deliver_data_hex_string_checksum)
        checksum2 = data_verify.check_sum_uint8(Deliver_data_hex_string_checksum_second)
        hex_checksum2 = data_verify.hexadecimal2string(checksum2)
        Deliver_data_hex_all = Deliver_data_hex_string_checksum + hex_checksum2
        Iap_Cmd.APP_data.append(Deliver_data_hex_all)
        Iap_Cmd.index = index
        Address_offset += file_size
        offset_finall = Address_offset

def COPY_file():
    global Iap_Cmd
    Bin_file_path = Bin_Unpacking.Bin_variable.Bin_file_path
    with open(Bin_file_path, 'rb') as file:
        COPY_data = file.read()
    Protocol_header_edit = Protocol_header
    Version_edit_data = version_number
    Sender_ID_edit_data = data_verify.Hexadecimal_to_integer(Sendid)
    End_ID_edit_data = data_verify.Hexadecimal_to_integer(Recid)
    Type_of_message_edit_data = Type_of_message
    Equence_number_edit_data = Equence_APP_burning
    Command_group_edit_data = Command_group
    Command_edit_data = Command_APP_burning
    length_edit_data = length_APP_burning

    # 元素列表
    COPY_file_elements = [
        f"{Protocol_header_edit:02X}",
        f"{Version_edit_data:02X}",
        f"{Sender_ID_edit_data:02X}",
        f"{End_ID_edit_data:02X}",
        f"{Type_of_message_edit_data:02X}",
        f"{Equence_number_edit_data:02X}",
        f"{Command_group_edit_data:02X}",
        f"{Command_edit_data:02X}",
    ]

    # 使用列表推导式来格式化length_edit_data1中的每个元素，并添加到hex_string_parts列表中
    COPY_file_elements.extend(f"{byte:02X}" for byte in length_edit_data)
    COPY_file_string = ''.join(COPY_file_elements)

    # 转 16 进制
    data_bytes = bytes.fromhex(COPY_file_string)

    # 第一次 ADD 校验
    COPY_file_checksum = data_verify.check_sum_uint8(data_bytes)
    COPY_file_hex_checksum = data_verify.hexadecimal2string(COPY_file_checksum)
    COPY_SUM32 = data_verify.check_sum_uint32(COPY_data)
    # 组合payload
    COPY_file_hex_string_checksum = COPY_file_string + COPY_file_hex_checksum + COPY_SUM32.to_bytes(4, byteorder='little').hex()
    # 第二次 ADD 校验
    COPY_file_hex_string_checksum_second = bytes.fromhex(COPY_file_hex_string_checksum)
    checksum2 = data_verify.check_sum_uint8(COPY_file_hex_string_checksum_second)
    hex_checksum2 = data_verify.hexadecimal2string(checksum2)
    COPY_file_hex_all = COPY_file_hex_string_checksum + hex_checksum2
    Iap_Cmd.COPY_file = COPY_file_hex_all
