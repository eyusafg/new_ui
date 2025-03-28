FULL_VERSION = False 
if FULL_VERSION:
    from full_features import show_file_info
else:
    from lite_features import show_file_info
import tkinter as tk
from tkinter import filedialog
import  os
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from Crypto.Util.Padding import pad
AES_BLOCK_SIZE = AES.block_size  # AES 加密数据块大小
AES_KEY_SIZE = 32  # AES 密钥长度，这里使用256位密钥


def generate_key(data,size=AES_KEY_SIZE):
    key_str = data
    key_bytes = key_str.encode('utf-8')  # 转换为字节串
    return key_bytes

def decrypt_data(key, encrypted_data):
    try:
        cipher = AES.new(key, AES.MODE_ECB)
        decrypted_padded_data = cipher.decrypt(encrypted_data)
        decrypted_data = unpad(decrypted_padded_data, AES.block_size, style='pkcs7')
        return decrypted_data
    except ValueError as e:
        print(f"解密失败: 填充验证失败 - {e}")
        return None
    except Exception as e:
        print(f"解密失败: {e}")
        return None

indextxt = 0
def read_txt_file_and_split(file_path):
    global packetstxt
    with open(file_path, 'r', encoding='utf-8') as infile:  # 使用'r'模式打开文件，并指定编码为utf-8
        for line in infile:
            packetstxt.append(line.strip())  # 读取每行，并使用strip()去除行尾的换行符
    return packetstxt

def select_and_split_txt_file():
    global Bin_variable
    global packetstxt
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择文件",
        filetypes=[("Text files", "*.txt")])
    
    if file_path:
        packetstxt = read_txt_file_and_split(file_path)
        print(f"文件 {file_path} 已成功读取，共包含 {len(packetstxt)} 行。")
        # 这里可以进一步处理packets，比如打印出来或进行其他操作
        for index, packet in enumerate(packetstxt):
            print(f"行 {index + 1}: {packet}")
    root.destroy()


class Bin_init:
    def __init__(self,Bin_file_path,Subcontract_address,hex_string_spaced,File_size,EachFile,chunk_size_bytes,packets,\
                 packet_index,inputfinish,file_name,password,uid):
        self.Bin_file_path = Bin_file_path
        self.Subcontract_address = Subcontract_address
        self.hex_string_spaced = hex_string_spaced
        self.File_size = File_size
        self.EachFile = EachFile
        self.chunk_size_bytes = chunk_size_bytes
        self.inputfinish = inputfinish
        self.file_name = file_name
        self.password = password
        self.uid = uid
Bin_variable = Bin_init(0, 0, '00', 0, 0, 0, [], 0 , 0, "暂无", 0 , 0)
packets = []
packetstxt = []

def read_bin_file_and_split(file_path, packet_size=1024):
    global packets
    global Bin_variable
    # data_key = int(Bin_variable.uid,16) 
    packet_index = 0  # 用于跟踪包的索引
    with open(file_path, 'rb') as infile:
        infile.seek(32)   # 加密前端32字节为uid_key以及uid  
        while True:
            packet = infile.read(packet_size)
            if not packet:
                break  # 如果没有更多数据可读，则跳出循环
            packets.append(packet)
            packet_index += 1
    return packets


import secrets

def generate_random_key(key_size: int = 16) -> bytes:
    """
    生成随机的加密密钥
    :param key_size: 密钥长度（默认16字节）
    :return: 随机密钥（字节数据）
    """
    return secrets.token_bytes(key_size)

def encrypt_uid(uid: str, uid_key: bytes) -> bytes:
    """
    使用随机密钥加密UID
    :param uid: 需要加密的UID字符串
    :param uid_key: 随机生成的加密密钥
    :return: 加密后的UID（字节数据）
    """
    cipher = AES.new(uid_key, AES.MODE_ECB)
    padded_uid = pad(uid.encode('utf-8'), AES.block_size, style='pkcs7')
    encrypted_uid = cipher.encrypt(padded_uid)
    return encrypted_uid


def encrypt_data(data):
    '''
    加密数据
    必须要先获取到UID, 如果没有UID, 则使用默认key进行加密
    '''
    global  Bin_variable
    print('Bin_variable.uid', Bin_variable.uid)
    data_key = int(Bin_variable.uid,16)  # 从下位机获取的uid
    print('加密时得 data_key', data_key)
    if data_key:   # 第一次烧录不会反馈uid
        key_content = str(data_key)  # UID
    else:
        key_content = '111111111'  # 默认key

    uid_key = generate_random_key()
    encrypted_uid = encrypt_uid(key_content, uid_key)
    uid_size = len(encrypted_uid)
    uid_key_size = len(uid_key)
    print('uid_size', uid_size)
    print('uid_key_size', uid_key_size)

    keytemp = (key_content * ((AES_KEY_SIZE + len(key_content) - 1) // len(key_content)))[:AES_KEY_SIZE]
    key = keytemp.encode('utf-8')
    try:
        cipher = AES.new(key, AES.MODE_ECB)  # 或者使用更安全的模式，如 AES.MODE_CBC
        padded_data = pad(data, AES.block_size, style='pkcs7')
        encrypted_data = cipher.encrypt(padded_data)
        return uid_key + encrypted_uid + encrypted_data
    except Exception as e:
        print(f"加密失败: {e}")
        return -1
    
def extract_components(combined_data: bytes) -> tuple:
    """
    从整体数据中提取加密密钥、加密后的UID和加密后的数据
    :param combined_data: 拼接后的整体数据
    :return: (加密密钥, 加密后的UID, 加密后的数据)
    """
    uid_key = combined_data[:16]  # 加密密钥（16字节）
    encrypted_uid = combined_data[16:32]  # 加密后的UID（16字节）
    encrypted_data = combined_data[32:]  # 加密后的数据
    return uid_key, encrypted_uid, encrypted_data

def decrypt_uid(encrypted_uid: bytes, uid_key: bytes) -> str:
    """
    解密UID
    :param encrypted_uid: 加密后的UID
    :param uid_key: 加密密钥s
    :return: 解密后的UID字符串
    """
    cipher = AES.new(uid_key, AES.MODE_ECB)
    decrypted_uid = unpad(cipher.decrypt(encrypted_uid), AES.block_size)
    # print("decrypted_uid", decrypted_uid)
    return decrypted_uid.decode('utf-8')

def select_and_encrypt_file():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    encrypt_file_path = filedialog.askopenfilename(
        title="选择要加密的bin文件",
        filetypes=[("Encrypted Binary files", "*.bin"), ("All files", "*.*")]
    )
    if encrypt_file_path:
        with open(encrypt_file_path, 'rb') as f:
            encrypt_file = f.read()
        encrypted_file = encrypt_data(encrypt_file)
        if encrypted_file is not None:
            output_file_path = encrypt_file_path.rsplit('.', 1)[0] + ".enc.bin"
            with open(output_file_path, 'wb') as outfile:
                outfile.write(encrypted_file)
            print(f"Encrypted data has been saved to {output_file_path}")   
            return True
    else:
        return False

def select_to_decrypt_file():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    encrypted_file_path = filedialog.askopenfilename(
        title="选择要解密的加密bin文件",
        filetypes=[("DEcrypted Binary files", "*.bin"), ("All files", "*.*")])
    # root.destroy()
    
    if encrypted_file_path:
        Bin_variable.Bin_file_path = encrypted_file_path
        return encrypted_file_path
    else:
        return None

# def show_file_info(filename):
#     root =tk.Tk()
#     root.withdraw()  # 隐藏主窗口
#     # 显示询问对话框
#     if 'collar_machine' in filename:
#         response = messagebox.askyesno("提示", "这是一个圆领固件文件，是否要进行烧录？")
#     else:
#         response = messagebox.askyesno("提示", "这是一个罗纹下摆固件文件，是否要进行烧录？")
#     return response

def encrypt_file():   
    '''
    导入烧录文件
    '''
    global  Bin_variable
    global  packets
    print('Bin_variable.uid', Bin_variable.uid)
    data_key = int(Bin_variable.uid,16)  # 从下位机获取的uid
    print('烧录时的 data_key', data_key)
  
    packets = []
    encrypted_file_path = select_to_decrypt_file()

    ret = show_file_info(encrypted_file_path)
    if ret:
        if '.enc' in encrypted_file_path:
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()  # 读取加密数据
                # 解密
                extracted_uid_key, extracted_encrypted_uid, _ = extract_components(encrypted_data)
                decrypted_uid = decrypt_uid(extracted_encrypted_uid, extracted_uid_key)
                print("解密后的UID:", decrypted_uid)
                if decrypted_uid != str(data_key): # 进行UID校验
                    print("UID不匹配，解密失败")
                    return -2  # 可以返回某个错误mark，用于显示错误窗口
        else:
            if encrypted_file_path:
                with open(encrypted_file_path, 'rb') as f:
                    data = f.read()  
                encrypted_data = encrypt_data(data)
                encrypted_file_path = encrypted_file_path.rsplit('.', 1)[0] + ".enc.bin"
                if encrypted_data != -1:
                    with open(encrypted_file_path, 'wb') as outfile:
                        outfile.write(encrypted_data)
                else:
                    return -1  # 加密失败，返回-1

        Bin_variable.inputfinish = 0
        if encrypted_file_path:
            Bin_variable.file_name = os.path.basename(encrypted_file_path)
            Bin_variable.File_size = os.path.getsize(encrypted_file_path) - 32
            hex_string_big_endian = Bin_variable.File_size.to_bytes(4, byteorder='little').hex()
            Bin_variable.hex_string_spaced = hex_string_big_endian
            packets = read_bin_file_and_split(encrypted_file_path)
            print(f"文件 {encrypted_file_path} 已成功分割成 {len(packets)} 个包。")
        Bin_variable.inputfinish = 1
        return True
    else:
        return -3


if __name__ == '__main__':
    ret = encrypt_file()
    print(ret)
