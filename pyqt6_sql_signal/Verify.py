import crcmod
import binascii


class data_verify:
    # 和校验
    def check_sum_uint8(data: bytes) -> int:
        Request_version_numbers_checksum = 0
        for byte in data:
            Request_version_numbers_checksum += byte
            # 限制结果在0-255之间，模拟C语言中uint8_t的回绕行为
        Request_version_numbers_checksum %= 256
        return Request_version_numbers_checksum
    # 32位 和校驗
    def check_sum_uint32(data: bytes) -> int:
        Request_version_numbers_checksum = 0
        for byte in data:
            Request_version_numbers_checksum += byte
            # 限制结果在0-4294967296之间，模拟C语言中uint32_t的回绕行为
        Request_version_numbers_checksum %= 4294967296
        return Request_version_numbers_checksum
    # CRC校验
    def crc16Add(read):
        crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
        data = bytes.fromhex(read.replace(" ", ""))
        crc_value = crc16(data)
        readcrcout = format(crc_value, '04X').upper()
        print('原始CRC16值:', readcrcout)
        read_with_crc = read + binascii.hexlify(crc_value.to_bytes(2, byteorder='little')).decode('ascii').upper()
        print('增加Modbus_CRC16校验:>>>', read_with_crc)
        return read_with_crc
    # 文本内容是16进制的转成整形
    def Hexadecimal_to_integer(data):
        return int(data, 16)

    # 转换为十六进制并格式化为两位字符串
    def hexadecimal2string(data):
        return hex(data)[2:].zfill(2)

    # 整数转换为一个2字节（16位）的字节串 小端
    def integer2string_little(data):
        return data.to_bytes(2, byteorder='little').hex()

    # 整数转换为一个2字节（16位）的字节串 大端
    def integer2string_big(data):
        return data.to_bytes(2, byteorder='big').hex()
