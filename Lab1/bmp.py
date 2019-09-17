import struct
import numpy as np


def one_bytes_trans(fr):
    x = struct.unpack('B', fr)  # 1字节
    return x[0]


def two_bytes_trans(fr):
    x = struct.unpack('H', fr)  # 2字节
    return x[0]


def four_bytes_trans(fr):
    x = struct.unpack('I', fr)  # 4字节
    return x[0]


def trans_one_bytes(fw):
    x = struct.pack('B', int(fw))  # 1字节
    return x


def trans_two_bytes(fw):
    x = struct.pack('H', int(fw))  # 2字节
    return x


def trans_four_bytes(fw):
    x = struct.pack('I', int(fw))  # 4字节
    return x


class BmpResolve():
    # 解析.bmp文件
    def parse(self, file_name):
        file = open(file_name, 'rb')
        # BMP文件头（14字节）
        self.bfType = two_bytes_trans(file.read(2))  # 文件的类型，必须为BM(1-2字节）
        self.bfSize = four_bytes_trans(file.read(4))  # 文件的大小，以字节为单位（3-6）
        self.bfReserved1 = two_bytes_trans(file.read(2))  # 文件保留字，必须为0(7-8）
        self.bfReserved2 = two_bytes_trans(file.read(2))  # 文件保留字，必须为0(9-10）
        self.bfOffBits = four_bytes_trans(file.read(4))  # 数据起始位置（11-14）

        # 位图信息头（40字节）
        self.biSize = four_bytes_trans(file.read(4))  # 本结构所占用字节数（15-18）
        self.biWidth = four_bytes_trans(file.read(4))  # 宽度，以像素为单位（19-22）
        self.biHeight = four_bytes_trans(file.read(4))  # 高度，以像素为单位（23-26）
        self.biPlanes = two_bytes_trans(file.read(2))  # 目标设备的级别，必须为1(27-28）
        self.biBitCount = two_bytes_trans(file.read(2))  # 每个像素所需的位数（29-30）
        self.biCompression = four_bytes_trans(file.read(4))  # 压缩类型（31-34）
        self.biSizeImage = four_bytes_trans(file.read(4))  # 位图的大小，字节为单位（35-38）
        self.biXPelsPerMeter = four_bytes_trans(file.read(4))  # 水平分辨率（39-42）
        self.biYPelsPerMeter = four_bytes_trans(file.read(4))  # 垂直分辨率（43-46)
        self.biClrUsed = four_bytes_trans(file.read(4))  # 实际使用的颜色表中的颜色数（47-50）
        self.biClrImportant = four_bytes_trans(file.read(4))  # 重要的颜色数（51-54）

        # 像素大小
        self.pixelNum = (self.bfSize - self.bfOffBits) // (
            self.biBitCount // 8)

        len1 = self.pixelNum
        len2 = self.biBitCount // 8
        self.bits = np.empty((len1, len2))  # 位图数据
        for i in range(len1):
            for j in range(len2):
                self.bits[i][j] = one_bytes_trans(file.read(1))
        file.close()

    # 重构.bmp文件
    def generate(self, file_name):
        file = open(file_name, 'wb')
        # 重构文件头
        file.write(trans_two_bytes(self.bfType))
        file.write(trans_four_bytes(self.bfSize))
        file.write(trans_two_bytes(self.bfReserved1))
        file.write(trans_two_bytes(self.bfReserved2))
        file.write(trans_four_bytes(self.bfOffBits))

        # 重构位图信息头
        file.write(trans_four_bytes(self.biSize))
        file.write(trans_four_bytes(self.biWidth))
        file.write(trans_four_bytes(self.biHeight))
        file.write(trans_two_bytes(self.biPlanes))
        file.write(trans_two_bytes(self.biBitCount))
        file.write(trans_four_bytes(self.biCompression))
        file.write(trans_four_bytes(self.biSizeImage))
        file.write(trans_four_bytes(self.biXPelsPerMeter))
        file.write(trans_four_bytes(self.biYPelsPerMeter))
        file.write(trans_four_bytes(self.biClrUsed))
        file.write(trans_four_bytes(self.biClrImportant))

        # 重构像素数据
        len1 = self.bits.shape[0]
        len2 = self.bits.shape[1]
        for i in range(len1):
            for j in range(len2):
                file.write(trans_one_bytes(self.bits[i][j]))
        file.close()
