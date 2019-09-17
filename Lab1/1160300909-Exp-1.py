import numpy as np
import os
import math
import bmp
import sys


# 取整
def rounding(num):
    re = num
    if re < 0:
        re = 0
    if re > 255:
        re = 255
    return np.uint8(re)


# 解析文件路径，返回文件名
def resolve_file(path):
    str = path
    pathlist = os.path.split(str)
    filename = pathlist[1].split('.bmp')
    return filename[0]


# RGB to YIQ
def rgb2yiq(filename):
    p = bmp.BmpResolve()
    p.parse(filename)
    bits = p.bits.copy()
    len1 = bits.shape[0]
    len2 = bits.shape[1]
    re = np.empty((len1, len2))  # 数据内容

    for i in range(len1):
        b = bits[i][0]
        g = bits[i][1]
        r = bits[i][2]
        re[i][0] = rounding(0.299 * r + 0.587 * g + 0.114 * b)  # Y
        re[i][1] = rounding(0.596 * r - 0.274 * g - 0.322 * b + 128)  # I
        re[i][2] = rounding(0.211 * r - 0.523 * g + 0.312 * b + 128)  # Q

    p.bits = re
    file = resolve_file(filename) + '-1160300909-YIQ.bmp'
    p.generate(file)


# RGB to HSI
def rgb2hsi(filename):
    p = bmp.BmpResolve()
    p.parse(filename)
    bits = p.bits.copy()
    len1 = bits.shape[0]
    len2 = bits.shape[1]
    re = np.empty((len1, len2))  # 数据内容

    for i in range(len1):
        sum = bits[i][0] + bits[i][1] + bits[i][2]
        if sum == 0:
            b = g = r = 1 / 3
        else:
            b = bits[i][0] / sum
            g = bits[i][1] / sum
            r = bits[i][2] / sum
        x2 = 0
        x1 = 2 * math.sqrt((r - g) * (r - g) + (r - b) * (g - b))
        if x1 != 0:
            x2 = math.acos(round((r - g + r - b) / x1, 6))
        x2 = x2 * (180 / math.pi)
        if g < b:
            x2 = 360 - x2
        re[i][0] = rounding(x2)  # H
        re[i][1] = rounding((1 - 3 * min(r, g, b)) * 100)  # S
        re[i][2] = rounding(sum / 3)  # I

    p.bits = re
    file = resolve_file(filename) + '-1160300909-HSI.bmp'
    p.generate(file)


# RGB to YCbCr
def rgb2ycbcr(filename):
    p = bmp.BmpResolve()
    p.parse(filename)
    bits = p.bits.copy()
    len1 = bits.shape[0]
    len2 = bits.shape[1]
    re = np.empty((len1, len2))  # 数据内容

    for i in range(len1):
        b = bits[i][0]
        g = bits[i][1]
        r = bits[i][2]
        re[i][0] = rounding(0.299 * r + 0.587 * g + 0.114 * b)  # Y
        re[i][1] = rounding(-0.169 * r - 0.331 * g + 0.500 * b + 128)  # Cb
        re[i][2] = rounding(0.500 * r - 0.419 * g - 0.081 * b + 128)  # Cr

    p.bits = re
    file = resolve_file(filename) + '-1160300909-YCbCr.bmp'
    p.generate(file)


# RGB to XYZ
def rgb2xyz(filename):
    p = bmp.BmpResolve()
    p.parse(filename)
    bits = p.bits.copy()
    len1 = bits.shape[0]
    len2 = bits.shape[1]
    re = np.empty((len1, len2))  # 数据内容

    for i in range(len1):
        b = bits[i][0]
        g = bits[i][1]
        r = bits[i][2]
        re[i][0] = rounding(0.412453 * r + 0.357580 * g + 0.180423 * b)  # X
        re[i][1] = rounding(0.212671 * r + 0.715160 * g + 0.072169 * b)  # Y
        re[i][2] = rounding(0.019334 * r + 0.119193 * g + 0.950227 * b)  # Z

    p.bits = re
    file = resolve_file(filename) + '-1160300909-XYZ.bmp'
    p.generate(file)


def run_main():
    filename = sys.argv[1]
    # filename = 'path/to/Test-Image-4.bmp'

    print('正在转换为YIQ......')
    rgb2yiq(filename)

    print('正在转换为HSI......')
    rgb2hsi(filename)

    print('正在转换为YCbCr......')
    rgb2ycbcr(filename)

    print('正在转换为XYZ......')
    rgb2xyz(filename)


if __name__ == '__main__':
    run_main()
    print('Finish！')
