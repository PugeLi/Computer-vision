import numpy as np
import cv2
from matplotlib import pyplot as plt
import pylab as pl
import os
import time
import sys


# 调整亮度，权值weight:（-255.255）
def brightness(filename, weight):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR to HSV(HSB)

    brightness = img_hsv[::, ::, 2]
    result = np.int16(img_hsv.copy())
    result[::, ::, 2] = np.int16(brightness) + weight  # B = B + weight

    result[result > 255] = 255  # 截断
    result[result < 0] = 0  # 截断
    result = np.uint8(result)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)  # HSV(HSB) to BGR
    return result


# 调整对比度，系数k>0
def contrast(filename, k):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    len1, len2, len3 = img.shape
    gray = np.dot(img[..., :3], [0.114, 0.587, 0.299])
    mean = np.sum(gray[::, ::]) / (len1 * len2)  # 平均亮度
    result = np.zeros(img.shape, np.int16)  # 记录结果

    for i in range(len1):
        for j in range(len2):
            sum_rgb = np.sum(img[i, j, ::])
            b = g = r = 1 / 3
            if sum_rgb != 0:
                b = img[i][j][0] / sum_rgb
                g = img[i][j][1] / sum_rgb
                r = img[i][j][2] / sum_rgb
            newgray = mean + k * (gray[i][j] - mean)
            newsum = newgray / (b * 0.114 + g * 0.587 + r * 0.299)
            result[i][j][0] = newsum * b
            result[i][j][1] = newsum * g
            result[i][j][2] = newsum * r

    result[result < 0] = 0
    result[result > 255] = 255
    result = np.uint8(result)
    return result


# 调整饱和度，权值weight:（-255.255）
def saturation(filename, weight):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # BGR to HLS

    saturation = img_hls[::, ::, 2]
    result = np.int16(img_hls.copy())
    result[::, ::, 2] = np.int16(saturation) + weight  # S = S + weight

    result[result > 255] = 255  # 截断
    result[result < 0] = 0  # 截断
    result = np.uint8(result)
    result = cv2.cvtColor(result, cv2.COLOR_HLS2BGR)  # HLS to BGR
    return result


# 调整色度，权值weight:（-255.255）
def hue(filename, weight):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # BGR to HLS

    hue = img_hls[::, ::, 0]
    result = np.int16(img_hls.copy())
    result[::, ::, 0] = np.int16(hue) + weight  # H = H + weight

    result[result > 255] = 255  # 截断
    result[result < 0] = 0  # 截断
    result = np.uint8(result)
    result = cv2.cvtColor(result, cv2.COLOR_HLS2BGR)  # HLS to BGR
    return result


# 直方图
def hist(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = np.dot(img[..., :3], [0.114, 0.587, 0.299])
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.reshape(np.uint8(gray), (-1))  # 二维转一维

    hist = np.zeros(256, int)
    for i in gray:
        hist[i] += 1
    histfre = (hist / len(gray)).tolist()

    plt.subplot(311)
    plt.bar(range(0, 256), histfre)  # 频率柱状图
    pl.ylabel("Frequency bar")
    plt.subplot(312)
    plt.hist(gray, bins=255, normed=len(gray))  # 频率直方图
    pl.ylabel("Frequency hist")
    plt.subplot(313)
    plt.hist(gray, bins=255, normed=len(gray), cumulative=True)  # 累计直方图
    pl.ylabel("Cumulative hist")
    pl.xlabel("n (0~255)")
    pl.show()

    return histfre  # 频率


# 读文件
def readimg(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    return img


# 预处理数据，填充最近像素值，以供滤波
def pretreatment_filter(img):
    len1, len2, len3 = img.shape
    image_out = np.zeros((len1 + 2, len2 + 2, len3), np.uint8)

    image_out[1:len1 + 1, 1:len2 + 1] = img  # 内部复制
    image_out[0:1, 1:len2 + 1] = img[0]  # 第一行
    image_out[len1 + 1:len1 + 2, 1:len2 + 1] = img[len1 - 1]  # 最后一行
    image_out[1:len1 + 1, 0:1] = np.reshape(img[:, 0], (len1, 1, 3))  # 第一列
    image_out[1:len1 + 1, len2 + 1:len2 + 2] = np.reshape(
        img[:, len2 - 1], (len1, 1, 3))  # 最后一列

    image_out[0, 0] = img[0, 0]  # 矩阵左上
    image_out[0, len2 + 1] = img[0, len2 - 1]  # 矩阵右上
    image_out[len1 + 1, 0] = img[len1 - 1, 0]  # 矩阵左下
    image_out[len1 + 1, len2 + 1] = img[len1 - 1, len2 - 1]  # 矩阵右下

    return image_out


# 均值滤波n*n窗口
def meanfilter(filename, n):
    k = n // 2
    preinit = pretreatment_filter(readimg(filename))  # 初始化，外部填充一层
    for i in range(k - 1):  # 初始化，外部填充多层
        pre = pretreatment_filter(preinit)
        preinit = pre.copy()

    len1, len2, len3 = preinit.shape
    image_out = preinit.copy()

    for i in range(k, len1 - k):
        for j in range(k, len2 - k):
            image_out[i][j][0] = np.average(
                preinit[i - k:i + k + 1, j - k:j + k + 1, 0])
            image_out[i][j][1] = np.average(
                preinit[i - k:i + k + 1, j - k:j + k + 1, 1])
            image_out[i][j][2] = np.average(
                preinit[i - k:i + k + 1, j - k:j + k + 1, 2])

    image_out = image_out[k:len1 - k, k:len2 - k]  # 取内层
    return image_out


# 中值滤波n*n窗口
def medianfilter(filename, n):
    k = n // 2
    preinit = pretreatment_filter(readimg(filename))  # 初始化，外部填充一层
    for i in range(k - 1):  # 初始化，外部填充多层
        pre = pretreatment_filter(preinit)
        preinit = pre.copy()

    len1, len2, len3 = preinit.shape
    image_out = preinit.copy()

    for i in range(k, len1 - k):
        for j in range(k, len2 - k):
            image_out[i][j][0] = np.median(
                preinit[i - k:i + k + 1, j - k:j + k + 1, 0])
            image_out[i][j][1] = np.median(
                preinit[i - k:i + k + 1, j - k:j + k + 1, 1])
            image_out[i][j][2] = np.median(
                preinit[i - k:i + k + 1, j - k:j + k + 1, 2])

    image_out = image_out[k:len1 - k, k:len2 - k]  # 取内层
    return image_out


# 中值滤波快速算法，n*n窗口
def fast_medianfilter(filename, n):
    k = n // 2
    th = n * n // 2
    preinit = pretreatment_filter(readimg(filename))  # 预处理，外部填充一层
    for i in range(k - 1):  # 外部填充多层
        pre = pretreatment_filter(preinit)
        preinit = pre.copy()
    len1, len2, len3 = preinit.shape
    image_out = preinit.copy()  # 记录结果

    #  每行维护直方图
    def winhist(win):
        for i in np.reshape(win[::, ::, 0], (-1)):
            hist[0][i] += 1
            if i <= medpoint[0]:
                num[0] += 1
        for j in np.reshape(win[::, ::, 1], (-1)):
            hist[1][j] += 1
            if j <= medpoint[1]:
                num[1] += 1
        for k in np.reshape(win[::, ::, 2], (-1)):
            hist[2][k] += 1
            if k <= medpoint[2]:
                num[2] += 1

    medpoint = np.zeros(3, np.int16)  # 记录中值，RGB三通道
    for i in range(k, len1 - k):
        image_out[i][k][0] = medpoint[0] = np.median(
            preinit[i - k:i + k + 1, 0:2 * k + 1, 0])
        image_out[i][k][1] = medpoint[1] = np.median(
            preinit[i - k:i + k + 1, 0:2 * k + 1, 1])
        image_out[i][k][2] = medpoint[2] = np.median(
            preinit[i - k:i + k + 1, 0:2 * k + 1, 2])

        win = preinit[i - k:i + k + 1, 0:2 * k + 1, ::]  # 每行的第一个窗口
        hist = np.zeros((3, 256), int)  # 直方图，RGB三通道
        num = np.zeros(3, int)  # 记录<=中值的数目，RGB三通道
        winhist(win)

        for j in range(k + 1, len2 - k):
            # 处理左列
            def leftdeal(left):
                for x in range(2 * k + 1):
                    for y in range(3):
                        hist[y][left[x][y]] -= 1
                        if left[x][y] <= medpoint[y]:
                            num[y] -= 1

            left = preinit[i - k:i + k + 1, j - k - 1, ::]
            leftdeal(left)

            # 处理右列
            def rightdeal(right):
                for x in range(2 * k + 1):
                    for y in range(3):
                        hist[y][right[x][y]] += 1
                        if right[x][y] <= medpoint[y]:
                            num[y] += 1

            right = preinit[i - k:i + k + 1, j + k, ::]
            rightdeal(right)

            for ii in range(3):
                while num[ii] > th:
                    medpoint[ii] -= 1
                    num[ii] -= hist[ii][medpoint[ii] + 1]
                while num[ii] < th:
                    medpoint[ii] += 1
                    num[ii] += hist[ii][medpoint[ii]]
                image_out[i][j][ii] = medpoint[ii]

    image_out = image_out[k:len1 - k, k:len2 - k]  # 截取内层数据
    return image_out


# Roberts算子
def roberts(filename):
    img_gray = cv2.imread(filename, 0)
    len1, len2 = img_gray.shape
    image_out = np.zeros((len1, len2), np.uint8)

    for i in range(0, len1 - 1):
        for j in range(0, len2 - 1):
            a = int(img_gray[i][j]) - int(img_gray[i + 1][j + 1])
            b = int(img_gray[i + 1][j]) - int(img_gray[i][j + 1])
            image_out[i][j] = np.sqrt(np.power(a, 2) + np.power(b, 2))

    plt.subplot(131), plt.imshow(img_gray, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(image_out, cmap='gray')
    plt.title('Roberts'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(255 - image_out, cmap='gray')
    plt.title('Roberts'), plt.xticks([]), plt.yticks([])
    plt.show()
    return image_out


# Sobel算子
def sobel(filename):
    img_gray = cv2.imread(filename, 0)
    len1, len2 = img_gray.shape
    image_out = np.zeros((len1, len2), np.uint8)

    for i in range(1, len1 - 1):
        for j in range(1, len2 - 1):
            a = int(img_gray[i - 1][j - 1] + 2 * img_gray[i][j - 1] +
                    img_gray[i + 1][j - 1])
            b = int(img_gray[i - 1][j + 1] + 2 * img_gray[i][j + 1] +
                    img_gray[i + 1][j + 1])
            c = int(img_gray[i + 1][j - 1] + 2 * img_gray[i + 1][j] +
                    img_gray[i + 1][j + 1])
            d = int(img_gray[i - 1][j - 1] + 2 * img_gray[i - 1][j] +
                    img_gray[i - 1][j + 1])
            image_out[i][j] = np.sqrt(np.power(a - b, 2) + np.power(c - d, 2))

    threshold = 50  # 阈值
    image_out[image_out <= threshold] = 0

    plt.subplot(131), plt.imshow(img_gray, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(image_out, cmap='gray')
    plt.title('Sobel'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(255 - image_out, cmap='gray')
    plt.title('Sobel'), plt.xticks([]), plt.yticks([])
    plt.show()
    return image_out


# 解析文件路径，返回文件名
def resolve_file(path):
    str = path
    pathlist = os.path.split(str)
    filename = pathlist[1].split('.')
    return filename[0]


def run_main():
    #filename = 'path/to/Test-Image-2.bmp'
    filename = sys.argv[1]
    fw = resolve_file(filename)
    con = 1
    while con == 1:
        print(" 1.  调整亮度")
        print(" 2.  调整对比度")
        print(" 3.  调整饱和度")
        print(" 4.  调整色度")
        print(" 5.  显示直方图")
        print(" 6.  中值滤波")
        print(" 7.  均值滤波")
        print(" 8.  Roberts算子边缘检测")
        print(" 9.  Sobel算子边缘检测")
        print(" 10. 快速中值滤波")

        flag = int(input('\n请输入所选序号（1-9）：'))
        if flag == 1:
            weight = int(input('请输入亮度调整权重（-255,255）：'))
            out = brightness(filename, weight)
            filewrite = fw + '--brightness(weight=' + str(weight) + ').bmp'
            cv2.imwrite(filewrite, out)
            print('已经存入 ' + filewrite + ' 文件')
        elif flag == 2:
            k = float(input('请输入对比度调整倍数（>0）：'))
            out = contrast(filename, k)
            filewrite = fw + '--contrast(k=' + str(k) + ').bmp'
            cv2.imwrite(filewrite, out)
            print('已经存入 ' + filewrite + ' 文件')
        elif flag == 3:
            weight = int(input('请输入饱和度调整权重（-255,255）：'))
            out = saturation(filename, weight)
            filewrite = fw + '--saturation(weight=' + str(weight) + ').bmp'
            cv2.imwrite(filewrite, out)
            print('已经存入 ' + filewrite + ' 文件')
        elif flag == 4:
            weight = int(input('请输入色度调整权重（-255,255）：'))
            out = hue(filename, weight)
            filewrite = fw + '--hue(weight=' + str(weight) + ').bmp'
            cv2.imwrite(filewrite, out)
            print('已经存入 ' + filewrite + ' 文件')
        elif flag == 5:
            hist(filename)
        elif flag == 6:
            n = int(input('请输入窗口大小n（窗口为n*n）：'))
            start = time.clock()
            out = medianfilter(filename, n)
            end = time.clock()
            print('用时 ', end - start, 's')
            filewrite = fw + '--medianfilter(n=' + str(n) + ').bmp'
            cv2.imwrite(filewrite, out)
            print('已经存入 ' + filewrite + ' 文件')
        elif flag == 7:
            n = int(input('请输入窗口大小n（窗口为n*n）：'))
            out = meanfilter(filename, n)
            filewrite = fw + '--meanfilter(n=' + str(n) + ').bmp'
            cv2.imwrite(filewrite, out)
            print('已经存入 ' + filewrite + ' 文件')
        elif flag == 8:
            roberts(filename)
        elif flag == 9:
            sobel(filename)
        elif flag == 10:
            n = int(input('请输入窗口大小n（窗口为n*n）：'))
            start = time.clock()
            out = fast_medianfilter(filename, n)
            end = time.clock()
            print('用时 ', end - start, 's')
            filewrite = fw + '--fast_medianfilter(n=' + str(n) + ').bmp'
            cv2.imwrite(filewrite, out)
            print('已经存入 ' + filewrite + ' 文件')
        else:
            print('输入错误！')
        con = int(input('\n是否继续?\n请输入1(是)或者0(否)：'))

    #omg1 = brightness(filename,50)
    #omg1 = contrast(filename,2.3)
    #omg1 = saturation(filename,80)
    #omg1 = hue(filename,80)
    #out = hist(filename)
    #out = medianfilter(filename, 5)
    #out = meanfilter(filename, 5)
    #out = roberts(filename)
    #out = sobel(filename)
    #out = fast_medianfilter(filename, 5)
    '''
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''


if __name__ == '__main__':
    run_main()
    print('Finish！')
