import cv2
import numpy as np
import os
import time
import sys


# 解析文件路径，返回文件名
def resolve_file(path):
    str = path
    pathlist = os.path.split(str)
    filename = pathlist[1].split('.')
    return filename[0]


# 磨皮算法
def dermabrasion(img):
    Gaussia_r = 1  # 高斯滤波半径
    guided_r = 3  # 导向滤波半径
    guided_eps = 125  # 导向滤波的正则化项，类似于bilateralFilter中的sigma
    # p = 80  # 透明度
    img2 = img.copy()

    temp1 = cv2.ximgproc.guidedFilter(img2, img2, guided_r, guided_eps)  # 导向滤波
    # dst = cv.ximgproc.guidedFilter(guide, src, radius, eps[, dst[, dDepth]])
    temp2 = (np.uint32(temp1) - np.uint32(img2) + 128)  # 高反差保留，应用图像
    temp2 = np.uint8(np.clip((temp2), 0, 255))  # 类型转换，截断
    temp3 = cv2.GaussianBlur(temp2, (Gaussia_r, Gaussia_r), 0)  #高斯模糊

    temp4 = np.uint32(img2) + 2 * np.uint32(temp3) - 256  # 线性光混合
    # dst = (np.uint32(img)*(100 - p) + temp4*p) / 100  # 透明度
    return np.uint8(np.clip(temp4, 0, 255))


# 肤色检测
def skin_detect(img):
    len1, len2, len3 = img.shape
    result = np.zeros(img.shape, np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    if len1 == 0 or len3 != 3:
        return result

    for i in range(len1):
        for j in range(len2):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            v = hsv[i, j, 2]
            cr = ycrcb[i, j, 1]
            cb = ycrcb[i, j, 2]
            if 128 <= cr <= 173 and 77 <= cb <= 122 and 0 <= h <= 35 and s >= 23 and v >= 90:
                result[i, j, 0] = img[i, j, 0]
                result[i, j, 1] = img[i, j, 1]
                result[i, j, 2] = img[i, j, 2]

    kernel = np.ones((15, 15), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result


# 合成肤色检测后的和经滤波处理后的图像
# pro_img：经滤波器处理后的图像
# det_img：肤色检测后后的图像
# ori_img：原始图像
def synthesis(pro_img, det_img, ori_img):
    for i in range(pro_img.shape[0]):
        for j in range(pro_img.shape[1]):
            if det_img[i, j, 0] == det_img[i, j, 1] == det_img[i, j, 2] == 0:
                pro_img[i, j] = ori_img[i, j]
    return pro_img


# 利用Sobel算子锐化
def sobel_sharpen(img):
    weight = 0.13  # 锐化参数
    len1, len2, len3 = img.shape
    image_out = img.copy()

    for i in range(1, len1 - 1):
        for j in range(1, len2 - 1):
            for k in range(len3):
                a = int(img[i - 1, j - 1, k] + 2 * img[i, j - 1, k] +
                        img[i + 1, j - 1, k])
                b = int(img[i - 1, j + 1, k] + 2 * img[i, j + 1, k] +
                        img[i + 1, j + 1, k])
                c = int(img[i + 1, j - 1, k] + 2 * img[i + 1, j, k] +
                        img[i + 1, j + 1, k])
                d = int(img[i - 1, j - 1, k] + 2 * img[i - 1, j, k] +
                        img[i - 1, j + 1, k])
                image_out[i][j][k] = np.sqrt(
                    np.power(a - b, 2) + np.power(c - d, 2))

    threshold = 50  # 阈值
    image_out[image_out <= threshold] = 0
    image_out = np.uint32(weight * image_out) + np.uint32(img)
    image_out = np.uint8(np.clip(image_out, 0, 255))
    return image_out


# 美白，方法来自于《A Two-Stage Contrast Enhancement Algorithm for Digital Images》一文
def whiten(img):
    beta = 1.8
    white = np.log(img / 255 * (beta - 1) + 1) / np.log(beta)
    white = np.uint8(np.clip(white * 255, 0, 255))
    return white


# 主程序
def run_main():
    filename = sys.argv[1]
    #filename = 'path/to/Test-Image-2.bmp'
    start = time.clock()
    ori_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    fw = resolve_file(filename)  # 解析路径

    out = dermabrasion(ori_img)  # 磨皮
    out = dermabrasion(out)  # 二次磨皮
    pro_img = dermabrasion(out)  # 三次磨皮

    det_img = skin_detect(ori_img)  # 皮肤检测
    out = synthesis(pro_img, det_img, ori_img)  # 合成图像
    out = whiten(out)  # 美白
    out = sobel_sharpen(out)  # 锐化

    filewrite = fw + '_1160300909.bmp'
    cv2.imwrite(filewrite, out)
    print('处理结果已经存入 ' + filewrite + ' 文件')

    end = time.clock()
    print('总用时', end - start, 's')


if __name__ == '__main__':
    print('正在处理......')
    run_main()
    print('完成！')
