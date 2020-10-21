# 10月20号改进
# 哈尔变换后再二值化
# 需要注意的是哈尔小波变换后ll变成510的，其他高频部分有正值有负值
# 这里用一次哈尔变换后的低频部分
# 最后存储的形式是N*196的7万条数据
# 图像变成了布尔型存储，布尔型也可以直接用于计算
import tools
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from PIL import Image
from skimage.restoration import denoise_wavelet
import cv2
import scipy.misc

# class_name_cifar10=np.load(os.path.join("dataset","batches.meta"))
(X_train, y_train), (X_test, y_test) = tools.load_npz("./dataset/mnist/mnist.npz")
# 初始化数据
X_all = np.vstack((X_train, X_test))
biWvXall = np.bool([len(X_all), 196])
biWvXall = np.full((len(X_all), 196), True)
N = len(X_all)
'''小波变换部分'''
# 要变换的图像
for i in range(N):
    LL1, coeffs2 = pywt.dwt2(X_all[i], 'haar')
    ret4, thresh4 = cv2.threshold(LL1, 255, 1, cv2.THRESH_BINARY)
    thresh4 = thresh4.astype(bool)
    biWvXall[i] = thresh4.reshape(-1, 196)
    # 这里只用低频主要部分的二值化结果
    # for j in range(len(coeffs2)):
    #     ret, thresh1 = cv2.threshold(coeffs2[j], 30, 255, cv2.THRESH_BINARY)

# X_train_wt = biWvXall[:int(0.85*N), :] # 这里乘完0.7之后变成浮点数了，需要转成int形式才能索引出来
# X_test_wt = biWvXall[int(0.85*N):, :]
# print(X_train_wt.shape)
# print(X_test_wt.shape)
yAll = np.concatenate([y_train, y_test])
np.savez(r'./dataset/mnist/mnistBiWv.npz', biWvXall = biWvXall, yAll = yAll)
