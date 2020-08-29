# 将ciar10图片保留三个颜色，再经过哈尔小波变换，保存图片
# XAllWt是N*768
import tools
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from PIL import Image

# class_name_cifar10=np.load(os.path.join("dataset","batches.meta"))
(X_train, y_train), (X_test, y_test) = tools.load_cifar10("haarWavelet_svdfree_518/dataset")

X_all = np.vstack((X_train, X_test))
X_red = np.zeros([X_all.shape[0],1024])
X_green = np.zeros([X_all.shape[0], 1024])
X_blue = np.zeros([X_all.shape[0], 1024])

for i in range(len(X_all)):
    red = X_all[i][0].reshape(-1, 1024)
    X_red[i] = red
    green = X_all[i][1].reshape(-1, 1024)
    X_green[i] = green
    blue = X_all[i][2].reshape(-1, 1024)
    X_blue[i] = blue

    #pic_grab.astype('float32') / 255

'''小波变换部分'''
xRedWt = np.zeros([X_all.shape[0],256])
xGreenWt = np.zeros([X_all.shape[0], 256])
xBlueWt = np.zeros([X_all.shape[0], 256])
for i in range(X_red.shape[0]):
    coeffs2 = pywt.dwt2(X_red[i].reshape(32, 32), 'haar')
    LL1, (LH, HL, HH) = coeffs2
    LL1 = LL1.reshape(-1, 256)
    xRedWt[i] = LL1

    coeffs2 = pywt.dwt2(X_green[i].reshape(32, 32), 'haar')
    LL1, (LH, HL, HH) = coeffs2
    LL1 = LL1.reshape(-1, 256)
    xGreenWt[i] = LL1

    coeffs2 = pywt.dwt2(X_blue[i].reshape(32, 32), 'haar')
    LL1, (LH, HL, HH) = coeffs2
    LL1 = LL1.reshape(-1, 256)
    xBlueWt[i] = LL1

XAllWt = np.hstack((xRedWt, xGreenWt, xBlueWt))
N = XAllWt.shape[0]
# print(N)
X_train_wt = XAllWt[:int(0.7*N), :] # 这里乘完0.7之后变成浮点数了，需要转成int形式才能索引出来
X_test_wt = XAllWt[int(0.7*N):, :]
# print(X_train_wt.shape)
# print(X_test_wt.shape)
np.savez(r'cifar10_wt_3.npz', x_train=X_train_wt, y_train=y_train, x_test=X_test_wt, y_test=y_test)
