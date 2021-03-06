# 这里将H直接近似为对角线为1的矩阵
# 10.21 能运行二值化的图像，测试集正确率在59%左右
import time
import numpy as np
import math
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import tools as tools
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import xlsxwriter

class lightweightPilae(object):
    def __init__(self, p, k):
        self.p = p
        self.k = k

        # 用于存储变量的list
        self.we = 0
        self.softmaxTrainAcc = []
        self.softmaxTestAcc = []
        self.dim = []
        self.rank = []


    # 专利一种基于伪逆学习快速训练自编码器的近似方法
    # inputX是d行N列的矩阵，d是特征数，N是总样本量
    # layer_th 是目前在第几层网络
    def svdfree(self, inputX):
        dim = inputX.shape[0]
        self.dim.append(dim)
        k = self.k
        p = self.p
        N = inputX.shape[1]
        H0 = np.hstack((np.eye(p), np.zeros((p, N-p))))
        pinvH0 = np.vstack((np.eye(p) * (1/1.0 + k), np.zeros((N-p, p))))
        wd = inputX.dot(pinvH0)
        we0 = wd.T
        xxt = inputX.dot(inputX.T) # 这里是d*d的矩阵
        we = self.norm(we0, xxt, k, p)
        return we

    def norm(self, we0, xxt, k, p):
        v = np.eye(xxt.shape[0])
        value, vector = self.qr_iteration(xxt, v)
        value.flags.writeable = True
        # 这里的参数可以调整！！
        value[value < 1e-5] = 0
        value[value != 0] = 1 / value[value != 0]
        b_tem = np.diag(value)  # 由特征值倒数组成的d*d的矩阵
        U = vector  # U是X*XT的左特征向量
        right = U.dot(b_tem).dot(U.T)  # 右边的三项相乘
        left = (1 + k) * np.eye(p)
        we = left.dot(we0).dot(right)
        return we

    # QR分解算法 返回特征值和特征向量
    def qr_iteration(self, A, v):
        for i in range(100):
            Q, R = np.linalg.qr(A)
            v = np.dot(v, Q)
            A = np.dot(R, Q)
        return np.diag(A), v

    def activeFunc(self, tempH, ):
        tempH[tempH <= 10] = 0
        tempH[tempH > 10] = 1
        return tempH

    # 这一步的目的是获取权重，然后调用分类器
    # 训练的时候把训练集和测试集的x都放进去，保证同分布
    def trainSvdfree(self, trainX):
        t1 = time.time()
        we = self.svdfree(trainX)
        self.we = we
        # hTem = we.dot(trainX)
        # diag_plot(hTem)
        # aTem = get_offDiag(hTem)
        # offDiag_plot(aTem)
        # 激活后应该也是近似单位阵的，所以对角线尽可能是1，其他地方是0，带偏移的激活函数将两者分开
        # H = self.activeFunc(hTem)
        # trainX = H
        # print("train auto-encoders cost time {:.5f}s".format(time.time() - t1))

    def classifier(self, trainX, trainY, testX, testY):
        # trainX = self.activeFunc(self.we.dot(trainX))
        # testX = self.activeFunc(self.we.dot(testX))
        trainX = self.we.dot(trainX)
        testX = self.we.dot(testX)
        train_acc, test_acc = self.predict_softmax(trainX.T, trainY, testX.T, testY)
        return train_acc, test_acc
        # trainX.T是N*d的矩阵

    # 训练逻辑回归
    def train_softmax(self, train_X, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)
        model.fit(train_X, train_y)
        return model

    # 测试逻辑回归
    def predict_softmax(self, train_X, train_y, test_X, test_y):
        model = self.train_softmax(train_X, train_y)
        train_y_predict = model.predict(train_X)
        test_y_predict = model.predict(test_X)
        train_acc = accuracy_score(train_y, train_y_predict) * 100
        test_acc = accuracy_score(test_y, test_y_predict) * 100
        self.softmaxTrainAcc.append(train_acc)
        self.softmaxTestAcc.append(test_acc)
        # print("Softmax Train accuracy:{}% | Test accuracy:{}%".format(train_acc, test_acc))
        return train_acc, test_acc
        # test_recall_score = recall_score(test_y, test_y_predict, average='micro') * 100
        # test_f1_score = f1_score(test_y, test_y_predict, average='micro') * 100
        # test_classification_report = classification_report(test_y, test_y_predict)
        # print("test recall:{}, f1_score:{}".format(test_recall_score, test_f1_score))
        # print(self.test_classification_report)

'''取出非对角线值并画直方图'''
def get_offDiag(e_tem):
    a_tem=[] # 存非对角线所有值
    for i in range(len(e_tem)):
        for j in range(len(e_tem[0])):
            if i!=j:
                a_tem.append(e_tem[i][j])
    return a_tem

def offDiag_plot(a_tem):
    non_bins = np.linspace(min(a_tem), max(a_tem), 300)  # 设置范围和分段
    plt.hist(a_tem, non_bins)
    x_major_locator = MultipleLocator(1)  # x标度
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  # 设置x标度
    plt.show()

'''取出对角线值并画直方图'''
def diag_plot(e_tem):
    diag_e = np.diag(e_tem)  # 取对角线
    print('对角线长度',len(diag_e))
    bins = np.linspace(min(diag_e),max(diag_e), 50)
    plt.hist(diag_e, bins)
    x_major_locator2 = MultipleLocator(1)
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(x_major_locator2)
    plt.show()

if __name__ == '__main__':
    k = 0.7  # 正则化系数 在提取特征值时好像消去了，在最后一层的解码器有用
    p = [100]  # 第一层隐层神经元数
    d = 196
    N = 60000  # 训练样本数量n
    testNum = 10000  # 测试样本数量
    # 若是mnist，特征值为784，白化除于255
    X, y = tools.load_all_npz(r'./dataset/mnist/mnistBiWv.npz')
    X_train, y_train, X_test, y_test = tools.split_dataset(X, y, 0.85)
    X_train = X_train.T # 此时的数据格式d*N
    X_test = X_test.T
    trainAcc = []
    testAcc = []
    for pp in range(60,150):
        myPilae = lightweightPilae(p=pp, k=k)
        myPilae.trainSvdfree(X_train)
        train_acc, test_acc = myPilae.classifier(X_train, y_train, X_test, y_test)
        trainAcc.append(train_acc)
        testAcc.append(test_acc)
    workbook = xlsxwriter.Workbook("svdfree_mnist_10.20_acc.xlsx")
    worksheet = workbook.add_worksheet()
    for i in range(len(trainAcc)):
        worksheet.write(i, 0, i+60)
        worksheet.write(i, 1, trainAcc[i])
        worksheet.write(i, 2, testAcc[i])
    workbook.close()

    # '''备用，直接测试softmax'''
    # def train_softmax(train_X, train_y):
    #     from sklearn.linear_model import LogisticRegression
    #     model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=200)
    #     model.fit(train_X, train_y)
    #     return model
    #
    # # 测试逻辑回归
    # def predict_softmax(train_X, train_y, test_X, test_y):
    #     model = train_softmax(train_X, train_y)
    #     train_y_predict = model.predict(train_X)
    #     test_y_predict = model.predict(test_X)
    #     train_acc = accuracy_score(train_y, train_y_predict) * 100
    #     test_acc = accuracy_score(test_y, test_y_predict) * 100
    #     print("Softmax Train accuracy:{}% | Test accuracy:{}%".format(train_acc, test_acc))
    # predict_softmax(X_train, y_train, X_test, y_test)
    # print('The lightweight pilae took ', time.time() - t1, 's')












