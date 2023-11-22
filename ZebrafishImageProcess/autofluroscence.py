#import matplotlib.pyplot as plt
#import numpy as np
#环境设置：firstEn
#环境路径：f:\python\anaconda\envs
import czifile as czi
image2 = czi.imread('MdkB_E4_KI_15hpf-02.czi')
global image2
'''
                                                #导入czi文件，测试样例文件大小1，121，2，60，1024，1024，1
                                                #这是一个七维的numpy
                                                #1不知道是什么
                                                #121指的是时间尺度
                                                #2是双通道
                                                #60指的是60层扫描
                                                #1024*1024指的是每张图的2x2
                                                #1可能是指这是一个灰度图，RGB的话会是3
#46开始读入，50结束，预计耗时5分钟

Q1：为什么这里会有两个1作为没有意义的维度？ A1:似乎是单元格的意思？
是否每个czi都有这样的东西，如果是的话，怎么统一不同的czi文件格式
'''

###############文件已经导入，下面开始进行寻找特征点###############
#napari安装命令conda install -c conda-forge napari
#python -m pip install "napari[all]"
import napari
#滤光器安装程序pip install Offset-Subtraction
import Offset_Subtraction as subtract

cells = image2  # grab some data

###############presentation,single layer,offset autofluroscence#############
import napari
import numpy as np
#a = np.zeros((121,2,60,1024,1024))
#内存溢出
from PIL import Image
im = Image.open('singlelayer_test_MdkB_E4_KI_15hpf-02-2.tif')
#im.show()
imarray = np.array(im)
viewer = napari.view_image(imarray,
                        colormap='green',
                       # depiction='volume',
                       # order=(0,6,1,2,3,4,5),
                       # ndisplay=3
                           )
#slice的格式（也就是start:end的形式，注意start是等于，end是小于没有等于）
#e[:, 1:5]
'''
viewer = napari.view_image(cells for i in 1024 [:，30，:，30，: ，: ，:]:
                                        for i in range(1024):
                                            for

                        colormap='green',
                        depiction='plane',
                        order=(0,6,1,2,3,4,5),
                        ndisplay=2
                           )
'''
###3Dz
viewer = napari.view_image(cells,
                        colormap='green',
                        depiction='volume',
                        order=(0,6,1,2,3,4,5),
                        ndisplay=3
                           )
####2D
viewer = napari.view_image(cells,
                        colormap='green',
                        depiction='plane',
                        order=(0,6,1,2,3,4,5),
                        ndisplay=2
                           )
help(napari.view_image)

from skimage import data
cell = data.cells3d()[30, 1]  # grab some data
viewer = napari.view_image(cells, colormap='magma')
help(napari.view_image)

'''测试样例
# create a Viewer and add an image here
viewer = napari.view_image(my_image_data)

# custom code to add data here
viewer.add_points(my_points_data)

# start the event loop and show the viewer
napari.run()

from skimage import data
import napari

viewer = napari.view_image(data.cells3d(), channel_axis=1, ndisplay=3)
napari.run()  # start the "event loop" and show the viewer
'''
'''
import numpy as np
test = np.asarray(1)
'''

'''
feature value 
feature martix


'''

#########sklearn machine learning
import sklearn.decomposition as sk_decomposition
from sklearn import datasets

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_Y = loaded_data.target
pca = sk_decomposition.PCA(n_components=2,whiten=False,svd_solver='auto')

pca.fit(data_X)
#PCA_X为降维后的数据
PCA_X = pca.transform(data_X)
print ('各个主成分方差比例',pca.explained_variance_ratio_)
print ('各个主成分方差值',pca.explained_variance_)
print ('降维后的特征数量',pca.n_components_)


###############feature value

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:35:57 2019
@author: Melo琦
"""
import pandas as pd
import numpy as np

# Read data
Raw_data = pd.read_excel(r'C:\Users\201810\Desktop\机器学习数据集\1.xls', sheet_name='Sheet1', index_col=None)
Raw_data = Raw_data.values  # 取值，且转化为二维数组
data = np.array(Raw_data)  # 二维数组(矩阵) 这里的np.array()是为了下面使用numpy包，转化为nummpy数据标准 ,并不改变维度
(m, n) = Raw_data.shape  # 矩阵的行数m列数n，返回一个元组


def entropy(vector, segnment):  # 自定义一个求解信息熵的函数，vector为向量，segment分段数值
    x_min = np.min(vector)
    x_max = np.max(vector)
    x_dis = np.abs(x_max - x_min)
    x_lower = x_min
    seg = 1.0 / segnment
    ternal = x_dis * seg
    list1 = []
    List1 = []
    #
    for i in range(len(vector)):
        if vector[i] >= x_lower + ternal:
            list1.append(vector[i])
    len_list1 = len(list1)
    List1.append(len_list1)
    #
    for j in range(1, segnment):
        list1 = []
        for i in range(len(vector)):
            if vector[i] >= x_lower + j * ternal and vector[i] < x_lower + (j + 1) * ternal:
                list1.append(vector[i])
        len_list1 = len(list1)
        List1.append(len_list1)
    #
    list1 = []
    for i in range(len(vector)):
        if vector[i] >= x_lower + (segnment - 1) * ternal:
            list1.append(vector[i])
    len_list1 = len(list1)
    List1.append(len_list1)
    List1 = List1 / np.sum(List1)

    y = 0
    Y = []
    for i in range(segnment):
        if List1[i] == 0:
            y = 0
            Y.append(y)
        else:
            y = -List1[i] * np.log2(List1[i]);
            Y.append(y)
    result = np.sum(Y)
    return result


# 数据预处理
data_feature = np.zeros(shape=(6, n))  # 特征二维数组(矩阵)的初始化 np.zeros(shape=(行，列))
for i in range(n):
    data_ave = np.mean(data[:, i])
    data_std = np.std(data[:, i], ddof=1)
    for j in range(1, m - 1):  # 基于拉伊达准则的数据异常值处理
        if np.abs(data[j, i]) > 3 * data_std:
            data[j, i] = 0.5 * (data[j - 1][i] + data[j + 1][i])
        else:
            continue
    data_ave = np.mean(data[:, i])  # 均值
    data_std = np.std(data[:, i])  # 标准差
    data_max = np.max(data[:, i])  # 最大值
    data_min = np.min(data[:, i])  # 最小值
    data_energy = np.sum(np.abs(data[:, i]))  # 能量：数据绝对值之和表示能量
    data_normal = (data[:, i] - data_min) / (data_max - data_min)  # 数据归一化(0,1)
    segnment = int(0.5 * m);
    data_etropy = entropy(data_normal, segnment)  # 信息熵
    data_feature[:, i] = [data_ave, data_std, data_max, data_min, data_energy, data_etropy]  # 特征二维数组

# 写入数据
data_f = pd.DataFrame(data_feature)  # 写入数据
data_f.columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz']  # 列标题
data_f.index = ['ave', 'std', 'max', 'min', 'energy', 'entropy']  # 行标题
writer = pd.ExcelWriter(r'C:\Users\201810\Desktop\机器学习数据集\f1.xls')  # 写入路径
data_f.to_excel(writer, 'data_feature', float_format='%.2f')  # data_feature为sheet名，float_format 数值精度
writer.save()  # 保存

