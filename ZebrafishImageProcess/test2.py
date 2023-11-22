#python file equals to the script in R

print('hello world')
print('nihaobuhao\tdashdsa')
print('hello\rdsah')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin) * np.random.rand(n) + vmin


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

import czifile as czi

image = czi.imread('test.czi')

czi.imread()
with open('F:\\AAAconfocal data\\MKDB_E4__KI_15hpf_timlapse\\MdkB_E4_KI_15hpf-02.czi') \
        as mkdb_14_timelapse:
        image3 = czi.imread('MdkB_E4_KI_15hpf-02.czi')
#image2 = czi.imread('F:\\AAAconfocal data\\MKDB_E4__KI_15hpf_timlapse\\MdkB_E4_KI_15hpf-02.czi\\')

open('E:\python\python\notfound.txt', 'r')

image2 = czi.imread('MdkB_E4_KI_15hpf-02.czi')  #导入czi文件，测试样例文件大小1，121，2，60，1024，1024，1
                                                #这是一个七维的numpy
                                                #1不知道是什么
                                                #121指的是时间尺度
                                                #2是双通道
                                                #60指的是60层扫描
                                                #1024*1024指的是每张图的2x2
                                                #1也不知道是什么

#Q1：为什么这里会有两个1作为没有意义的维度？
#是否每个czi都有这样的东西，如果是的话，怎么统一不同的czi文件格式

###############文件已经导入，下面开始进行寻找特征点###############

import Offset_Subtraction as subtract




