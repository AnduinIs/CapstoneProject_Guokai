import czifile as czi
image2 = czi.imread('mb2_MDKE4KI.czi')

import napari
import numpy as np
import cv2
from PIL import Image
imarray = np.array(image2)
none,timelapse,channel,zlayer,high,wide,RGB = imarray.shape[:7]
#针对于灰度值，不可以用于RGB
blank_np = np.zeros([zlayer, high, wide], dtype=imarray.dtype)
new_np = blank_np #传参用矩阵
#注意：这里需要指出timeplapse的参数for 3D分析，如果4D分析的话，运行注释中的代码
#Attention：indicate the parameter of timelapse for 3D-analysis, if 4D applicable, run the code anotated

for z in range(zlayer):
    for y in range(high):
        for x in range(wide):
            new_np[z,y,x] = imarray[none-1,timelapse-1,0,z,y,x,RGB-1] #这个地方的0代表第一个信号通道


z_stack = blank_np#接收输出用矩阵

for i in range(zlayer):
    img = new_np[i,:,:]
    #prewitt算子-卷积核
    prewitt_x = np.array([[-1,0,-1],[-1,0,-1],[-1,0,-1]],dtype=np.float32)
    prewitt_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype = np.float32)
    #卷积操作
    prewitt_grad_x = cv2.filter2D(img,cv2.CV_32F,prewitt_x)
    prewitt_grad_x = cv2.convertScaleAbs(prewitt_grad_x)
    from imutils import contours
    from skimage import measure

    gray = prewitt_grad_x
    #滤波
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    #cv2.imshow('blurred',blurred)
    #cv2.waitKey(0)
    thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
    z_stack[i,:,:] = thresh
cv2.imshow('img',thresh)
cv2.waitKey(0)

#查找荧光点
plot_positive = blank_np
for i in range(zlayer):
    for y in range(high):
        for x in range(wide):
            if z_stack[i,y,x]>1:
                plot_positive[i,y,x] = z_stack[i,y,x]

from array import array
x_fluro = array('i', []) #创建空列表
y_fluro = array('i', [])
z_fluro = array('i', [])
#储存的是荧光点的信息，沿着z轴逐层储存荧光点坐标
for i in range(zlayer):
    for y in range(high):
        for x in range(wide):
            if z_stack[i,y,x]>1:
                #arrays的第一个元素是一个1x3的数组，分别对应zyx
                np.append(x_arrays,x)
                np.append(y_arrays,y)
                np.append(z_arrays,i)



# Import libraries
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Creating figure
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
ax.scatter3D(x_fluro, y_fluro, z_fluro, color="green")
plt.title("simple 3D scatter plot")

# show plot
plt.show()

verts, faces, _, _ = measure.marching_cubes(z_stack, 120)
#使用matplotlib 库显示
import vtk
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces], alpha=0.70)
face_color = [0.45, 0.45, 0.75]
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)

ax.set_xlim(0, img3d.shape[0])
ax.set_ylim(0, img3d.shape[1])
ax.set_zlim(0, img3d.shape[2])

plt.show()

czi.
czi.czi2tif(dsa)
