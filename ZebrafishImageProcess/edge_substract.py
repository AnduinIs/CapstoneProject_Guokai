#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：lucky
# albert time:2020/9/29
#importing the required libraries
import numpy as np
import cv2 as cv



#reading the image
#img = cv.imread('puppy.jpg')
# cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
# cv.imshow('input',img)
img = imarray
# #将图片转化为灰度图
# imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#prewitt算子-卷积核
prewitt_x = np.array([[-1,0,-1],[-1,0,-1],[-1,0,-1]],dtype=np.float32)
prewitt_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype = np.float32)
#sobel算子
sobel_x = np.array([[-1,0,-1],[-2,0,-2],[-1,0,-1]],dtype=np.float32)
sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32)
'''原算子(确实原算子会把信号抹去，为什么？Q2）
prewitt_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=np.float32)
prewitt_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype = np.float32)
'''

'''偷懒
prewitt_x = sobel_x 
prewitt_y = sobel_y
'''

#卷积操作
prewitt_grad_x = cv.filter2D(img,cv.CV_32F,prewitt_x)
prewitt_grad_y = cv.filter2D(img,cv.CV_32F,prewitt_y)
prewitt_grad_x = cv.convertScaleAbs(prewitt_grad_x)
prewitt_grad_y = cv.convertScaleAbs(prewitt_grad_y)

#展示图片
# cv.imshow("prewitt x", prewitt_grad_x);
# cv.imshow("prewitt y", prewitt_grad_y);
# print(img.shape)
'''
#将三张图像合并到一张图像上
h,w = img.shape[:2]
img_result = np.zeros([h,3*w,3],dtype= img.dtype)
img_result[0:h,0:w,:] = img
img_result[0:h,w:2*w,:] = prewitt_grad_x
img_result[0:h,2*w:3*w,:] = prewitt_grad_y
'''
h,w = img.shape[:2]
img_result = np.zeros([h,w],dtype= img.dtype)
img_result = prewitt_grad_x - prewitt_grad_y


#图片展示
cv.imshow('result', img_result)
cv.imshow("prewitt x", prewitt_grad_x)
cv.imshow("prewitt y", prewitt_grad_y)
cv.waitKey()
#保存
cv.imwrite("prewitt.png",img_result)
cv.imwrite("D:\ZebrafishImageProcess\sobel.png",prewitt_grad_x)
#添加延迟
cv.waitKey(0)
#关闭所有窗口创建byCV2
cv.destroyAllWindows()
