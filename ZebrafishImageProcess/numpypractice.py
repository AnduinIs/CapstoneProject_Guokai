import numpy as np
from PIL import Image

import czifile as czi
image2 = czi.imread('mb2_MDKE4KI.czi')

#
none,timelapse,channel,zlayer,high,wide,RGB = image2.shape[:7]
newnu = np.zeros([none,timelapse,channel,zlayer,high,wide,RGB],dtype= imarray.dtype)

y = np.nditer(image2)
z = np.reshape(y,[2,132,1024,1024])

for i in range(0:timelapse):
    img_result[i,zlayer,high,wide] = imarray[i,timelapse,channel,zlayer,high,wide,RGB]
    for j in range(0:channel):
        for ii in range(0:zlayer):
            for yy in range(0:high):
                for xx in range(0:wide):
                    img_result[i,zlayer,high,wide] = imarray[i,timelapse,channel,zlayer,high,xx,RGB]
            img_result[i,zlayer,high,wide] = imarray[i,timelapse,channel,zlayer,yy,wide,RGB]
