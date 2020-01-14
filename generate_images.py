import nn
import numpy as np
import sys
from util import *
from layers import *
import cv2

print("Reading CIFAR dataset")
_, _, XTest, YTest = readCIFAR10()

XTest = XTest[0:500,:,:,:]
YTest = YTest[0:500,:]

label_name={
	0 : "airplane",
	1 : "automobile",
	2 : "bird",
	3 : "cat",
	4 : "deer",
	5 : "dog",
	6 : "frog",
	7 : "horse",
	8 : "ship",
	9 : "truck",

}

for i in range(500):
	img = 255*XTest[i].transpose(1,2,0)
	label = np.where(YTest[i]==1)[0][0]
	cv2.imwrite('./images/' + label_name[label] + str(i) + '.png',img)

print("Images written")