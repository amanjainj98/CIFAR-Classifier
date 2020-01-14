import nn
import numpy as np
import sys
from util import *
from layers import *
import cv2

print("Reading CIFAR dataset")

XTrain, YTrain, XTest, YTest = readCIFAR10()

XTrain = XTrain[0:5000,:,:,:]
XTest = XTest[0:1000,:,:,:]
YTrain = YTrain[0:5000,:]
YTest = YTest[0:1000,:]

modelName = sys.argv[1]


nn1 = nn.NeuralNetwork(10, 0.2, 20, 12)
nn1.addLayer(ConvolutionLayer([3,32,32], [8,8], 4, 3))
nn1.addLayer(FlattenLayer())
nn1.addLayer(FullyConnectedLayer(4*9*9,20))
nn1.addLayer(FullyConnectedLayer(20,10))
print("Model created")


model = np.load(modelName)
k,i = 0,0
for l in nn1.layers:
	if type(l).__name__ != "FlattenLayer": 
		nn1.layers[i].weights = model[k]
		nn1.layers[i].biases = model[k+1]
		k+=2
	i+=1
print("Model Loaded... ")


print("Testing the model")

accuracy = nn1.test(XTest, YTest)

print('Test Accuracy : ',accuracy)
