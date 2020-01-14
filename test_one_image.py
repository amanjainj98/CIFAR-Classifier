import nn
import numpy as np
import sys
from util import *
from layers import *
import cv2

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

modelName = sys.argv[1]
image = sys.argv[2]

nn1 = nn.NeuralNetwork(10, 0.2, 20, 12)
nn1.addLayer(ConvolutionLayer([3,32,32], [8,8], 4, 3))
nn1.addLayer(FlattenLayer())
nn1.addLayer(FullyConnectedLayer(4*9*9,20))
nn1.addLayer(FullyConnectedLayer(20,10))
print("Model created")


model = np.load(modelName)
k,i = 0,0
for l in nn1.layers:
	if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
		nn1.layers[i].weights = model[k]
		nn1.layers[i].biases = model[k+1]
		k+=2
	i+=1
print("Model Loaded... ")


print("Testing the model on given image")

img = np.array(cv2.imread(image))
img = img.transpose(2,0,1)
testActivations = nn1.feedforward(np.array([img]))
pred = np.argmax(testActivations[-1], axis=1)[0]


print("Prediction : ",label_name[pred])








