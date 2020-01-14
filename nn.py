import numpy as np
import random
from util import oneHotEncodeY
import itertools

class NeuralNetwork:

	def __init__(self, out_nodes, alpha, batchSize, epochs):
		
		self.alpha = alpha
		self.batchSize = batchSize
		self.epochs = epochs
		self.layers = []
		self.out_nodes = out_nodes

	def addLayer(self, layer):
		# Method to add layers to the Neural Network
		self.layers.append(layer)

	def train(self, trainX, trainY, printTrainStats=True, saveModel=False, modelName=None):
		for epoch in range(self.epochs):
			if printTrainStats:
				print("Epoch: ", epoch)

			X = np.asarray(trainX)
			Y = np.asarray(trainY)
			perm = np.arange(X.shape[0])
			np.random.shuffle(perm)
			X = X[perm]
			Y = Y[perm]

			trainLoss = 0

			numBatches = int(np.ceil(float(X.shape[0]) / self.batchSize))
			for batchNum in range(numBatches):
				XBatch = np.asarray(X[batchNum*self.batchSize: (batchNum+1)*self.batchSize])
				YBatch = np.asarray(Y[batchNum*self.batchSize: (batchNum+1)*self.batchSize])

				activations = self.feedforward(XBatch)	

				loss = self.computeLoss(YBatch, activations)
				trainLoss += loss
				
				predLabels = oneHotEncodeY(np.argmax(activations[-1], axis=1), self.out_nodes)

				acc = self.computeAccuracy(YBatch, predLabels)
				trainAcc += acc
				self.backpropagate(activations, YBatch)

			trainAcc /= numBatches
			if printTrainStats:
				print("Epoch ", epoch, " Training Loss=", loss, " Training Accuracy=", trainAcc)
			
			if saveModel:
				model = []
				for l in self.layers:
					if type(l).__name__ != "FlattenLayer": 
						model.append(l.weights) 
						model.append(l.biases)
				np.save(modelName, model)
				print("Model Saved... ")



	def computeLoss(self, Y, predictions):
		loss = (Y - predictions[-1]) ** 2
		loss = np.mean(loss)
		return loss

	def computeAccuracy(self, Y, predLabels):
		correct = 0
		for i in range(len(Y)):
			if np.array_equal(Y[i], predLabels[i]):
				correct += 1
		accuracy = (float(correct) / len(Y)) * 100
		return accuracy

	def test(self, testX, testY):
		testActivations = self.feedforward(testX)
		pred = np.argmax(testActivations[-1], axis=1)
		testPred = oneHotEncodeY(pred, self.out_nodes)
		testAcc = self.computeAccuracy(testY, testPred)
		return testAcc

	def feedforward(self, X):
		activations = [X]
		for l in self.layers:
			activations.append(l.forwardpass(activations[-1]))
		return activations


	def backpropagate(self, activations, Y):
		delta = activations[-1] - Y
		for i in range(len(self.layers)-1, -1, -1):
			delta = self.layers[i].backwardpass(self.alpha, activations[i], delta)
