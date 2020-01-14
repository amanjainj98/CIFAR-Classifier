import numpy as np
import gzip
import _pickle as cPickle

def oneHotEncodeY(Y, nb_classes):
	# Calculates one-hot encoding for a given list of labels
	# Input :- Y : An integer or a list of labels
	# Output :- Coreesponding one hot encoded vector or the list of one-hot encoded vectors
	return (np.eye(nb_classes)[Y]).astype(int)

def readCIFAR10():
	XTrain = np.zeros((50000,3,32,32));
	XTest = np.zeros((10000,3,32,32));
	
	YTrain = np.zeros((50000,1), dtype=np.int8);
	YTest = np.zeros((10000,1), dtype=np.int8);
	
	for i in range(1,6):
		with open('./datasets/cifar-10/data_batch_' + str(i), mode='rb') as f:
			data = cPickle.load(f, encoding='bytes')
			XTrain[(i-1)*10000:i*10000, :,:,:] = np.asarray(data[b'data']).reshape(10000,3,32,32)
			YTrain[(i-1)*10000:i*10000,0] = data[b'labels']
	
	with open('./datasets/cifar-10/test_batch', mode='rb') as f:
		data = cPickle.load(f, encoding='bytes')
		XTest[:,:,:,:] = np.asarray(data[b'data']).reshape(10000,3,32,32)
		YTest[:,0] = data[b'labels']# print(train_set)
	
	YTrain = np.array(oneHotEncodeY(YTrain, 10))
	YTest = np.array(oneHotEncodeY(YTest, 10))
	
	XTrain /=255
	XTest /=255
	

	YTrain = np.reshape(YTrain, (50000,10))
	YTest = np.reshape(YTest, (10000,10))
	
	return XTrain, YTrain, XTest, YTest


