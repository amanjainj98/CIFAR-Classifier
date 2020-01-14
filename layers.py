import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		self.data = None
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))


	def forwardpass(self, X):
		n = X.shape[0]  # batch size
		
		self.data = np.matmul(X,self.weights) + self.biases
		return sigmoid(self.data)

		
	def backwardpass(self, lr, activation_prev, delta):
		n = activation_prev.shape[0] # batch size

		ds = derivative_sigmoid(self.data)

		new_delta =  np.matmul(np.multiply(ds,delta),np.transpose(self.weights))

		self.weights -= lr*np.matmul(np.transpose(activation_prev),np.multiply(ds,delta))
		self.biases -= lr*np.multiply(ds,delta).sum(axis=0)

		
		return new_delta


class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		n = X.shape[0]  # batch size


		output = np.zeros((n,self.out_depth,self.out_row,self.out_col))
		data = np.zeros((n,self.out_depth,self.out_row,self.out_col))

		for i in range(self.out_row):
			for j in range(self.out_col):
				for d in range(self.out_depth):
					for b in range(n):

						x1 = i*self.stride
						x2 = i*self.stride+self.filter_row
						y1 = j*self.stride
						y2 = j*self.stride+self.filter_col

						data[b,d,i,j] = np.sum(np.multiply(self.weights[d],X[b,:,x1:x2,y1:y2])) + self.biases[d]
						output[b,d,i,j] = sigmoid(data[b,d,i,j])

		self.data = data
		return output


	def backwardpass(self, lr, activation_prev, delta):
		n = activation_prev.shape[0] # batch size

		new_delta = np.zeros((n , self.in_depth , self.in_row , self.in_col))
		ds = derivative_sigmoid(self.data)
		
		updated_weights = self.weights
		updated_biases = self.biases

		
		for i in range(self.out_row):
			for j in range(self.out_col):
				for d in range(self.out_depth):
					for b in range(n):
					
						x1 = i*self.stride
						x2 = i*self.stride+self.filter_row
						y1 = j*self.stride
						y2 = j*self.stride+self.filter_col

						new_delta[b,:,x1:x2,y1:y2] += ds[b,d,i,j] * delta[b,d,i,j] * self.weights[d]

						updated_weights[d] -= lr * ds[b,d,i,j] * delta[b,d,i,j] * activation_prev[b,:,x1:x2,y1:y2]
						updated_biases[d] -= lr * ds[b,d,i,j] * delta[b,d,i,j]

		

		self.weights = updated_weights
		self.biases = updated_biases
		
		return new_delta
	

class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
