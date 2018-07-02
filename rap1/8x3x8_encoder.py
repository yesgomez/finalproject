## Original code located in 8x3x8_encoder iPython notebook. 
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class Neural_network(object):
	''' Defining the neural network class. '''
	def __init__(self, i=int(sys.argv[1]), o=int(sys.argv[2]), h=int(sys.argv[3])):
		# Parameters that do not change while script is running (hyperparameters)
		self.inputLayerSize = i
		self.outputLayerSize = o
		self.hiddenLayerSize = h
		
		# Initializing weights
		self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

	def forward(self,x):
		# Forward propagation
		self.z2 = np.dot(x, self.w1) # inputs x synapse weights 1
		self.a2 = _sigmoid(np.array(self.z2)) # apply activation function

		self.z3 = np.dot(self.a2, self.w2) # output of layer x synapse weights 2
		y_hat = _sigmoid(self.z3) # apply activation function again
		return y_hat

def _sigmoid(z):
	# Applying a sigmoidal activation function
	return 1/(1 + np.exp(-z))

def sigmoidPrime(z):
	# Derivative of above sigmoid function
	return np.exp(-z)/((1 + np.exp(-z))**2)

def costFunction(X,y):
	# Using the sum of the squared error as cost with an added regularization factor 
	NN.y_hat = NN.forward(X)
	cost = 0.5*sum((y-NN.y_hat)**2)/X.shape[0] + (Lambda/2)*(np.sum(NN.w1**2) + sum(NN.w2**2)) 
	return cost

def costFunctionPrime(X, y):
	# Derivative with respect to w1 and w2
	NN.y_hat = NN.forward(X)

	delta3 = np.multiply(-(y - NN.y_hat), sigmoidPrime(NN.z3))
	print(delta3.shape, y.shape, sigmoidPrime(NN.z3).shape)
	dJdw2 = np.dot(NN.a2.T, delta3) + Lambda*NN.w2
	delta2 = np.multiply(delta3, NN.w2.T) * (sigmoidPrime(NN.z2))
	dJdw1 = np.dot(X.T, delta2) + Lambda*NN.w1
	
	return dJdw1, dJdw2

def getParams():
	# Get w1 and w2 as vectors
	params = np.concatenate((NN.w1.ravel(), NN.w2.ravel()))
	return params

def setParams(params):
	# Take w as vectors and set new w1 and w2
	w1_start = 0
	w1_end = NN.hiddenLayerSize * NN.inputLayerSize
	NN.w1 = np.reshape(params[w1_start:w1_end], (NN.inputLayerSize, NN.hiddenLayerSize))
	w2_end = w1_end + NN.hiddenLayerSize*NN.outputLayerSize
	NN.w2 = np.reshape(params[w1_end:w2_end], (NN.hiddenLayerSize, NN.outputLayerSize))

def computeGradient(X,y):
	# Computes a gradient (duh)
	dJdw1, dJdw2 = costFunctionPrime(X,y)
	return np.concatenate((dJdw1.ravel(), dJdw2.ravel()))
	
def newGradient(X, y, learningRate): 
	# Meant to replace the old gradient since it didn't work with the 3x8/8x8 arrays 
	theta = np.zeros((8,8))
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int(NN.inputLayerSize*NN.hiddenLayerSize + NN.hiddenLayerSize*NN.outputLayerSize )
	grad = np.zeros(parameters)
	error = _sigmoid(X * theta.T) - y

	grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
	# intercept gradient is not regularized
	grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
	print ("New grad",np.array(grad).ravel())
	return np.array(grad).ravel()

def computeNumericalGradient(N,X,y):
	''' Estimating the gradient numerically to compare values. '''
	paramsInitial = getParams()
	numgrad = np.zeros(paramsInitial.shape)
	perturb = np.zeros(paramsInitial.shape)
	e = 1e-4
	
	for p in range(len(paramsInitial)):
		# Set perturbation vector
		perturb[p] = e
		setParams(paramsInitial + perturb)
		loss2 = costFunction(X,y)
		setParams(paramsInitial - perturb)
		loss1 = costFunction(X,y)
		
		# Compute num grad
		print ((loss2 - loss1)/(2*e))
		numgrad[p] = (loss2 - loss1)/(2*e)
		
		#Return changed value to zero
		perturb[p] = 0
	
	# Return params to original values
	setParams(paramsInitial)

	return numgrad 


class trainer(object):
	''' Defining a training class. Modified to check testing error during training. '''
	def __init__(self, N):
		# Make a local reference to the neural network
		self.N = N    
		
	def callbackF(self, params):
		setParams(params)
		self.J.append(costFunction(self.X, self.y))
		self.testJ.append(costFunction(self.testX, self.testy))
	
	def costFunctionWrapper(self, params, X, y):
		# To track the cost function over training time
		setParams(params)
		cost = costFunction(X, y)
		# grad = computeGradient(X, y) #
		grad = newGradient(X, y, learningRate)
		print("Grad:", grad.shape)
		return cost, grad
	
	# This is the actual training function
	def train(self, trainX, trainy, testX, testy):
		# Make internal variable for callback
		self.X = trainX
		self.y = trainy
		self.testX = testX
		self.testy = testy
		
		# Make empty list to store costs
		self.J = []
		self.testJ = []
		params0 = getParams()
		print("Params:", params0.shape)
		
		options = {'maxiter': 200, 'disp': True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', args = (X,y), options = options, callback = self.callbackF)
		
		# Update params with new values from last iteration
		setParams(_res.x)
		self.optimizationResults = _res
		return


''' Execution with Training and Testing values. Using the first half of the 8x8 
	identity matrix as a training set and the second half as a testing set. '''

# Declaring the test inputs and outputs
NN = Neural_network()
X = np.identity(8)
y = np.identity(8)

# Regularization parameter
Lambda = 0.0003
learningRate = 0.5
testRun = NN.forward(X)

# Visualize how close am I to the correct output
testRunClearer = np.round(testRun, decimals=0) 
print('Test run clearer:', testRunClearer)

cost1 = costFunction(X,y)
#dJdw1, dJdw2 = costFunctionPrime(X,y)
# When we account for derivatives (by subtracting them) the cost should go down
#NN.w1 = NN.w1 - 3*dJdw1
#NN.w2 = NN.w2 - 3*dJdw2
cost3 = costFunction(X,y)

print ('Shapes', cost1.shape, cost3.shape)
# NOTE: The problem is I have 8 costs and the reason is that my initial activation  
# function calculation is multiplying an (8,8) by an (8,3) giving me 8 scalar costs. 

newgrad = newGradient(X, y, learningRate)
print(newgrad)
#numgrad = computeNumericalGradient(NN, X, y)
#grad = computeGradient(X,y)
# This measures how similar they are (should be < 10^8)
#np.linalg.norm(grad-newgrad) / np.linalg.norm(grad+newgrad)

# Using half the matrix for training and the other half for testing
trainX = np.identity(8)[0:4]
trainy = np.identity(8)[0:4]
testX = np.identity(8)[4:8]
testy = np.identity(8)[4:8]

T = trainer(NN)
T.train(trainX, trainy, testX, testy)

# plt.plot(T.J)
# plt.plot(T.testJ)
# plt.grid(1)
# plt.ylabel('Cost')
# plt.xlabel('Iterations')
# plt.legend(['Training', 'Testing'])
# plt.show()

# NN.forward(X)