from scripts.8x3x8_encoder import *
import numpy as np

def forward_prop_test():
	# Declaring the test inputs and outputs
	X = np.identity(8)
	y = np.identity(8)

	# Regularization parameter
	Lambda = 0.0003
	learningRate = 0.5
	testRun = NN.forward(X)
	# Visualize how close am I to the correct output
	testRunClearer = np.round(testRun, decimals=0) 
	print('F.P. Results:', testRunClearer, "\nForward propagation successful.")
	return

def cost_function_test():
	# Declaring the test inputs and outputs
	X = np.identity(8)
	y = np.identity(8)
	
	cost1 = costFunction(X,y)
	dJdw1, dJdw2 = costFunctionPrime(X,y)
	# When we account for derivatives (by subtracting them) the cost should go down
	NN.w1 = NN.w1 - 3*dJdw1
	NN.w2 = NN.w2 - 3*dJdw2
	cost3 = costFunction(X,y)
	print ('Shapes', cost1.shape, cost3.shape, "\Cost function calculation successful.")
	return

def gradient_comparison_test():
	# Declaring the test inputs and outputs
	X = np.identity(8)
	y = np.identity(8)
	
	newgrad = newGradient(X, y, learningRate)
	print(newgrad)
	numgrad = computeNumericalGradient(NN, X, y)
	grad = computeGradient(X,y)
	# This measures how similar they are (should be < 10^8)
	print (np.linalg.norm(grad-newgrad) / np.linalg.norm(grad+newgrad), "\nGradient comparison successful.")
	return

def full_train_test():
	# Using half the matrix for training and the other half for testing
	trainX = np.identity(8)[0:4]
	trainy = np.identity(8)[0:4]
	testX = np.identity(8)[4:8]
	testy = np.identity(8)[4:8]

	T = trainer(NN)
	T.train(trainX, trainy, testX, testy)

