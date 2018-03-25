## Script adapted from web (https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6)
import sys
import pandas as pd
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold # import KFold

dictionary = {'A':2, 'C':-2, 'T':3, 'G':-3, 'Y':1, 'N':-1}
revdict = {2:'A', -2:'C', 3:'T', -3:'G', 1:'Y', -1:'N'}

def line2bits(line):
    # return [bin(ord(x))[2:].zfill(8) for x in line]
    newline = []
    for x in line:
    	newx = dictionary[x]
    	newline.append(newx)
    return newline

def translate(dataset):
		templist = []
		newdataset = []
		for line in dataset:
			line = str(line.split()[0])
			bits = line2bits(line)
			newdataset.append(bits)
		print (len(newdataset), newdataset[0])
		return newdataset

def NN(file1, file2):
	# Declaring variables
	posdata = []
	negdata = []

	# Turn positive file into usable data 
	with open(file1) as f:
		posdata = f.readlines()
		for x, n in enumerate(posdata):
			n = n.split()[0]
			n = n + 'Y'
			posdata[x] = n
		posdata = np.array(posdata)
		print(len(posdata), posdata[0])

	# Turn negative file into usable data 
	with open(file2) as f:
		negdata = f.readlines()
		for x, n in enumerate(negdata):
			n = n.split()[0]
			n = n + 'N'
			negdata[x] = n
		# There is a lot more negative data, so choose a random, equal size number of samples
		i = np.random.randint(0,3000)
		j = i + len(posdata)
		negdata = np.array(negdata[i:j])
		print(len(negdata), negdata[0])

	data = np.array(translate(np.append(posdata, negdata)))
	newd = data.T[0:-1]
	
	# Import data as dataframes 
	columns = list(range(17))
	df = pd.DataFrame(newd.T, columns=columns) # load the dataset as a pandas data frame
	y = data.T[-1] # define the target variable (dependent variable) as y

	# create training and testing vars
	X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
	print (X_train.shape, y_train.shape)
	print (X_test.shape, y_test.shape)

	# fit a model
	lm = linear_model.LinearRegression()
	model = lm.fit(X_train, y_train)
	predictions = lm.predict(X_test)

	# for p in predictions:
	# 	binpred = []
	# 	newp = p.round()
	# 	# print(p, newp)
	# 	binpred.append(newp)

	print(predictions, y_test)
	return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = NN(sys.argv[1], sys.argv[2])
print("\nScore:", model.score(X_test, y_test))

kf = KFold(n_splits=3, random_state=None, shuffle=False) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf) 

# for train_index, test_index in kf.split(X):
# 	print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = X[train_index], X[test_index]
# 	y_train, y_test = y[train_index], y[test_index]