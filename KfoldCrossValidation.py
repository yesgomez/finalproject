import os
import sys
import numpy as np
import binascii
from Bio import SeqIO
from itertools import chain
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

''' Given two matrices comprising positive and negative data,
	this script splits each into K folds, using all folds but
	one as a training set, the remaining fold as a test set, 
	and then calculates testing accuracy. '''

dictionary = {'A':2, 'C':-2, 'T':3, 'G':-3, 'Y':1, 'N':-1}
revdict = {2:'A', -2:'C', 3:'T', -3:'G', 1:'Y', -1:'N'}

def line2bits(line):
    # return [bin(ord(x))[2:].zfill(8) for x in line]
    newline = []
    for x in line:
    	newx = dictionary[x]
    	newline.append(newx)
    return newline
# def line2bits(line):
#     return [bin(ord(x))[2:].zfill(8) for x in line]
# print(line2bits("ATCG"))

class crossValidation(object):

	# Reading in text files
	def read(self, filename, site):
		name = filename.split('fn')[0]
		with open(filename) as f:
			name = f.readlines()
		
		for x, n in enumerate(name):
			n = n.split()[0]
			n = n + site
			name[x] = n
		print(name[0])
		return name

	# Combine the current pos and neg folds into one dataset and translate letters to binary
	def translate(self, dataset):
		print("Translating dataset")
		templist = []
		newdataset = []

		for line in dataset:
			line = str(line.split()[0])
			bits = line2bits(line)
			# print (bits)
			newdataset.append(bits)
		print (len(newdataset))
		return newdataset

	def prep_data(self, testFold):
		indices = list(range(K))
		indices.remove(testFold)
		
		# Add testFold to testdata set
		testdata = np.append(pfolds[testFold], nfolds[testFold])
		traindata = []
		
		# Add all other folds to traindata set
		for x in indices:
			traindata.append(np.append(pfolds[x], nfolds[x]))
			# print (x)
		traindata = list(chain(*traindata)) # flatten to single list
		print ("Test examples",len(testdata),"\nTrain examples",len(traindata), "\n", traindata[0], testdata[0])

		# Turn AA into binary, with last column for binding classification
		train = self.translate(traindata)
		test = self.translate(testdata)
		return train, test

	# Run the NN using the assigned train folds
	def NN_run(self, train, test):
		print ("Running the NN.")
		# Import data as dataframes 
		columns = list(range(17))
		training = np.array(train)
		testing = np.array(test)

		t = training.T[0:-1]
		e = testing.T[0:-1]
		dt = pd.DataFrame(t.T, columns=columns) # load the dataset as a pandas data frame
		de = pd.DataFrame(e.T)
		yt = training.T[-1] # define the target variable (dependent variable) as y
		ye = testing.T[-1]

		# fit a model
		lm = linear_model.LinearRegression()
		model = lm.fit(dt, yt)
		predictions = lm.predict(de)

		print(predictions[0:5], "\nScore:", model.score(de, ye))

	# Calculate testing accuracy
	def calc_accuracy():
		print("Calculating R^2.")

cV = crossValidation()

# Declaring global stuff
K = int(sys.argv[1])
positivefn = sys.argv[2]
positive = []
negativefn = sys.argv[3]
negative = []
i = 0

# Split both datasets into K equal partitions (or "folds")
positive = cV.read(positivefn, 'Y')
negative = cV.read(negativefn, 'N')
pos = np.array(positive)
neg = np.array(negative)

l = len(pos)
fraction = np.round(int(l) / int(K))
pfolds = []
nfolds = []

for j in range(K):
	i = int(j*int(fraction))
	h = int(i+int(fraction))
	print(j,i,h)
	pfolds.append(pos[i:h])
	nfolds.append(neg[i:h])
if len(pfolds)==len(nfolds):
	print("Number of folds:",len(pfolds))

for j in range(K):
	print ("\nRun", j)
	train, test = cV.prep_data(j)
	cV.NN_run(train, test)

# Use the average testing accuracy as the estimate of out-of-sample accuracy
# avgAcccuracy = np.average(*)