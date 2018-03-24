import os
import sys
import numpy as np
from Bio import SeqIO 

''' Given two matrices comprising positive and negative data,
	this script splits each into K folds, using all folds but
	one as a training set, the remaining fold as a test set, 
	and then calculates testing accuracy. '''

# Declaring global stuff
K = int(sys.argv[1])
positivefn = sys.argv[2]
positive = []
negativefn = sys.argv[3]
negative = []
i = 0

# Reading in text files
def read(filename):
	name = filename.split('fn')[0]
	with open(filename) as f:
		name = f.readlines()
	for n in name:
		n = n.split()[0]
	print(name[0].split()[0])
	return name

positive = read(positivefn)
negative = read(negativefn)

# Split the dataset into K equal partitions (or "folds")
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
	# i = i+int(fraction)
print(len(pfolds), len(nfolds[-1]))

# Run the NN using the assigned train folds
def test_run(testFold):
	data = np.concatenate(pfolds[testFold],nfolds[testFold])
	print(data.shape)
test_run(0)
# Calculate testing accuracy
# def calc_accuracy():

# for i in range(K):
# 	test_run()

# Use the average testing accuracy as the estimate of out-of-sample accuracy