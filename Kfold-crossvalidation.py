import os
import sys
from Bio import SeqIO 

''' Given two matrices comprising positive and negative data,
	this script splits each into K folds, using all folds but
	one as a training set, the remaining fold as a test set, 
	and then calculates testing accuracy. '''

# Declaring global stuff
K = sys.argv[1]
positivefn = sys.argv[2]
positives = []
negativefn = sys.argv[3]
negatives = []

with open(positivefn) as f:
	positives = f.readlines()
	for p in positives:
		p = p.split()[0]
print(positives[0])

with open(negativefn) as f:
	negatives = f.readlines()
	for n in negatives:
		n = n.split()[0]
print(negatives[0])

# Split the dataset into K equal partitions (or "folds")
# def split_data()

# def test_run(Ntestfold, ):
	# Use fold 1 as the testing set and the union of the other folds as the training set
	# Testing set = 30 observations (fold 1)
	# Training set = 120 observations (folds 2-5)

# Calculate testing accuracy
# def calc_accuracy():

# for i in range(K):
# 	test_run()

# Use the average testing accuracy as the estimate of out-of-sample accuracy