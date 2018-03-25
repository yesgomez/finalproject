import os
import sys
import numpy as np
import pandas as pd
# import binascii
from Bio import SeqIO
from itertools import chain
from matplotlib import pyplot as plt
from scipy import interp
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

''' Given two matrices comprising positive and negative data,
	this script splits each into K folds, using all folds but
	one as a training set, the remaining fold as a test set, 
	and then calculates testing accuracy. '''

dictionary = {'A':2, 'C':-2, 'T':3, 'G':-3, 'Y':1, 'N':0}
revdict = {2:'A', -2:'C', 3:'T', -3:'G', 1:'Y', 0:'N'}

def line2bits(line):
	# return [bin(ord(x))[2:].zfill(8) for x in line]
	newline = []
	for x in line:
		newx = dictionary[x]
		newline.append(newx)
	return newline
#   return [bin(ord(x))[2:].zfill(8) for x in line]
# print(line2bits("ATCG"))

def R_sq(a, b):
	hat = np.sum(a) / len(a)
	top = []
	bottom = []
	for i in range(len(a)):
		top.append((a[i] - b[i])**2)
		bottom.append((a[i] - hat)**2)
	R = 1 - (np.sum(top) / np.sum(bottom))
	return R

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

		# Encode AA, with last column for binding classification
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

		## No scaling was done since my data does not have units

		# Use one hidden layer with logistic activation function, and stochastic gradient descent. 
		mlp = MLPClassifier(hidden_layer_sizes=(16,), activation='logistic', solver='sgd', max_iter=2000, alpha=0.001, learning_rate='adaptive')

		# Fit a model
		model = mlp.fit(dt,yt)
		predictions = mlp.predict(de)
		rawpredicts = mlp.predict_proba(de)

		print("Score:", model.score(de, ye))
		print(metrics.classification_report(ye,predictions))
		return mlp, model, predictions, rawpredicts, model.score(de, ye), ye

	# Calculate testing accuracy
	def calc_accuracy(self, predictions, rawpredicts, ye):
		print("Calculating R squared error.")
		score = R_sq(ye,predictions)
		print(score)
		return score

cV = crossValidation()

# Declaring global stuff
K = int(sys.argv[1])
positivefn = sys.argv[2]
positive = []
negativefn = sys.argv[3]
negative = []
i = 0
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Split both datasets into K equal partitions (or "folds")
positive = cV.read(positivefn, 'Y')
negative = cV.read(negativefn, 'N')
pos = np.array(positive)
neg = np.array(negative)
combinedData = np.append(pos,neg)
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

# Before cross-validating, try a variety of parameters to find best mlp classifier 
print ("\n--- Finding best classifier parameters using test data --- ")
# alpha = [0.0001, 0.001, 0.01, 1, 10, 100] # , 0.01, 1, 
# hls = list(range(2,17))
# best = []
# ix = np.round(0.8 * len(combinedData))
# fulltrain = []
# fullltest = []
# for a in alpha:
# for h in hls:
# 	train, test = cV.prep_data(0)
# 	predictions, score, ye = cV.NN_run(train, test, h)
# 	best.append(score)
# best = np.array(best) # This array lets you visually compare the scores from each param combination
# np.set_printoptions(precision=3)
# # best.resize((len(alpha),len(hls)))
# print(np.array(best), best.max())

# Running the NN class functions for each fold combination and output scores/accuracy
print ("\n--- Running K fold cross validation using best settings ---")
netscores = []
neterrors= []

def run_Kfold():
	i = 0
	for j in range(K):
		print ("\nRun", j)
		train, test = cV.prep_data(j)
		mlp, model, predict, rawpredict, score, ye = cV.NN_run(train, test)
		netscores.append(score)
		error = cV.calc_accuracy(predict, rawpredict, ye)
		neterrors.append(error)
		# Setting up metrics for plotting ROC curves below
		fpr, tpr, thresholds = metrics.roc_curve(ye, predict)
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = metrics.auc(fpr, tpr)
		aucs.append(roc_auc)
		plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

		i += 1
	return model, mlp

Kfoldmodel, Kmlp = run_Kfold()

# Plotting ROC curve for different folds 
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
# plt.show()

# Use the average testing accuracy as the estimate of out-of-sample accuracy
avgScore = np.mean(netscores)
avgError= np.mean(neterrors)
print("\nThe average score for %s fold cross-validation is %s according to Scikit-learn and the accuracy is %s according to my own R^2 function." %(K, avgScore, avgError))


# Part 5. Testing on unlabelled data
ULdata = cV.read("rap1-lieb-test.txt",'')
ul = cV.translate(ULdata)
ul = pd.DataFrame(ul)
ULpredictions = Kmlp.predict(ul)
ULrawpredicts = Kmlp.predict_proba(ul)
rotate = ULrawpredicts.T[1]
print(rotate[0:20])

with open("predictions.txt",'w') as w:
	w.write ("Sequence        \tProbability\n")
	for x, y in enumerate(ULdata):
		w.write("%s\t%s\n" %(y, rotate[x]))
