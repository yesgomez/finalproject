import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

''' Task 1. Implement a feed-forward, three-layer, neural network with standard sigmoidal units. Allow for variation in the size of input layer, hidden layer, and output layer. Need to support cross-validation. '''

## Learn 8x3x8 encoder problem ##

bin_x = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
bin_y = np.array([1,2,3,4,5,6,7,8])
print ("Shape:", bin_x.shape, bin_y.shape)
X_train, X_test, y_train, y_test = train_test_split(bin_x, bin_y, test_size=0.4, random_state=0)
print ("Data sets:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test) 
print (clf.score(X_test, y_test))

plt.figure(figsize=(8,8))
plt.subplot(321)
plt.title("Sample Plot", fontsize='small')
plt.scatter(bin_x[:, 0], bin_x[:, 1], marker='o', c=bin_y, s=25, edgecolor='k')
plt.show()

## Some sample stuff ## 
# plt.figure(figsize=(8,8))
# plt.subplot(321)
# plt.title("Sample Plot", fontsize='small')

# X1, Y1 = datasets.make_classification(n_samples=200, n_features=2, n_classes=2, n_clusters_per_class=2, n_redundant=0, n_informative=2, weights=None)
# print ("Shape:", X1.shape, Y1.shape)

# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

# X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.4, random_state=0)
# print ("Data sets:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# scores = cross_val_score(clf, X1, Y1, cv=5)
# clf.score(X_test, y_test) 
# print (clf.score(X_test, y_test), "Scores:", scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# predicted = cross_val_predict(clf, X1, Y1, cv=10)
# print( metrics.accuracy_score(Y1, predicted) )
# # plt.show()