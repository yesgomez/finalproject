# Load libraries
import sys
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Set random seed
np.random.seed(0)

# Number of features
number_of_features = 17

# Import features matrix and target vector

features, target = 
	
# X : array of shape [n_samples, n_features]
# y : array of shape [n_samples]


# Create function returning a compiled network
def create_network():
    
    # Start neural network
    network = models.Sequential()
    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))
    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu'))
    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))
    # Compile neural network
    network.compile(loss='binary_crossentropy', # Cross-entropy
        optimizer='rmsprop', # Root Mean Square Propagation
        metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return network

# Wrap Keras model so it can be used by scikit-learn
neural_network = KerasClassifier(build_fn=create_network, 
	epochs=10, 
	batch_size=100, 
	verbose=0)

# Evaluate neural network using three-fold cross-validation
print (cross_val_score(neural_network, features, target, cv=3))
