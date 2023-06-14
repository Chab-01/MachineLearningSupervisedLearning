import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv("bm.csv", header=None).values

x = data[:,:-1]
y = data[:,-1]
dataLength = len(data) 
trainingSetSize = round(int(0.9*dataLength)) #90% of the data
testSetSize = dataLength - trainingSetSize #Remaining 10%  
index = np.arange(dataLength) #We arrange the indices for the data
np.random.shuffle(index) #We shuffle the indices randomly

xTrain = x[index[:trainingSetSize]] 
yTrain = y[index[:trainingSetSize]] 
xTest = x[index[:testSetSize]] 
yTest = y[index[:testSetSize]] 

numberOfTrees = 100
bootstrapSize = 5000
trees = []

for i in range(numberOfTrees):
    
    sample_indices = np.random.choice(len(xTrain), size=bootstrapSize, replace=True) #Create a bootstrap sample of the training data
    xSamples = xTrain[sample_indices]
    ySamples = yTrain[sample_indices]
    
   
    tree = DecisionTreeClassifier() #Train a decision tree on the bootstrap sample
    tree.fit(xSamples, ySamples)
    trees.append(tree)

def predict(xTest): #Make predictions with each decision tree  
    predictions = []
    for tree in trees:
        predictions.append(tree.predict(xTest))
    combinedPred = np.round(np.mean(predictions, axis=0)) #Combine the predictions using a majority vote
    return combinedPred

accuracies = []
for tree in trees: #Iterate through the trees and make predictions for each and then calculate the accuracies
    yPred = tree.predict(xTest)
    acc = accuracy_score(yTest, yPred)
    accuracies.append(acc)
    
averageError = 1 - (np.mean(accuracies))
yPred = predict(xTest)
accuracy = accuracy_score(yTest, yPred)
generalizationError = 1 - accuracy
print(f"Estimate of generalization: {generalizationError:.4f}")
print(f"Average generalization error of individual trees: {averageError:.4f}")

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

#Generate predictions for each decision tree and plot their decision boundaries
plt.figure(1)
for i, tree in enumerate(trees):
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(10, 10, 1+i)
    plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))

#Generate predictions for the ensemble model and plot its decision boundary
plt.figure(2)
Z = predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
plt.title('Ensemble')

plt.show()