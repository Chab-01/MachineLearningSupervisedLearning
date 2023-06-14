import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt

#Load data
data = pd.read_csv("dist_val.csv", header=None, delimiter=";")
X = data.iloc[:, :-1] # Extract input features (all but the last column)
y = data.iloc[:, -1]  # Extract target variable (last column)

#Define hyperparameters grid for an SVM model using different kernels and parameters to be tuned
#The grid is a dictionary where each key corresponds to a kernel and the values are lists of dictionaries
#containing the hyperparameters to be tested for that kernel
paramGrid = {'linear': [ {'kernel': ['linear'], 'C': [1, 5, 10]} ],
              'rbf': [ {'kernel': ['rbf'], 'C': [1, 5, 10], 'gamma': [0.1, 1, 10]} ],
              'poly': [ {'kernel': ['poly'], 'C': [1, 5, 10], 'degree': [2, 3, 4]} ]}

#Perform grid search for each kernel and evaluate F1 score
bestScore = 0   #Keep track of the best F1 score found so far
bestParams = {}  #Keep track of the hyperparameters that gave the best F1 score

for kernel in paramGrid:
    print("Kernel: ", kernel)
    
    for params in ParameterGrid(paramGrid[kernel]): #For each kernel we try all combinations of hyperparameters
        #We train SVM model with the current hyperparameters
        clf = SVC(**params)
        clf.fit(X, y)
        yPred = clf.predict(X) #We use the trained model to make predictions   
        score = f1_score(y, yPred) #Compute the F1 score of the predictions
        print(f"Parameters: {params}")
        print(f"F1 score: {score}")
        
        if score > bestScore: #If the current models F1 score is better than the best score found so far we update the best score and parameters
            bestScore = score
            bestParams = params
            #Plot decision boundary for best model
            h = 0.1
            x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
            y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z)
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=np.where(y==1, 'g', 'b'))
            plt.title(f"Decision boundary for best {kernel} model with parameters: {params} Score: {bestScore}")
            plt.show()

print(f"Best F1 score: {bestScore}")
print(f"Best parameters: {bestParams}")
