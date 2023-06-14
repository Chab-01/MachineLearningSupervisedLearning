import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("mnist_train.csv", header=0).values
x = data[:,1:]
y = data[:,0]

#Select 10000 random data points for train and 1000 for validation set
xTrain, xVal, yTrain, yVal = train_test_split(x, y, train_size=10000, test_size=1000)

paramGrid = {"C": [1, 5, 10], "gamma": [0.000001, 0.00001, 0.001]} #Values we will iterate and test

#Train the model and find best parameters. We get best hyperparameters:  {'C': 5, 'gamma': 1e-06}
#svm = SVC(kernel="rbf")
#gridSearch = GridSearchCV(svm, paramGrid)
#gridSearch.fit(xVal, yVal)
#print("Best hyperparameters: ", gridSearch.best_params_)
#print("Validation score: ", gridSearch.best_score_)

clf = SVC(kernel="rbf", gamma="scale", C=5)
clf.fit(xTrain, yTrain)

#Predict and get accuracy
yPred = clf.predict(xVal)
accuracy = (accuracy_score(yVal, yPred)) * 100 #Times 100 to get in percentage
print(f"Accuracy OVO: {accuracy:.2f}%")

def OVA(X, y, C):
    models = {}
    classes = np.unique(y)
    for i in classes:       
        binaryY = np.where(y == i, 1, 0) #Create binary labels 1 if i, 0 otherwise
        model = SVC(kernel='rbf', C=C, gamma="scale", probability=True) #train SVM model
        model.fit(X, binaryY)
        models[i] = model
    return models

models = OVA(xTrain, yTrain, C=5)
probabilities = np.zeros((len(xVal), len(models))) #Creates array of zeros with size given as arguments

for i, model in models.items(): #Iterate through the models with i and model that takes on values
    probabilities[:, i] = model.predict_proba(xVal)[:, 1] #Gets the probabilities fro each sample
yPredOVA = np.argmax(probabilities, axis=1) #choose the class with the highest probability for each sample

# compute accuracy
accuracyOVA = (accuracy_score(yVal, yPredOVA)) * 100
print(f"Accuracy OVA: {accuracyOVA:.2f}%")
