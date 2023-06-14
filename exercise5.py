import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

data = pd.read_csv("fashion-mnist_train.csv", header=0).values
dataTest = pd.read_csv("fashion-mnist_test.csv", header=0).values
labels = data[:, 0]
x = data[:, 1:]
labelsTest = dataTest[:,0]
xTest = dataTest[:, 1:]

randIndicies = np.random.choice(len(labels), size=16, replace=False)  #Select 16 random indices

plt.figure(1)
for i in range(len(randIndicies)):
    plt.subplot(4, 4, 1+i)
    image = np.reshape(x[randIndicies[i]], (28, 28)) #Get the index of x and reshape to 28x28 pixel image data
    label = labels[randIndicies[i]] #Get corresponding label
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")


def MLP(hiddenLayers, units, regularization, learningRate):
    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_shape=(784,)))

    for _ in range(hiddenLayers):
        model.add(Dense(units=units, activation='relu'))

    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#Wrap the keras model using KerasClassifier
model = KerasClassifier(build_fn=MLP)

#Define the hyperparameter grid..... Best Hyperparameters:  {'hiddenLayers': 3, 'learningRate': 0.001, 'regularization': 0.01, 'units': 256}
                                      # Given the paramgrid 
#paramGrid = {
#    'hiddenLayers': [1, 2, 3],
#    'units': [64, 128, 256],
#    'regularization': [0.001, 0.01, 1],
#    'learningRate': [0.001, 0.01, 0.1]
#}

paramGrid = { #These are the best parameters i could find from the grid above, i run them solo here to reduce time.
    'hiddenLayers': [3],
    'units': [256],
    'regularization': [0.01],
    'learningRate': [0.001]
}

#Perform grid search using GridSearchCV
gridSearch = GridSearchCV(model, paramGrid, cv=3, scoring='accuracy')
gridSearch.fit(x, labels)

#Get the best hyperparameters
bestParams = gridSearch.best_params_
print(f"Best Hyperparameters: {bestParams}")

#Evaluate the model on the test set
bestModel = gridSearch.best_estimator_
testAcc = bestModel.score(xTest, labelsTest) 
print(f"Test Accuracy:  {testAcc*100:.2f}%")

#Plot confusion matrix
plt.figure(2)
yPred = bestModel.predict(xTest)
confMatrix = confusion_matrix(labelsTest, yPred)
sns.heatmap(confMatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")

plt.show()

